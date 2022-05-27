import os
import torch
from models import models as networks
from models.models_HiFi import Generator as model_HiFi
from modules import DTW_align, GreedyCTCDecoder, AttrDict, RMSELoss
from utils import data_denorm, audio_denorm, mel2wav_vocoder, perform_STT_bundle, imgSave
import torch.nn as nn
import torch.nn.functional as F
from myDataset import myDataset
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio
from torchmetrics import CharErrorRate, WordErrorRate
import json
import argparse
    
def train(args, train_loader, models, criterions, optimizers, epoch, trainValid=True):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: losses
    '''
    
    # switch to train mode
    assert type(models) != tuple, "More than two models should be inputed (generator and discriminator)"
        
    (model_g, model_d, vocoder, model_STT, decoder_STT) = models
    (criterion_recon, criterion_adv, CER, WER) =  criterions
        
    if trainValid:
        (optimizer_g, optimizer_d) = optimizers
        
    if trainValid:
        model_g.train()
        model_d.train()
        vocoder.train()
        model_STT.train()
    else:
        model_g.eval()
        model_d.eval()
        vocoder.eval()
        model_STT.eval()
    
    
    epoch_loss_g = []
    epoch_loss_g_recon = []
    epoch_loss_g_valid = []
    epoch_loss_d = []
    epoch_loss_d_valid = []
    
    epoch_acc_g_valid=[]
    epoch_cer_gt = []
    epoch_cer_recon = []
    epoch_acc_d_real = []
    epoch_acc_d_fake = []

    total_batches = len(train_loader)
    
    for i, (input, target, target_cl, voice, data_info) in enumerate(train_loader):    
        start_time = time.time()
        

        print("\rBatch [%5d / %5d]"%(i,total_batches), sep=' ', end='', flush=True)
        
        # Adversarial ground truths 1:real, 0: fake
        valid = torch.ones((len(input), 1), dtype=torch.float32).cuda()
        fake = torch.zeros((len(input), 1), dtype=torch.float32).cuda()
        
        input = input.cuda()
        target = target.cuda()
        target_cl = target_cl.cuda()
        voice = torch.squeeze(voice,dim=-1).cuda()
        labels = torch.argmax(target_cl,dim=1) 

        ###############################
        # Train Generator
        ###############################
        if trainValid:
            for p in model_g.parameters():
                p.requires_grad_(True)  # unfreeze G
            for p in model_d.parameters():
                p.requires_grad_(False)  # freeze D
                
            # set zero grad    
            optimizer_g.zero_grad()
            
            # run models
            output = model_g(input)
            g_valid = model_d(output)
        else:
            with torch.no_grad():
                output = model_g(input)
                g_valid = model_d(output)
            
        # when not overt, DTW is needed
        out_DTW = output.clone()
        if args.task[0] == 'I' and epoch > 100: 
            out_DTW = DTW_align(out_DTW, target)
        
        # generator loss
        loss1 = criterion_recon(out_DTW, target)
        
        # GAN loss
        loss_valid = criterion_adv(g_valid,valid)
        
        
        ###############################
        # Loss from Vocoder - STT
        ###############################
        if trainValid:
            for p in vocoder.parameters():
                p.requires_grad_(False)  # freeze vocoder
            for p in model_STT.parameters():
                p.requires_grad_(False)  # freeze model_STT
        
        # out_DTW
        target_denorm = data_denorm(target, data_info[0], data_info[1])
        output_denorm = data_denorm(out_DTW, data_info[0], data_info[1])
        
        gt_label=[]
        for j in range(len(target)):
            gt_label.append(args.word_label[labels[j].item()])

        # target
        ##### HiFi-GAN
        wav_target = vocoder(target_denorm)
        wav_target = torch.reshape(wav_target, (len(wav_target),wav_target.shape[-1]))
        
        #### resampling
        wav_target = torchaudio.functional.resample(wav_target, args.sample_rate_mel, args.sample_rate_STT)   
        if wav_target.shape[1] !=  voice.shape[1]:
            p = voice.shape[1] - wav_target.shape[1]
            p_s = p//2
            p_e = p-p_s
            wav_target = F.pad(wav_target, (p_s,p_e))
        
        # recon
        ##### HiFi-GAN
        wav_recon = vocoder(output_denorm)
        wav_recon = torch.reshape(wav_recon, (len(wav_recon),wav_recon.shape[-1]))
        
        #### resampling
        wav_recon = torchaudio.functional.resample(wav_recon, args.sample_rate_mel, args.sample_rate_STT)   
        if wav_recon.shape[1] !=  voice.shape[1]:
            p = voice.shape[1] - wav_recon.shape[1]
            p_s = p//2
            p_e = p-p_s
            wav_recon = F.pad(wav_recon, (p_s,p_e))

        ##### STT Wav2Vec 2.0
        emission_gt, _ = model_STT(voice)
        emission_recon, _ = model_STT(wav_recon)
       
        # decoder STT
        transcript_gt = []
        transcript_recon = []

        for j in range(len(voice)):
            transcript = decoder_STT(emission_gt[j])   
            transcript_gt.append(transcript)
                
            transcript = decoder_STT(emission_recon[j])
            transcript_recon.append(transcript)

        cer_gt = CER(transcript_gt, gt_label)
        cer_recon = CER(transcript_recon, gt_label)

        # total generator loss
        loss_g = args.l_g[0] * loss1 + args.l_g[2] * loss_valid + args.l_g[3] * cer_recon 
        
        # accuracy
        acc_g_valid = (g_valid.round() == valid).float().mean()
        
        epoch_loss_g.append(loss_g.item())
        epoch_loss_g_recon.append(loss1.item())
        epoch_loss_g_valid.append(loss_valid.item())
        epoch_acc_g_valid.append(acc_g_valid.item())
        
        if torch.isnan(cer_gt):
            epoch_cer_gt.append(np.array([0]))
        else:
            epoch_cer_gt.append(cer_gt.item())

        if torch.isnan(cer_recon):
            epoch_cer_recon.append(np.array([0]))
        else:
            epoch_cer_recon.append(cer_recon.item())

        
        loss_g.backward() 
        optimizer_g.step()
            
        ###############################
        # Train Discriminator
        ###############################
        for p in model_g.parameters():
            p.requires_grad_(False)  # freeze G
            
        if args.pretrained and args.prefreeze:
            for total_ct, _ in enumerate(model_d.children()):
                ct=0
            for ct, child in enumerate(model_d.children()):
                if ct > total_ct-1: # unfreeze classifier 
                    for param in child.parameters():
                        param.requires_grad = True  # unfreeze D    
        else:
            for p in model_d.parameters():
                p.requires_grad_(True)  # unfreeze D   
                    
                
        # set zero grad
        optimizer_d.zero_grad()
    
        # run model cl
        real_valid = model_d(target)
        fake_valid = model_d(out_DTW.detach())

        
        loss_d_real_valid = criterion_adv(real_valid, valid)
        loss_d_fake_valid = criterion_adv(fake_valid, fake)
        
        
        loss_d_valid = 0.5 * (loss_d_real_valid + loss_d_fake_valid)
        
        loss_d = loss_d_valid
        
        # accuracy
        acc_d_valid = (real_valid.round() == valid).float().mean()
        acc_d_fake = (fake_valid.round() == fake).float().mean()
        
        epoch_loss_d.append(loss_d.item())
        epoch_loss_d_valid.append(loss_d_valid.item())
        epoch_acc_d_real.append(acc_d_valid.item())
        epoch_acc_d_fake.append(acc_d_fake.item())
        
        loss_d.backward()
        optimizer_d.step()

        time_taken = time.time() - start_time

        
    args.loss_g = sum(epoch_loss_g) / len(epoch_loss_g)
    args.loss_g_recon = sum(epoch_loss_g_recon) / len(epoch_loss_g_recon)
    args.loss_g_valid = sum(epoch_loss_g_valid) / len(epoch_loss_g_valid)
    args.acc_g_valid = sum(epoch_acc_g_valid) / len(epoch_acc_g_valid)
    args.cer_gt = sum(epoch_cer_gt) / len(epoch_cer_gt)
    args.cer_recon = sum(epoch_cer_recon) / len(epoch_cer_recon)
    
    args.loss_d = sum(epoch_loss_d) / len(epoch_loss_d)
    args.loss_d_valid = sum(epoch_loss_d_valid) / len(epoch_loss_d_valid)
    args.acc_d_real = sum(epoch_acc_d_real) / len(epoch_acc_d_real)
    args.acc_d_fake = sum(epoch_acc_d_fake) / len(epoch_acc_d_fake)


    print('\n[%3d/%3d] G_valid: %.4f D_R: %.4f D_F: %.4f / CER-gt: %.4f CER-recon: %.4f / g-RMSE: %.4f g-losscl: %.4f g-lossValid: %.4f Time: %.4f' 
          % (i, total_batches, 
             args.acc_g_valid, args.acc_d_real, args.acc_d_fake, 
             args.cer_gt, args.cer_recon, 
             args.loss_g_recon, args.loss_g_cl, args.loss_g_valid, time_taken))
    
    return args


def save_checkpoint(state, is_best, save_path, filename):
    """
    Save model checkpoint.
    :param state: model state
    :param is_best: is this checkpoint the best so far?
    :param save_path: the path for saving
    """
    
    torch.save(state, os.path.join(save_path, filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(save_path, 'BEST_' + filename))


def main(args):
    
    torch.cuda.set_device(args.device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
    print(torch.cuda.device_count())
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # define generator
    config_file = os.path.join(os.path.split(args.HiFiGRU_config)[0], 'config_g.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_g = AttrDict(json_config)

    model_g = networks.Generator(h_g).cuda()
    
    # define discriminator
    config_file = os.path.join(os.path.split(args.HiFiGRU_config)[0], 'config_d.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h_d = AttrDict(json_config)

    model_d = networks.Discriminator(h_d).cuda()
    
    # vocoder HiFiGAN
    # LJ_FT_T2_V3/generator_v3,   
    config_file = os.path.join(os.path.split(args.vocoder_pre)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    vocoder = model_HiFi(h).cuda()
    state_dict_g = torch.load(args.vocoder_pre) #, map_location=args.device)
    vocoder.load_state_dict(state_dict_g['generator'])
    
    # STT Wav2Vec
    bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
    model_STT = bundle.get_model().cuda()
    decoder_STT = GreedyCTCDecoder(labels=bundle.get_labels())
    
    # Parallel setting
    model_g = nn.DataParallel(model_g, device_ids=args.num_GPUs)
    model_d = nn.DataParallel(model_d, device_ids=args.num_GPUs)
    vocoder = nn.DataParallel(vocoder, device_ids=args.num_GPUs)
    model_STT = nn.DataParallel(model_STT, device_ids=args.num_GPUs)
    
    if args.pretrained:
        if os.path.isfile(args.pretrainedLoc):
            # pretrianed model
            print("=> loading checkpoint '{}'".format(args.pretrainedLoc))
            checkpoint = torch.load(args.pretrainedLoc, map_location='cpu')
            checkpoint_cl = torch.load(args.pretrainedLoc_cl, map_location='cpu')
            # args = checkpoint['args']
            model_g.load_state_dict(checkpoint['state_dict'])
            model_d.load_state_dict(checkpoint_cl['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no Pretraoned model found at '{}'".format(args.pretrained))
    
    start_epoch = 0
    if args.resume:  
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resumeLoc))
            checkpoint = torch.load(args.resumeLoc, map_location='cpu')
            checkpoint_cl = torch.load(args.resumeLoc_cl, map_location='cpu')
            start_epoch = checkpoint['epoch']
            # args = checkpoint['args']
            model_g.load_state_dict(checkpoint['state_dict'])
            model_d.load_state_dict(checkpoint_cl['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion_recon = RMSELoss().cuda()
    criterion_adv = nn.BCELoss().cuda()
    CER = CharErrorRate().cuda()
    WER = WordErrorRate().cuda()
    
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    optimizer_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr_g, betas=(0.8, 0.99), weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(model_d.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_g_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.lr_d_decay, last_epoch=-1)
    
    
    # Data loader define
    generator = torch.Generator().manual_seed(args.seed)
    trainset = myDataset(mode=0, data=args.savedata,task=args.task,recon=args.recon)  # file='./EEG_EC_Data_csv/train.txt'
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, pin_memory=True)

    lr_g = 0
    lr_d = 0
    epoch = start_epoch
    is_best = False
    
    for epoch in range(start_epoch, args.max_epochs):
        start_time = time.time()
        scheduler_g.step(epoch)
        scheduler_d.step(epoch)
        
        for param_group in optimizer_g.param_groups:
            lr_g = param_group['lr']
        for param_group in optimizer_d.param_groups:
            lr_d = param_group['lr']
        
        print("Epoch : %d/%d" %(epoch, args.max_epochs) )
        print("Learning rate for G: %.9f" %lr_g)
        print("Learning rate for D: %.9f" %lr_d)

        args = train(args, train_loader, 
                     (model_g, model_d, vocoder, model_STT, decoder_STT), 
                     (criterion_recon, criterion_adv, CER, WER), 
                     (optimizer_g, optimizer_d), 
                     epoch) 

        # Save checkpoint
        state_g = {'epoch': epoch + 1,
                   'arch': str(model_g),
                   'state_dict': model_g.state_dict(),
                   'lossGTr': args.loss_g,
                   'lossGTs': args.loss_g,
                   'lossGVal': args.loss_g,
                   ' lr': lr_g}
        
        state_d = {'epoch': epoch + 1,
                   'arch': str(model_d),
                   'state_dict': model_d.state_dict(),
                   'lossDTr': args.loss_d,
                   'lossDTs': args.loss_d,
                   'lossDVal': args.loss_d,
                   ' lr': lr_d}

        
        save_checkpoint(state_g, is_best, args.savedir, 'checkpoint_g.pth.tar')
        save_checkpoint(state_d, is_best, args.savedir, 'checkpoint_d.pth.tar')
        
        time_taken = time.time() - start_time
        print("Time: %.2f\n"%time_taken)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('trained_model', help='config & checkpoint for G & D folder path')
    parser.add_argument('vocoder_pre', help='pretrained vocoder file path')
    parser.add_argument('--gpuNum', type=int, default=[0])
    parser.add_argument('--save', type=str, default='./')
    parser.add_argument('--dataLoc', type=str, default='./sample_data/')
    parser.add_argument('--task', type=str, default='SpokenEEG_vec')
    parser.add_argument('--recon', type=str, default='Voice_mel')
    args = parser.parse_args()

    main(args)        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

