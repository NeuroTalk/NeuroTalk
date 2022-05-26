import torch
from modules import DTW_align
from utils import data_denorm
import torch.nn.functional as F
import time
import torch.optim.lr_scheduler
import numpy as np
import torchaudio

    
def eval(args, train_loader, models, criterions, optimizers, epoch, trainValid=False):
    '''
    :param args: general arguments
    :param train_loader: loaded for training/validation/test dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    
    # switch to train mode
    if type(models) != tuple:
        print("Two models should be inputed (generator and discriminator)")
        
    (model, model_cl, vocoder, model_STT, decoder_STT) = models
    (criterion, criterion_cl, criterion_adv, CER, WER) =  criterions
    
    if trainValid:
        (optimizer,optimizer_cl) = optimizers
        
    if trainValid:
        model.train()
        model_cl.train()
        vocoder.train()
        model_STT.train()
    else:
        model.eval()
        model_cl.eval()
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
        
        if args.onGPU:
            input = input.cuda()
            target = target.cuda()
            target_cl = target_cl.cuda()
            voice = torch.squeeze(voice,dim=-1).cuda()
        labels = torch.argmax(target_cl,dim=1) 

        ###############################
        # Train Generator
        ###############################
        if trainValid:
            for p in model.parameters():
                p.requires_grad_(True)  # unfreeze G
            for p in model_cl.parameters():
                p.requires_grad_(False)  # freeze D
                
            # set zero grad    
            optimizer.zero_grad()
            
            # run models
            output = model(input)
            g_valid, out_cl = model_cl(output)
            
        else:
            # model = model.weight_norm()
            with torch.no_grad():
                output = model(input)
                g_valid, out_cl = model_cl(output)
        
        # when not overt, DTW is needed
        out_DTW = output.clone()
        if args.task[0] == 'I' and epoch > 100: 
            out_DTW = DTW_align(out_DTW, target)
        
        # generator loss
        loss1 = criterion(out_DTW, target)
        
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

        
        if trainValid:
            loss_g.backward() 
            optimizer.step()
            
            
        ###############################
        # Train Discriminator
        ###############################
        if trainValid:
            for p in model.parameters():
                p.requires_grad_(False)  # freeze G
                
            if args.pretrained and args.prefreeze:
                for total_ct, _ in enumerate(model_cl.children()):
                    ct=0
                for ct, child in enumerate(model_cl.children()):
                    if ct > total_ct-1: # unfreeze classifier 
                        for param in child.parameters():
                            param.requires_grad = True  # unfreeze D    
            else:
                for p in model_cl.parameters():
                    p.requires_grad_(True)  # unfreeze D   
                    
                
            # set zero grad
            optimizer_cl.zero_grad()
        
            # run model cl
            real_valid = model_cl(target)
            fake_valid = model_cl(out_DTW.detach())
        else:
            with torch.no_grad():
                real_valid = model_cl(target)
                fake_valid = model_cl(out_DTW.detach())
        
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
        
        if trainValid:
            loss_d.backward()
            optimizer_cl.step()

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
