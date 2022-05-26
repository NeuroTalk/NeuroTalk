# NeuroTalk: Voice Reconstruction from Brain Signals
This repository is the official implementation of NeuroTalk: Voice Reconstruction from Brain Signals

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```


## Training
To train the model in the paper, run this command:
```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
>📋 the arguments of models

## Evaluation
To evaluate my model on an example data, run:

```eval
python eval.py --model-file mymodel.pth --benchmark example
```

## Pre-trained Models

You can download pretrained models here:
- [Pretrained model](https://drive.google.com/mymodel.pth) trained on participant 1


## Results
Our model achieves the following performance on :

## Contributing
- We generated voice from the EEG of imagined speech. The fundamental constraint of the imagined speech-based BTS system lacking the ground truth voice have been addressed with generalized EEG of spoken speech to link the imagined speech EEG, spoken speech EEG, and the spoken speech audio.

- We propose a generative model based on multi-receptive residual modules with recurrent neural networks that can extract frequency characteristics and sequential information from neural signals, to enhance the generation of mel-spectrogram of the user's own voice from non-invasive brain signals.

- Character loss was utilized to a large extent to adapt various phonemes from the small amount of data. Therefore, unseen words of the imagined speech were able to be reconstructed from the pre-trained model. This implies that our model trained the character level information from the brain signal, which displays the potential of phoneme prediction using a small dataset of few words or phrases. 


<!--
**NeuroTalk/NeuroTalk** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
