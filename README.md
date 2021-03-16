# Contrastive-Regularization-MMER
codes of paper "Contrastive Regularization for Multimodal Emotion Recognition Using Audio and Text"
## 1. First check required python packages:
* Python 3.6.5
* torch 0.4.0
* numpy 1.17.0
* gensim 3.8.3
* deepspeech 0.5.1
## 2. Dataset preprocessing
1) use mocap_data_collect.py to convert raw IEMOCAP dataset to dictionary format
2) choose 4-class emotion, 5531 samples
3) get transcriptions using gensim and deepspeech packages
4) extract linguistic and acoustic features
## 3. Run run_training.py to acquire linguistic and acoustic emotion recognition models
## 4. Run run_training_ensemble.py /path to acquire ensemble model, i.e., baseline model
## 5. Run run_training_ensemble_cl.py /path to acquire model which adds contrastive regularization
