# 소중한 AI 경진대회

주어진 오디오가 실제 사람의 음성(real)인지 가짜 음성(fake)인지 판별하는 AI 모델 개발

### Environment
* Python == 3.8.0
* Pytorch == 1.7.1+cu110
* CUDA == 12.2
* scipy
* pandas
* joblib
* tqdm
* librosa
* seaborn


### Data Preparation
1. The dataset can be downloaded from this [link](https://www.kaggle.com/competitions/hbnu-fake-audio-detection-competition).
   
2. Place the dataset inside the `dataset` directory.
   * Please ensure the data structure is as below.
   
~~~~
├── dataset
   ├── train
       ├── real
           ├── 0000.wav
           ├── 0002.wav
           └── ...
       └── fake
           ├── 0001.wav
           ├── 0003.wav
           └── ...
   └── test
       ├── 0000.wav
       ├── 0001.wav
       └── ...
~~~~

### Experimental settings

|Foler name|Model|Features|Notes|Accuracy|
|:---|:---|:---|:---|:---:|
|Model_ver_1|Resnet18|Spectogram + Mel_spectogram + Chromagram|Custom augmentation (random rectangles)|90.91|
|Model_ver_2|Resnet18|Spectogram + Mel_spectogram + Chromagram|Hyperparameter tuning|88.75|
|Model_ver_3|Resnet18|Spectogram + Mel_spectogram + Chromagram|Hyperparameter tuning|90.41|
|Model_ver_4|Resnet18|Spectogram + Mel_spectogram + Chromagram|Hyperparameter tuning + Test time adaptation|88.41|
|Model_ver_5|Resnet18|Spectogram|Test time adaptatio|88.16|

## Usage
### feature extract
We use three features extracted from audio (spectrogram, mel spectrogram, and chromagram).
Since the process of extracting features takes a long time, we extract them in advance, save them as pkl files, and then use them.
~~~~
make_feature_from_audio.py
~~~~

### Training
By executing the script provided below, you can easily train the model.

If you want to try other options, please refer to `options.py`.

~~~~
python main.py
~~~~

### Testing
The pre-trained model can be found [here](https://drive.google.com/file/d/1ybT3-Syq_BeLZRaX2ptI-XmgVuadeJZV/view?usp=sharing).
You can test the model by running the command below.
* Place the pre-trained model in the `models` directory.

~~~~
python main_eval.py
~~~~

## References
We referenced the repo below for the code.   
Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization (ICCV 2021) [[paper](https://arxiv.org/abs/2108.05029)] [[code](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points)]   
