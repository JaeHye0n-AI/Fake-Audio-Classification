# 소중한 AI 경진대회

주어진 오디오가 실제 사람의 음성(real)인지 가짜 음성(fake)인지 판별하는 AI 모델 개발  

**[Private score 3rd]**

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
   ├── test
       ├── 0000.wav
       ├── 0001.wav
       └── ...
   └── submission.csv
~~~~

### Experimental settings
We conducted a total of five experiments. Each experiment is organized into separate directories.
|<center>Directory</center>|<center>Model|<center>Features</center>|<center>Notes</center>|<center>Accuracy</center>|
|:---|:---|:---|:---|:---:|
|Model_ver_1|Resnet18|Spectogram + Mel_spectogram + Chromagram|Custom augmentation (random rectangles)|90.91|
|Model_ver_2|Resnet18|Spectogram + Mel_spectogram + Chromagram|Hyperparameter tuning|88.75|
|Model_ver_3|Resnet18|Spectogram + Mel_spectogram + Chromagram|Hyperparameter tuning|90.41|
|Model_ver_4|Resnet18|Spectogram + Mel_spectogram + Chromagram|Test time adaptation (Tent)|88.41|
|Model_ver_5|Resnet18|Spectogram|Test time adaptation (Tent)|88.16|

## Usage
### feature extract
We use three features extracted from audio (spectrogram, mel spectrogram, and chromagram).
Since the process of extracting features takes a long time, we extract them in advance, save them as pkl files, and then use them.
~~~~
python make_feature_from_audio.py
~~~~

The following is performed independently for each directory (e.g., `Model_ver_1`).

### Training
By executing the script provided below, you can easily train the model.
If you want to try other options, please refer to `options.py`.

~~~~
python main.py
~~~~

### Testing
The pre-trained model can be found [here](https://drive.google.com/file/d/1tLa3NnLTFmUnd0mR9pBaHW6SBpGDCkjs/view?usp=sharing).
* Place the pre-trained model as shown below.

~~~~
├── Model_ver_1
   ├── outputs
       ├── model_ver_1
           └── best_model_ver_1.pt
├── Model_ver_2
   ├── outputs
       ├── model_ver_2
           └── best_model_ver_2.pt
└── ...
~~~~

You can test the model by running the command below.

~~~~
python main_test.py
~~~~

### Ensemble
We ensemble the five experiments conducted above. Place each experiment's result CSVs in the `ensemble_csv` directory and execute the following command.

~~~~
python ensemble_from_csv.py
~~~~

As a final result, the file `ensemble.csv` will be created in the `ensemble_csv` directory.

## References
We referenced the repo below for the code.   
Tent: Fully Test-Time Adaptation by Entropy Minimization [[paper](https://openreview.net/forum?id=uXl3bZLkr3c)] [[code](https://github.com/DequanWang/tent)]   
