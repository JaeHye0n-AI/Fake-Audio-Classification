from tqdm import tqdm
import IPython
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

import librosa
import pandas as pd
import glob
import shutil
import warnings
import random
from utils import *
import librosa
import pickle
import os
from librosa import feature

from options import *
from config import *
args = parse_args()
config = Config(args)


### train data

# make spectogram
if not os.path.exists(f"{config.train_data_path}/spectogram_feature.pkl"):
    print('preparing spectogram_train_feature')

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    warnings.filterwarnings('ignore')
    data = []
    labels = []
    folders = ['fake', 'real']  # fake: 0, real: 1

    for folder in folders:
        file_paths = glob.glob(f"{config.train_data_path}/{folder}/*.wav")
        for curr_path in tqdm(file_paths):
            audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
            real_spec = np.abs(librosa.stft(audio))
            real_spec = librosa.amplitude_to_db(real_spec, ref=np.max)
            padded_spec = pad2d(real_spec, 100)
            data.append(padded_spec)
            labels.append(folder)

    feature_df = pd.DataFrame({"features": data, "class": labels})

    feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

    ### save spectogram feature ###
    feature_df.to_pickle(f"{config.train_data_path}/spectogram_feature.pkl")

else:
    print('already spectogram_train_feature')
    feature_df = pd.read_pickle(f"{config.train_data_path}/spectogram_feature.pkl")


# make mel_spectogram
if not os.path.exists(f"{config.train_data_path}/mel_spectogram_feature.pkl"):
    print('preparing mel_spectogram_train_feature')

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    warnings.filterwarnings('ignore')
    data = []
    labels = []
    folders = ['fake', 'real']  # fake: 0, real: 1

    for folder in folders:
        file_paths = glob.glob(f"{config.train_data_path}/{folder}/*.wav")
        for curr_path in tqdm(file_paths):
            audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
            real_mel_spec = feature.melspectrogram(y=audio, sr=sample_rate)
            real_mel_spec = librosa.power_to_db(real_mel_spec, ref=np.max)
            padded_mel_spec = pad2d(real_mel_spec, 100)
            data.append(padded_mel_spec)
            labels.append(folder)


    feature_df = pd.DataFrame({"features": data, "class": labels})

    feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

    ### save mel_spectogram feature ###
    feature_df.to_pickle(f"{config.train_data_path}/mel_spectogram_feature.pkl")

else:
    print('already mel_spectogram_train_feature')
    feature_df = pd.read_pickle(f"{config.train_data_path}/mel_spectogram_feature.pkl")

# make chromagram
if not os.path.exists(f"{config.train_data_path}/chromagram_feature.pkl"):
    print('preparing chromagram_train_feature')

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    warnings.filterwarnings('ignore')
    data = []
    labels = []
    folders = ['fake', 'real']  # fake: 0, real: 1

    for folder in folders:
        file_paths = glob.glob(f"{config.train_data_path}/{folder}/*.wav")
        for curr_path in tqdm(file_paths):
            audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
            real_chroma = feature.chroma_cqt(y=audio, sr=sample_rate, bins_per_octave=36)
            padded_chroma = pad2d(real_chroma, 100)
            data.append(padded_chroma)
            labels.append(folder)

    feature_df = pd.DataFrame({"features": data, "class": labels})

    feature_df["class"] = label_encoder(feature_df["class"])  # fake: 0, real: 1

    ### save chromagram feature ###
    feature_df.to_pickle(f"{config.train_data_path}/chromagram_feature.pkl")

else:
    print('already chromagram_train_feature')
    feature_df = pd.read_pickle(f"{config.train_data_path}/chromagram_feature.pkl")



### test data

# make spectogram
if not os.path.exists(f"{config.test_data_path}/spectogram_feature.pkl"):
    print('preparing spectogram_test_feature')

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    test_data = []
    test_file_paths = sorted(glob.glob(f"{config.test_data_path}/*.wav"))
    for curr_path in tqdm(test_file_paths):
        audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
        real_spec = np.abs(librosa.stft(audio))
        real_spec = librosa.amplitude_to_db(real_spec, ref=np.max)
        padded_spec = pad2d(real_spec, 100)
        test_data.append(padded_spec)
    with open(f"{config.test_data_path}/spectogram_feature.pkl", "wb") as f:
        pickle.dump(test_data, f)

else:
    print('already spectogram_test_feature')
    with open(f"{config.test_data_path}/spectogram_feature.pkl", "rb") as f:
        test_data = pickle.load(f)


# make mel_spectogram
if not os.path.exists(f"{config.test_data_path}/mel_spectogram_feature.pkl"):
    print('preparing mel_spectogram_test_feature')

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    test_data = []
    test_file_paths = sorted(glob.glob(f"{config.test_data_path}/*.wav"))
    for curr_path in tqdm(test_file_paths):
        audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
        real_mel_spec = feature.melspectrogram(y=audio, sr=sample_rate)
        real_mel_spec = librosa.power_to_db(real_mel_spec, ref=np.max)
        padded_mel_spec = pad2d(real_mel_spec, 100)
        test_data.append(padded_mel_spec)
    with open(f"{config.test_data_path}/mel_spectogram_feature.pkl", "wb") as f:
        pickle.dump(test_data, f)

else:
    print('already mel_spectogram_test_feature')
    with open(f"{config.test_data_path}/mel_spectogram_feature.pkl", "rb") as f:
        test_data = pickle.load(f)


# make chromagram
if not os.path.exists(f"{config.test_data_path}/chromagram_feature.pkl"):
    print('preparing chromagram_test_feature')

    pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))

    test_data = []
    test_file_paths = sorted(glob.glob(f"{config.test_data_path}/*.wav"))
    for curr_path in tqdm(test_file_paths):
        audio, sample_rate = librosa.load(curr_path, res_type="kaiser_fast")
        real_chroma = feature.chroma_cqt(y=audio, sr=sample_rate, bins_per_octave=36)
        padded_chroma = pad2d(real_chroma, 100)
        test_data.append(padded_chroma)
    with open(f"{config.test_data_path}/chromagram_feature.pkl", "wb") as f:
        pickle.dump(test_data, f)

else:
    print('already chromagram_test_feature')
    with open(f"{config.test_data_path}/chromagram_feature.pkl", "rb") as f:
        test_data = pickle.load(f)

