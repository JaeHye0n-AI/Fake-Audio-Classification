import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
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
from torch.optim import Adam
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
import glob
import shutil
import warnings
import random
import sys

from datasets import *
from train_test import *

from options import *
from config import *
args = parse_args()
config = Config(args)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import *


os.environ["CUDA_VISIBLE_DEVICES"]= "1"

### experiment record ###
study_name = "model_ver_3"
study_path = f"{config.save_output_path}/{study_name}"

if not os.path.exists(study_path):
    os.makedirs(study_path)

save_config(config, f'{config.save_output_path}/{study_name}/config.txt')

### seed fix ###
if config.seed >= 0:
    set_seed(config.seed)
    worker_init_fn = np.random.seed(config.seed)


### train, val data prepare ###
train_loader, val_loader = mk_train_val_dataset(config.train_data_path, config.batch_size)
# train_loader = mk_train_val_dataset(config.train_data_path, config.batch_size)

### model prepare ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
model.conv1 = conv1

model = model.to(device)


### start train ###
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0)

train(model, train_loader, val_loader, device, optimizer, criterion, study_path)