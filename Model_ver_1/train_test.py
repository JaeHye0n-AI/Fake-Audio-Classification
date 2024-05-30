import copy

import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
import glob
import shutil
import warnings
import random
from utils import *
import cv2
from torch.optim import Adam
from torchvision import datasets, transforms

from options import *
from config import *

args = parse_args()
config = Config(args)

def aug_1(data): #random rectangles

    aug_data = copy.deepcopy(data)

    for i in range(len(aug_data)):
        num_rectangles = random.randint(10, 40)

        for _ in range(num_rectangles):
            # Generate random parameters for the rectangle
            rect_width = random.randint(1, 5)
            rect_height = random.randint(1, 5)
            start_x = random.randint(0, 100 - rect_width)
            start_y = random.randint(0, 40 - rect_height)

            # Draw the rectangle on the image
            end_x = start_x + rect_width
            end_y = start_y + rect_height
            aug_data[i][start_y:end_y, start_x:end_x] = 1.0
            # cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (255), -1)  # -1 fills the rectangle

    return aug_data


def train(model, train_loader, val_loader,device, optimizer, criterion, study_path):
    for epoch in range(config.num_iters):
        model.train()
        preds = []
        labels = []
        running_loss = 0.0

        best_acc = -1

        for inputs, label in tqdm(train_loader):
            # Expand input tensor to [N, C, H, W]

            inputs = torch.unsqueeze(inputs, 1)

            inputs = inputs.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            preds += predicted.detach().cpu().numpy().tolist()
            labels += label.detach().cpu().numpy().tolist()

        train_accuracy = accuracy_score(labels, preds)
        print(f'train_accuracy: {train_accuracy}')

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, label in tqdm(val_loader):
                # Expand input tensor to [N, C, H, W]
                inputs = torch.unsqueeze(inputs, 1)
                inputs = inputs.to(device)
                label = label.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_preds += predicted.detach().cpu().numpy().tolist()
                val_labels += label.detach().cpu().numpy().tolist()

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f'val_accuracy: {val_accuracy}')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f'{study_path}/{epoch}ep_ACC_{val_accuracy:.4f}.pt')
            shutil.copyfile(f'{study_path}/{epoch}ep_ACC_{val_accuracy:.4f}.pt',
                            f'{study_path}/best_model_ver_1.pt')


def test(model, test_loader, device, study_path, test_df, pretrained_model = None):

    model.eval()
    test_predictions = []

    with torch.no_grad():
        for inputs in tqdm(test_loader):
            # Expand input tensor to [N, C, H, W]
            inputs = torch.unsqueeze(inputs[0], 1)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_predictions += predicted.detach().cpu().numpy().tolist()

    test_df['label'] = test_predictions

    test_df.to_csv(f'{study_path}/test_submission_ver_1.csv', index=False)

# def test_time_adap(model, test_loader, optimizer, device, study_path, test_df, pretrained_model = None):
#
#
#     # model.eval()
#     test_predictions = []
#
#     optimizer = Adam(model.parameters(), lr=0.0001)
#
#     with torch.no_grad():
#         for inputs in tqdm(test_loader):
#             # Expand input tensor to [N, C, H, W]
#             inputs = torch.unsqueeze(inputs[0], 1)
#             inputs = inputs.to(device)
#
#             model = tent.configure_model(model)
#             params, param_names = tent.collect_params(model)
#             optimizer = Adam(params, lr=0.0001)
#             tented_model = tent.Tent(model, optimizer)
#
#             outputs = tented_model(inputs)  # now it infers and adapts!
#
#             _, predicted = torch.max(outputs.data, 1)
#             test_predictions += predicted.detach().cpu().numpy().tolist()
#
#     test_df['label'] = test_predictions
#
#     test_df.to_csv(f'{study_path}/test_submission.csv', index=False)


