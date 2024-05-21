import pandas as pd
import os
from options import *
from config import *


args = parse_args()
config = Config(args)

test_df = pd.read_csv(config.submission_path)

# folder path
folder_path = './ensemble_csv'

# all CSV file in folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

data_frames = []


for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, index_col=0)
    data_frames.append(df)

# concatenate all dataframe
combined_df = pd.concat(data_frames, axis=1)

test_predictions = []

row_count = combined_df.shape[1]
same_num = 0
for _, row in combined_df.iterrows():
    row_sum = row.sum()
    row_avg = row_sum / row_count
    if row_avg == 0.5:
        same_num += 1
    if row_avg > 0.5:
        test_predictions.append(1)
    else:
        test_predictions.append(0)


# save
test_df['label'] = test_predictions

test_df.to_csv(f'{folder_path}/ensemble.csv', index=False)
