import argparse
import os

def parse_args():
    descript = 'Pytorch Implementation of \'fake_audio\''
    parser = argparse.ArgumentParser(description=descript)

    parser.add_argument('--train_data_path', default='../dataset/train')
    parser.add_argument('--test_data_path', default='../dataset/test')
    parser.add_argument('--submission_path', default='../dataset/submission.csv')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--num_iters', type=int, default=10)

    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--save_output_path', type=str, default='./outputs')
    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.save_output_path):
        os.makedirs(args.save_output_path)

    return args
