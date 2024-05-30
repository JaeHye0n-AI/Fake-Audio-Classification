import numpy as np
import os


class Config(object):
    def __init__(self, args):

        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.submission_path = args.submission_path

        self.batch_size = args.batch_size
        self.seed = args.seed
        self.num_iters = args.num_iters

        self.num_workers = args.num_workers

        self.save_output_path = args.save_output_path


    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')
