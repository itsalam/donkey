"""
The Gluon DataLoader method and Dataset Class for the MxNet Variation for the DonkeyCar.
@Author Vincent Lam - @vlamai
"""

import json
import os
import mxnet as mx
import numpy as np
import re
from mxnet.gluon.data.dataloader import DataLoader
from PIL import Image
from mxnet import nd, gluon
from donkeycar import util
from donkeycar.parts.gluon_model import format_img_arr
from multiprocessing import cpu_count
from donkeycar.util.data import linear_bin
import random


def get_train_val_sets(paths, train_split, batch_size=128, flip_chance=.2, crop_chance=.1, recover_turn=.5):
    """
    Method that returns training DataLoader and test DataLoader objects.
    :param str paths: A string of paths to collect data from, seperated by commas
    :param float train_split: A value from 0-1 for the ratio of training data files to test data files.
    :param int batch_size: The batch size to instantiate the DataLoader objects with.
    :param float flip_chance: A percentage chance for the image to be flipped when retrieved.
    :param float crop_chance: A percentage chance for the image to be cropped horizontally in half.
    :param float recover_turn: Value to add the angular value to when cropped in half.
    :returns Tuple(DataLoader, DataLoader): The training DataLoader, and the test DataLoader
    """
    tub_paths = util.files.expand_path_arg(paths)
    train_json_records = []
    test_json_records = []
    for path in tub_paths:
        records = [path + '/' + f for f in os.listdir(path) if 'record_' in f]
        train_split_index = int(len(records) * train_split)
        train_json_records += records[:train_split_index]
        test_json_records += records[train_split_index:]

    def augment_img(data, label):
        """
        Method that returns training DataLoader and test DataLoader objects.
        :param nd data: The image data from the sample
        :param nd label: The label from the sample
        :returns Tuple(nd,nd): The adjusted data and label.
        """
        data = nd.array(data)
        if random.random() < flip_chance:
            data = nd.flip(data, axis=1)
            label[0] *= -1
        if random.random() < crop_chance:
            x0 = 0 if random.random() < .5 else 80
            label[0] += 1*recover_turn if x0 is 0 else -1*recover_turn
            data = mx.image.fixed_crop(data, x0=x0, y0=0, w=80, h=120, size=(160, 120))
        label = np.clip(label, -1, 1)
        return data, label

    train_dataset = GluonDataSet(train_json_records, transform=augment_img)
    test_dataset = GluonDataSet(test_json_records)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=cpu_count())

    return train_dataloader, test_dataloader


class GluonDataSet(gluon.data.Dataset):
    """
    An implementation of the Gluon Dataset base class. Holds either a list of pathway strings or actual data in memory.
    """
    def __init__(self, record_list, load_in_memory=True, transform=None):
        """
        The default constructor that requires a list of directories to the .json record it uses.
        :param list record_list: The list of directories to the .json records
        :param bool load_in_memory: The bool to determine whether the data will be full loaded into data
        :param function transform: The function to modify the sample with if given.
        """
        super(GluonDataSet, self).__init__()
        self._in_memory = load_in_memory
        self._transform = transform
        if self._in_memory:
            self._json_records = []
            for record in record_list:
                self._json_records.append(self._load_record_data(record))
        else:
            self._json_records = record_list

    def __getitem__(self, item):
        """
        Retrieves the data from the item'th index.
        :param int item: the index to retrieve the sample.
        :returns Tuple(nd,nd): the sample of data.
        """
        record = self._json_records[item]
        if not self._in_memory:
            record = self._load_record_data(record)
        if self._transform is not None:
            record = self._transform(*record)
        data = nd.array(format_img_arr(record[0]))
        record[1][0] = np.argmax(linear_bin(record[1][0]))
        return data, record[1]

    @staticmethod
    def _load_record_data(path):
        """
        Retrieves the .json data, given the directory
        :param str path: the path to retrieve the sample.
        :returns Tuple(np,np): the sample of data.
        """
        with open(path, 'r') as fp:
            json_data = json.load(fp)
        base_path, file = os.path.split(path)
        img_path = json_data["cam/image_array"]
        last_num = re.compile(r'(?:(\d*))+')
        number = last_num.search(img_path)
        next_num = str(int(number.group(0)) - 1)
        start, end = number.span(1)
        last_img_path = img_path[:max(end - len(number.group(1)), start)] + next_num + img_path[end:]
        try:
            img = Image.open(base_path + '/' + last_img_path)
        except FileNotFoundError:
            img = Image.open(base_path + '/' + img_path)
        image_arr = np.array(img).astype('float32')

        throttle = np.float32(json_data["user/throttle"])
        angle = np.float32(json_data["user/angle"])
        label = np.array([angle, throttle])
        return image_arr, label

    def __len__(self):
        """
        States the length of samples
        :returns int: the length of data samples.
        """
        return len(self._json_records)
