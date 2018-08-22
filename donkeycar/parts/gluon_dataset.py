import json
import os
import mxnet as mx
import numpy as np
from mxnet.gluon.data.dataloader import DataLoader
from PIL import Image
from mxnet import nd, gluon
from donkeycar import util
from donkeycar.parts.gluon_model import format_img_arr
from multiprocessing import cpu_count
import random

def get_train_val_sets(paths, train_split, batch_size):
    tub_paths = util.files.expand_path_arg(paths)
    train_json_records = []
    test_json_records = []
    for path in tub_paths:
        records = [path + '/' + f for f in os.listdir(path) if 'record_' in f]
        train_split_index = int(len(records) * train_split)
        train_json_records += records[:train_split_index]
        test_json_records += records[train_split_index:]

    train_dataset = GluonDataSet(train_json_records)
    test_dataset = GluonDataSet(test_json_records)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=cpu_count())

    return train_dataloader, test_dataloader


class GluonDataSet(gluon.data.Dataset):
    def __init__(self, record_list, load_in_memory=True, crop_chance=.1, flip_chance=.2):
        super(GluonDataSet, self).__init__()
        self.in_memory = load_in_memory
        self.crop_chance = crop_chance
        self.flip_chance = flip_chance
        if self.in_memory:
            self.json_records = []
            for record in record_list:
                self.json_records.append(self.get_record_data(record))
        else:
            self.json_records = record_list

    def __getitem__(self, item):
        record = self.json_records[item]
        if not self.in_memory:
            record = self.get_record_data(record)
        data, label = self.augment_img(*record)
        data = nd.array(format_img_arr(data.asnumpy()))
        return data, label

    def get_record_data(self, path):
        with open(path,'r') as fp:
            json_data = json.load(fp)

        base_path, file = os.path.split(path)
        img_path = json_data["cam/image_array"]
        image = np.array(Image.open(base_path + '/' + img_path))

        throttle = np.float32(json_data["user/throttle"])
        angle = np.float32(json_data["user/angle"])
        label = np.array([angle, throttle])
        return image.astype('float32'), label

    def augment_img(self, data, label):
        data = nd.array(data)
        if random.random() < self.flip_chance:
            data = nd.flip(data, axis=1)
            label[0] *= -1
        if random.random() < self.crop_chance:
            x0 = 0 if random.random() < .5 else 80
            label[0] += -1 if x0 is 0 else 1
            data = mx.image.fixed_crop(data, x0=x0, y0=0, w=80, h=120, size=(160, 120))
        return data, label

    def __len__(self):
        return len(self.json_records)