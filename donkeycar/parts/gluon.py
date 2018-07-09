import mxnet as mx
from mxnet import nd, autograd, gluon


from donkeycar import util
import os.path
from os.path import isfile
from os import listdir
import json

import numpy as np

import re

import time

class GluonPilot:

    def __init__(self):
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        self.net = None
        self.throttle_output = False

    def load(self, path):
        if os.path.isdir(path):
            p, folder_name = os.path.split(path)
            path += '/' + folder_name + '-0000.params'
            self.net.load_params(path, self.ctx)
            print('Sucessfully loaded ', folder_name)
        else:
            print('Folder not found.')

    def save(self, path):
        while os.path.exists(path):
            print("Existing folder found, creating a new one ...")

            path, folder_name = os.path.split(path)
            copy_num = folder_name.find('_')
            if copy_num != -1:
                last_num = re.compile(r'(?:[^\d]*(\d+)[^\d]*)+')
                number = last_num.search(folder_name)
                if number:
                    next_num = str(int(number.group(1)) + 1)
                    start, end = number.span(1)
                    folder_name = folder_name[:max(end - len(next_num), start)] + next_num + folder_name[end:]
            else:
                folder_name += "_1"
            path = path + '/' + folder_name

        print("New folder made at: ", path)
        os.makedirs(path)
        self.net.export(path + '/' + os.path.basename(path))

    def compile_model(self, loss=None, optimizer='sgd', learning_rate=1e-3, init_magnitude=2.24):
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=init_magnitude), ctx=self.ctx)
        self.loss = gluon.loss.L2Loss() if loss is None else loss
        self.optimizer = mx.gluon.Trainer(self.net.collect_params(), optimizer, {'learning_rate': learning_rate})

    def evalulate_accuracy(self, data_iterator):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.net(data)
            acc.update(output, label)
        return acc.get()[1]

    def predict(self, img_arr):
        output = self.net(img_arr).asnumpy()
        return output[0]

    def train(self, train_gen, val_gen, saved_model_path, epochs=100, steps=100, train_split=0.8):

        train_dataload = mx.gluon.data.DataLoader(train_gen, batch_size=128, shuffle=True)
        test_dataload = mx.gluon.data.DataLoader(val_gen, batch_size=128)

        smoothing_constant = .01
        old_loss = float('Inf')
        for epoch_index in range(epochs):
            for i, (data, label) in enumerate(train_dataload):
                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)
                with autograd.record(train_mode=True):
                    output = self.net(data)
                    loss = self.loss(output, label)

                loss.backward()
                self.optimizer.step(data.shape[0])

                current_loss = nd.mean(loss).asscalar()
                moving_loss = (current_loss if ((i == 0) and (epoch_index == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant * current_loss))
            test_accuracy = self.evalulate_accuracy(test_dataload)
            train_accuracy = self.evalulate_accuracy(train_dataload)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (
                epoch_index, moving_loss, train_accuracy, test_accuracy))
            if old_loss < moving_loss:
                break
            old_loss = moving_loss

        self.save(saved_model_path)


class GluonLinear(GluonPilot):
    def __init__(self, net=None):
        super(GluonLinear, self).__init__()
        if net:
            self.net = net
        else:
            self.net = default_linear()
        self.compile_model()
        self.elapsed = 0.0
        self.run_count = 0

    def run(self, img_arr):
        start = time.time()
        img_arr = nd.array(img_arr).expand_dims(axis=0).transpose(axes=(0, 3, 1, 2))
        output = self.predict(img_arr)
        # print('throttle', throttle)
        # angle_certainty = max(angle_binned[0])
        self.elapsed += time.time() - start
        self.run_count += 1
        if self.run_count % 100 == 99:
            print(self.elapsed / self.run_count)
        return output[0], output[1]


def default_linear():
    net = gluon.nn.HybridSequential(prefix='')
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=24, kernel_size=5, strides=(2, 2), activation='relu'))
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, strides=(2, 2), activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=5, strides=(2, 2), activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(2, 2), activation='relu'))
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(1, 1), activation='relu'))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(100, activation='relu'))
        net.add(gluon.nn.Dropout(.1))
        net.add(gluon.nn.Dense(50, activation='relu'))
        net.add(gluon.nn.Dropout(.1))

        net.add(gluon.nn.Dense(2))

    net.hybridize()
    return net


# Helper function to convert the generator of batches into a list

def load_data_in_path(paths):
    data_arr = []
    label_arr = []
    paths = util.files.expand_path_arg(paths)
    for path in paths:
        path += '/'
        for file in listdir(path):
            if 'record' in file:
                with open(path + file) as data:
                    json_data = json.load(data)
                assert isfile(path + json_data['cam/image_array'])
                image = mx.image.imread(path + json_data['cam/image_array'])
                image = np.transpose(image, axes= (2, 0, 1))
                data_arr.append(image.astype('float32'))
                label_arr.append(np.array([json_data['user/angle'], json_data['user/throttle']]).astype('float32'))
    return data_arr, label_arr

# Helper Class to adjust the Donkey Car data generator to train the Gluon NN


class GluonDataSet(gluon.data.Dataset):
    def __init__(self, paths):
        self.data, self.label = load_data_in_path(paths)

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def split_data(paths, split_ratio):
        data, label = load_data_in_path(paths)
        data_split_index = int(len(data) * split_ratio)
        data_train = data[:data_split_index]
        label_train = label[:data_split_index]
        data_test = data[data_split_index:]
        label_test = label[data_split_index:]
        return GluonDataSet(data_train,label_train), GluonDataSet(data_test,label_test)