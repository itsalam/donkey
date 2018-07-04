import mxnet as mx
from mxnet import nd, autograd, gluon

import donkeycar as dk

import os.path
import numpy as np

import re


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
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss() if loss is None else loss
        self.optimizer = mx.gluon.Trainer(self.net.collect_params(), optimizer, {'learning_rate': learning_rate})

    def evalulate_accuracy(self, data_iterator):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.net(data)
            predictions = nd.argmax(output, 1)
            acc.update(predictions, label)
        return acc.get()[1]

    def predict(self, img_arr):
        # adjust for default gluon CNN input
        input_arr = nd.array(img_arr, ctx=self.ctx).transpose(axes=(0, 3, 1, 2))
        throttle_output = 0.0
        angle_output = self.net(input_arr)

        return angle_output, throttle_output

    def train(self, train_gen, val_gen, saved_model_path, epochs= 100, steps=100, train_split=0.8):

        self.compile_model()
        self.load(saved_model_path)

        test_steps = int(round(steps * (1.0 - train_split)))

        # Unbatch the dataset
        train_dataset = GluonDataset(train_gen, steps)
        val_dataset = GluonDataset(val_gen, test_steps)

        train_dataload = mx.gluon.data.DataLoader(train_dataset, batch_size=128)
        test_dataload = mx.gluon.data.DataLoader(val_dataset, batch_size=128)

        smoothing_constant = .01
        old_loss = 100
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
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s Batch_# %s" % (
                epoch_index, moving_loss, train_accuracy, test_accuracy, i))
            if old_loss < moving_loss:
                break
            old_loss = moving_loss

        self.save(saved_model_path)


class GluonCategorical(GluonPilot):
    def __init__(self, net=None):
        super(GluonCategorical, self).__init__()
        if net:
            self.net = net
        else:
            self.net = default_categorical()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.predict(img_arr)
        # print('throttle', throttle)
        # angle_certainty = max(angle_binned[0])
        angle_unbinned = dk.utils.linear_unbin(angle_binned.asnumpy())
        return angle_unbinned, throttle


def default_categorical():
    net = gluon.nn.HybridSequential(prefix='')
    with net.name_scope():
        # 24 features, 5 pixel x 5 pixel kernel (convolution, feature) window, 2wx2h stride, relu activation
        net.add(gluon.nn.Conv2D(channels=24, kernel_size=5, strides=(2, 2), activation='relu'))
        # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, strides=(2, 2), activation='relu'))
        # 64 features, 5px5p kernel window, 2wx2h stride, relu
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=5, strides=(2, 2), activation='relu'))
        # 64 features, 3px3p kernel window, 2wx2h stride, relu
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(2, 2), activation='relu'))
        # 64 features, 3px3p kernel window, 1wx1h stride, relu
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(1, 1), activation='relu'))

        # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

        net.add(gluon.nn.Flatten())  # Flatten to 1D (Fully connected)
        net.add(gluon.nn.Dense(100, activation='relu'))  # Classify the data into 100 features, make all negatives 0
        net.add(gluon.nn.Dropout(.1))  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
        net.add(gluon.nn.Dense(50, activation='relu'))  # Classify the data into 50 features, make all negatives 0
        net.add(gluon.nn.Dropout(.1))  # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)

        # categorical output of the angle

        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
        angle_out = gluon.nn.Dense(15)

        # continous output of throttle
        # throttle_out = gluon.nn.Dense(1, activation='relu')  # Reduce to 1 number, Positive number only

        net.add(angle_out)
    net.hybridize()
    return net


# Helper function to convert the generator of batches into a list


def transform_gen(data_gen, num_batches):
    data_batches, label_batches = map(list, zip(*[(x[0], y[0]) for (x, y), i in zip(data_gen, range(num_batches))]))

    data_list = []
    label_list = []

    for data_batch, label_batch in zip(data_batches, label_batches):
        for data in data_batch:
            data = np.swapaxes(data, 2, 0)
            data = np.swapaxes(data, 1, 2)
            data_list.append(data.astype('float32'))
        for label in label_batch:
            label_list.append(label)

    return data_list, label_list

# Helper Class to adjust the Donkey Car data generator to train the Gluon NN


class GluonDataset(gluon.data.Dataset):
    def __init__(self, data_gen, num_batches):
        data, label = transform_gen(data_gen, num_batches)
        assert (len(data) == len(label))
        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], mx.nd.topk(mx.nd.array(self.label[item]))

    def __len__(self):
        return len(self.data)


