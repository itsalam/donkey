"""
A Gluon variation for the NN back-end for the DonkeyCar. Requires MxNet to be Installed.
"""

import os.path
import re
import time
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

from datetime import datetime

class GluonPilot:
    """
    Base Pilot class to handle the Network feed and training
    """
    def __init__(self):
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        self.net = None
        self.throttle_output = False
        self.accuracy_threshold = .1

    def compile_model(self, loss=None, optimizer='sgd', learning_rate=1e-3, init_magnitude=1):
        """
        Initializes the net and instantiates the loss and optimization parameters
        :param loss: The gluon.loss.Loss() class to instantiates
        :param optimizer: A String to define the optimzation
        :param learning_rate: A float to set the Trainer's learning rate
        :param init_magnitude: Scale of variance for initial weights
        :return: None
        """
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=init_magnitude), ctx=self.ctx)
        self.loss = gluon.loss.L2Loss() if loss is None else loss
        self.optimizer = mx.gluon.Trainer(self.net.collect_params(), optimizer, {'learning_rate': learning_rate})

    def evalulate_accuracy(self, data_generator, steps):
        """
        Evaluates the net's output to actual label. Compares the predicted output to the actual output.
        If the absolute difference exceeds self.accuracy_threshold, the prediction is considered inaccurate.
        Returns the percentages of outputs that are considered accurate for steering and throttle predictions.
        :param data_generator: The dataset generator, iterates data and its label shuffled in batch sizes of 128.
        :param steps: The number of times to iterate through the generator.
        :return: Two floats representing the accuracy of the steering angle and throttle.
        """
        angle_acc = 0
        throttle_acc = 0
        data_count = 0

        for i, (data, label) in zip(range(steps), data_generator):
            data, label = self.format_gen_data(data, label)
            output = self.net(data)
            for yLabel, yPrediction in zip(label, output):
                angle_err = (yLabel[0] - yPrediction[0]).abs().asscalar()
                throttle_err = (yLabel[1] - yPrediction[1]).abs().asscalar()
                angle_acc += 1 if angle_err < self.accuracy_threshold else 0
                throttle_acc += 1 if throttle_err < self.accuracy_threshold else 0
                data_count += 1

        angle_acc /= float(data_count)
        throttle_acc /= float(data_count)
        return angle_acc, throttle_acc

    def predict(self, img_arr):
        """
        Predicts the output given the image array input.
        :param img_arr: The Mxnet ND image array in the format (1 x Channel x Height x Width)
        :return: The steering angle and throttle as Numpy Array.
        """
        output = self.net(img_arr).asnumpy()
        return output[0]

    def format_gen_data(self, data, label):
        """
        Formats the Data Generator's Numpy data and label into ND arrays and adjusts the layout
        to feed through the NN optimally.
        :param data: A Numpy array of the form (1, Batch_size, Height, Width, Channel)
        :param label: A Numpy array of the form (Outputs, Batch_size)
        :return: The data and label in ND Array format, with data being the form (BCHW) and the label (Batch_size, Output)
        """

        data = data[0]
        data = np.transpose(data, axes=(0, 3, 1, 2))
        data = nd.array(data, self.ctx)
        label = nd.array(label, self.ctx)
        label = label.swapaxes(0, 1)
        return data, label

    def train(self, train_gen, val_gen, saved_model_path, epochs=100, steps=100, train_split=0.8):
        """
        Trains a Neural Network, and saves the results.
        :param train_gen: The training data generator. Yields Numpy arrays with the data and label.
        :param val_gen: The validation(test) data generator. Yields Numpy arrays with the data and label.
        :param saved_model_path: Directory to save the model to.
        :param epochs: Number of epochs to run
        :param steps: Number of iterations on the Generator per Epoch
        :param train_split: Ratio of datapoints split for training and testing.
        :return: None, but saves a Neural Network to the stated directory.
        """
        test_steps = int(steps * (1.0 - train_split) / train_split)
        smoothing_constant = .01
        old_loss = float('Inf')
        epoch_retries = 0
        for epoch_index in range(epochs):
            for steps, (data, label) in zip(range(steps), train_gen):
                data, label = self.format_gen_data(data, label)
                with autograd.record(train_mode=True):
                    output = self.net(data)
                    loss = self.loss(output, label)

                loss.backward()
                self.optimizer.step(data.shape[0])

                current_loss = nd.mean(loss).asscalar()
                moving_loss = (current_loss if ((steps == 0) and (epoch_index == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant * current_loss))
            test_accuracy = self.evalulate_accuracy(train_gen, test_steps)
            train_accuracy = self.evalulate_accuracy(val_gen, test_steps)
            print("Epoch %s. Loss: %.4f, Train_acc: angle=%.4f throttle=%.4f, "
                  "Test_acc: angle=%.4f throttle=%.4f Epoch Retries: %s" % (
                      epoch_index, moving_loss,
                      train_accuracy[0], train_accuracy[1],
                      test_accuracy[0], test_accuracy[1],
                      epoch_retries))
            if old_loss < moving_loss:
                epoch_retries += 1
                if epoch_retries > 5:
                    break
            old_loss = moving_loss

        self.save(saved_model_path)

    def load(self, path):
        """
        Loads the parameters found in the directory to the NN
        :param path: Directory to load from.
        :return: None
        """
        if os.path.isdir(path):
            p, folder_name = os.path.split(path)
            path += '/' + folder_name + '-0000.params'
            self.net.load_params(path, self.ctx)
            print('Sucessfully loaded ', folder_name)
        else:
            print('Folder not found.')

    def save(self, path):
        """
        Saves the network to a newly made directory. If the directory exists, adjust the directory name and retry.
        :param path: Directory to create and save parameters to.
        :return: None
        """
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


class GluonLinear(GluonPilot):
    """
    A GluonPilot implementation with a Network with a linear output for throttle and angle.
    """
    def __init__(self, net=None):
        """
        Instantiates network through default_linear() and other objects with complile_model()
        :param net: Network to apply (optional) will use the default linear net otherwise.
        """
        super(GluonLinear, self).__init__()
        self.net = net if net is not None else default_linear()
        self.compile_model()
        self.elapsed = 0.0
        self.run_count = 0

    def run(self, img_arr):
        """
        Takes in the Numpy array of the output (the image) and predicts the angle and throttle
        :param img_arr: The numpy array of the image in the format (Height, Width, Channel)
        :return: Angle and throttle as floats
        """
        start = time.time()

        img_arr = np.transpose(img_arr, axes=(2, 0, 1))
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = nd.array(img_arr)

        output = self.predict(img_arr)

        self.elapsed += time.time() - start
        self.run_count += 1
        if self.run_count % 100 == 99:
            print(self.elapsed / self.run_count)
        return output[0], output[1]


class ClippedNN(gluon.nn.HybridSequential):
    """
    A Gluon Hybrid Block that clips the output from -1 to 1 to reflect the output range.
    """
    def __init__(self):
        super(ClippedNN, self).__init__('')

    def hybrid_forward(self, F, x):
        x = super().hybrid_forward(F, x)
        return F.clip(x, -1, 1)


def default_linear():
    """
    The default linear NN parameters.
    :return:
    """
    net = ClippedNN()
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
