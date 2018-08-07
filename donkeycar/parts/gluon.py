"""
A Gluon variation for the NN back-end for the DonkeyCar. Requires MxNet to be Installed.
"""

import os.path
import re

import mxnet as mx
import numpy as np
from mxnet import nd, sym, autograd, gluon
from mxnet.ndarray import NDArray
from mxnet.symbol import Symbol


class GluonPilot:
    """
    Base Pilot class to handle the Network feed and training
    """

    def __init__(self):
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        self.net = None
        self.accuracy_threshold = .15
        self.angle_classes = 1
        # self.mean_rgb = np.array([[126.504562], [106.227355], [76.436961]])
        # self.std_rgb = np.array([[59.213406], [53.068041], [57.518464]])

    def train(self, train_gen, val_gen, saved_model_path, epochs=20, steps=100, train_split=0.8):



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
                data, label = self.format_gen_data(data, label, self.angle_classes)
                with autograd.record(train_mode=True):
                    output = self.net(data)
                    loss = self.loss(output, label)
                    loss.backward()
                self.optimizer.step(data.shape[0])

                current_loss = nd.mean(loss)
                moving_loss = (current_loss if ((steps == 0) and (epoch_index == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant * current_loss))
            moving_loss = moving_loss.asscalar() if not isinstance(moving_loss, np.float32) else moving_loss
            test_accuracy = self.evalulate_accuracy(val_gen, test_steps)
            train_accuracy = self.evalulate_accuracy(train_gen, test_steps)
            print("Epoch %s, Loss: %.8f, Train_acc: angle=%.4f throttle=%.4f, "
                  "Test_acc: angle=%.4f throttle=%.4f Epoch Retries: %s" % (
                      epoch_index, moving_loss,
                      train_accuracy[0], train_accuracy[1],
                      test_accuracy[0], test_accuracy[1],
                      epoch_retries))
            if old_loss <= moving_loss:
                epoch_retries += 1
                if epoch_retries > 8:
                    break
            else:
                epoch_retries = 0
            old_loss = moving_loss

        self.save(saved_model_path)

    def compile_model(self, loss=None, optimizer='sgd', learning_rate=1e-3, init_magnitude=2.24):
        """
        Initializes the net and instantiates the loss and optimization parameters
        :param loss: The gluon.loss.Loss() class to instantiates
        :param optimizer: A String to define the optimzation
        :param learning_rate: A float to set the Trainer's learning rate
        :param init_magnitude: Scale of variance for initial weights
        :return: None
        """
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=init_magnitude), ctx=self.ctx)
        # self.net.angle_output.collect_params().initialize(mx.init.Xavier(magnitude=init_magnitude), ctx=self.ctx)
        # self.net.throttle_output.collect_params().initialize(mx.init.Xavier(magnitude=init_magnitude), ctx=self.ctx)

        self.loss = gluon.loss.L2Loss() if loss is None else loss
        self.optimizer = mx.gluon.Trainer(self.net.collect_params(), optimizer,
                                          {'learning_rate': learning_rate
                                              # , 'wd': 0.001
                                           })

    def evalulate_accuracy(self, data_generator, steps):
        """
        Evaluates the net's output to actual label. Compares the predicted output to the actual output.
        If the absolute difference exceeds self.accuracy_threshold, the prediction is considered inaccurate.
        Returns the percentages of outputs that are considered accurate for steering and throttle predictions.
        :param data_generator: The dataset generator, iterates data and its label shuffled in batch sizes of 128.
        :param steps: The number of times to iterate through the generator.
        :return: Two floats representing the accuracy of the steering angle and throttle.
        """
        acc = mx.metric.Accuracy()
        throttle_acc = 0
        data_count = 0

        for i, (data, label) in zip(range(steps), data_generator):
            data, label = self.format_gen_data(data, label, self.angle_classes)
            output = self.net(data)
            angle_output = nd.argmax(output[0], axis=1)
            acc.update(angle_output, label[0])
            for throttle_Label, throttle_Prediction in zip(label[1], output[1]):
                throttle_err = (throttle_Label - throttle_Prediction).abs().asscalar()
                throttle_acc += 1 if throttle_err < self.accuracy_threshold else 0
                data_count += 1
        throttle_acc /= float(data_count)
        return acc.get()[1], throttle_acc

    def predict(self, img_arr):
        """
        Predicts the output given the image array input.
        :param img_arr: The Mxnet ND image array in the format (1 x Channel x Height x Width)
        :return: The steering angle and throttle as ND Array.
        """
        output = self.net(img_arr)
        return output

    def format_gen_data(self, data, label, num_classes=None):
        """
        Formats the Data Generator's Numpy data and label into ND arrays and adjusts the layout
        to feed through the NN optimally.
        :param numpy.array data: A Numpy array of the form (1, Batch_size, Height, Width, Channel)
        :param numpy.array label: A Numpy array of the form (Outputs, Batch_size)
        :param int num_classes: number of classes the angle should output; for categorical networks
        :return: The data and label in ND Array format, with data being the form (BCHW) and the label (Batch_size, Output)
        """
        data = data[0]
        data = np.transpose(data, axes=(0, 3, 1, 2))
        data = nd.array(data, self.ctx)

        label = nd.array(label, self.ctx)
        if num_classes > 1:
            for i, angle in enumerate(label[0]):
                label[0][i] = linear_bin(angle.asscalar(), num_classes)
        return data, label

    def load(self, path):
        """
        Loads the parameters found in the directory to the NN
        :param path: Directory to load from.
        :return: None
        """
        if os.path.isdir(path):
            p, folder_name = os.path.split(path)
            sym_path = path + '/' + folder_name + '-symbol.json'
            param_path = path + '/' + folder_name
            # self.net.base = gluon.nn.SymbolBlock(outputs= mx.sym.load_json(sym_path), inputs=mx.sym.var('data'))
            self.net.load_params(param_path, self.ctx)
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
        self.net.save_params(path + '/' + os.path.basename(path))

    def format_img_arr(self, img_arr):
        img_arr = img_arr.astype('float32') - self.mean_rgb.transpose()
        img_arr /= self.std_rgb.transpose()
        img_arr = np.transpose(img_arr, axes=(2, 0, 1))
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = nd.array(img_arr, self.ctx)
        return img_arr


class GluonLinear(GluonPilot):
    def __init__(self):
        super(GluonLinear, self).__init__()
        self.net = LinearOutput()
        # self.net.hybridize()
        self.compile_model(loss=gluon.loss.L2Loss())

    def run(self, img_arr):
        img_arr = self.format_img_arr(img_arr)
        output = self.predict(img_arr)
        return output[0], output[1]


class GluonHybrid(GluonPilot):
    """
    A GluonPilot implementation with a Network with a categorical output for angle and linear output for throttle
    """

    def __init__(self, num_classes=15):
        """
        Instantiates network through default_linear() and other objects with complile_model()
        :param int num_classes: Number of angular outputs classes to divide the reuslt to
        """
        super(GluonHybrid, self).__init__()
        self.angle_classes = num_classes
        self.net = CategoricalOutput(num_classes)
        self.softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss(weight=.9)
        self.L1_loss = gluon.loss.L1Loss(weight=.1)
        self.compile_model(loss=self.hybrid_loss, optimizer='adam')

    def run(self, img_arr):
        """
        Takes in the Numpy array of the output (the image) and predicts the angle and throttle
        :param img_arr: The numpy array of the image in the format (Height, Width, Channel)
        :return: Angle and throttle as floats
        """
        img_arr = self.format_img_arr(img_arr)

        output = self.predict(img_arr)
        output[0] = linear_unbin(output[0], self.angle_classes)
        return output[0], output[1][0].asscalar()

    def hybrid_loss(self, output, label):
        softmax_loss = self.softmax_loss(output[0], label[0])
        l1_loss = self.L1_loss(output[1], label[1])
        return softmax_loss + l1_loss


class LinearOutput(gluon.HybridBlock):
    def __init__(self):
        super(LinearOutput, self).__init__()
        with self.name_scope():
            self.base = gluon.nn.HybridSequential()
            with self.base.name_scope():
                self.base.add(gluon.nn.Conv2D(channels=24, kernel_size=5, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=32, kernel_size=5, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=64, kernel_size=5, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(1, 1), activation='relu'))

                self.base.add(gluon.nn.Flatten())
                self.base.add(gluon.nn.Dense(100))
                self.base.add(gluon.nn.Dropout(.50))
                self.base.add(gluon.nn.Dense(50))
                self.base.add(gluon.nn.Dropout(.25))
            self.angle_output = gluon.nn.Dense(1)
            self.throttle_output = gluon.nn.Dense(1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        """

        :param nd or sym F:
        :param NDArray or Symbol x:
        :return:
        """
        x = self.base(x)
        angle_output = F.clip(self.angle_output(x), -1, 1)
        throttle_output = F.clip(self.angle_output(x), 0, 1)
        return angle_output, throttle_output


class CategoricalOutput(gluon.HybridBlock):
    def __init__(self, angle_classes=15):
        super(CategoricalOutput, self).__init__()
        self.angle_classes = angle_classes
        with self.name_scope():
            self.base = gluon.nn.HybridSequential()
            with self.base.name_scope():
                self.base.add(gluon.nn.Conv2D(channels=24, kernel_size=5, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=32, kernel_size=5, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=64, kernel_size=5, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(2, 2), activation='relu'))
                self.base.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(1, 1), activation='relu'))

                self.base.add(gluon.nn.Flatten())
                self.base.add(gluon.nn.Dense(100, activation='relu'))
                self.base.add(gluon.nn.Dropout(.1))
                self.base.add(gluon.nn.Dense(50, activation='relu'))
                self.base.add(gluon.nn.Dropout(.1))

            # self.base = gluon.model_zoo.vision.resnet18_v2(pretrained=True, ctx=mx.gpu()).features

            self.angle_output = gluon.nn.Dense(self.angle_classes)
            self.throttle_output = gluon.nn.Dense(1, activation='relu')
        self.hybridize()
    def hybrid_forward(self, F, x, *args, **kwargs):
        """

        :param nd or sym F:
        :param NDArray or Symbol x:
        :return:
        """
        x = self.base(x)

        return [self.angle_output(x), self.throttle_output(x)]


def linear_bin(angle, num_classes):
    angle = angle + 1
    classifier = 2. / (num_classes - 1)
    b = round(angle / classifier)
    return int(b)


def linear_unbin(b, num_classes):
    b = nd.argmax(b, axis=1).asscalar()
    a = b * (2. / (num_classes - 1)) - 1
    return a
