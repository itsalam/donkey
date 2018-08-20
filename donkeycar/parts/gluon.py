"""
A Gluon variation for the NN back-end for the DonkeyCar. Requires MxNet to be Installed.
@Author Vincent Lam - @vlamai
"""
import re
import json
import os
import mxnet as mx
import numpy as np
from PIL import Image
from mxnet import nd, sym, autograd, gluon
from mxnet.ndarray import NDArray
from mxnet.symbol import Symbol
from donkeycar import util
from multiprocessing import cpu_count

class GluonPilot(gluon.nn.HybridBlock):
    """
    A Pilot that runs a Neural Network with a categorical output for angle and linear output for throttle
    """
    def __init__(self, loss=None, num_classes = 15):
        super(GluonPilot, self).__init__()
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        self.num_classes = num_classes
        self.softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss(weight=.9)
        self.L1_loss = gluon.loss.L1Loss(weight=.01)
        self.create_model()
        self.compile_model(loss=loss)

        self.throttle_acc_threshold = .1
        self.epoch_retries = 5

    def create_output(self):
        self.angle_output = gluon.nn.Dense(self.num_classes)
        self.throttle_output = gluon.nn.Dense(1, activation='relu')

    def create_model(self):
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
            self.create_output()

    def compile_model(self, loss=None, optimizer='adam', learning_rate=1e-3):
        """
        Initializes the net and instantiates the loss and optimization parameters
        :param loss: The gluon.loss.Loss() class to instantiates
        :param optimizer: A String to define the optimzation
        :param learning_rate: A float to set the Trainer's learning rate
        :return: None
        """
        self.collect_params().initialize(mx.init.Uniform(.05), ctx=self.ctx)
        self.loss = self.hybrid_loss if loss is None else loss
        self.optimizer = mx.gluon.Trainer(self.collect_params(), optimizer,
                                          {'learning_rate': learning_rate,
                                           'wd': 0.001
                                           })
        self.hybridize()

    def hybrid_loss(self, output, label):
        """
        The loss function in the form of two loss function for separate values. Output and Label index 0
        is the classification output for angle, index 1 is regression for throttle.
        :param NDArray output: The predicted values
        :param NDArray label: The label values
        :return gluon.loss.Loss: Computed loss object
        """
        softmax_loss = self.softmax_loss(output[0], label[0])
        l1_loss = self.L1_loss(output[1], label[1])
        return softmax_loss + l1_loss

    def hybrid_forward(self, F, x, *args, **kwargs):
        """
        Defines the handling on the input data and the output
        :param nd or sym F:
        :param NDArray or Symbol x:
        :return: List of sym
        """
        x = self.base(x)
        return [self.angle_output(x), self.throttle_output(x)]

    def train(self, train_gen, val_gen, saved_model_path, epochs=50, steps=100, train_split=0.8):
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
        best_loss = float('inf')
        epoch_retries = 0
        for epoch_index in range(epochs):
            for steps, (data, label) in enumerate(train_gen):
                data, label = self.format_gen_data(data, label)
                with autograd.record(train_mode=True):
                    output = self(data)
                    loss = self.loss(output, label)
                loss.backward()
                self.optimizer.step(data.shape[0])
                current_loss = nd.mean(loss)
                moving_loss = (current_loss if ((steps == 0) and (epoch_index == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant * current_loss))
            moving_loss = moving_loss.asscalar()
            test_angle_acc, test_throttle_acc = self.evalulate_accuracy(val_gen, test_steps)
            train_angle_acc, train_throttle_acc = self.evalulate_accuracy(train_gen, test_steps)
            if moving_loss > best_loss:
                epoch_retries += 1
                if epoch_retries >= self.epoch_retries:
                    break
            best_loss = moving_loss if moving_loss < best_loss else best_loss
            print("Epoch %s, Loss: %.8f, Train_acc: angle=%.4f throttle=%.4f, "
                  "Test_acc: angle=%.4f throttle=%.4f Epoch Retries: %s" % (
                      epoch_index, moving_loss,
                      train_angle_acc, train_throttle_acc,
                      test_angle_acc, test_throttle_acc,
                      epoch_retries))
        self.save(saved_model_path)

    def evalulate_accuracy(self, data_generator, steps):
        """
        Evaluates the net's output to actual value.
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
            data, label = self.format_gen_data(data, label)
            output = self(data)
            angle_output = nd.argmax(output[0], axis=1)
            acc.update(angle_output, label[0])
            for throttle__label, throttle__prediction in zip(label[1], output[1]):
                throttle_err = (throttle__label - throttle__prediction).abs().asscalar()
                throttle_acc += 1 if throttle_err < self.throttle_acc_threshold else 0
                data_count += 1
        throttle_acc /= float(data_count)
        return float(acc.get()[1]), throttle_acc

    def run(self, img_arr):
        """
        Takes in the Numpy array of the output (the image) and predicts the angle and throttle
        :param img_arr: The numpy array of the image in the format (Height, Width, Channel)
        :return: Angle and throttle as floats
        """

        img_arr = format_img_arr(img_arr.astype('float32'))
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = nd.array(img_arr, self.ctx)
        output = self(img_arr)
        angle_output = linear_unbin(output[0][0].asnumpy())
        return angle_output, output[1][0].asscalar()

    def format_gen_data(self, data, label):
        """
        Formats the Data Generator's Numpy data and label into ND arrays to feed through the NN.
        :param numpy.array data: A Numpy of the shape [1(formally number of batches), Batch_size, Height, Width, Channel]
        :param numpy.array label: A Numpy array of the shape [1, Batch_size, (Angle_label, Throttle_label)]
        :return: The data and label in NDArray, with data being the shape (BCHW) and the label (Batch_size, Output)
        """
        data = nd.array(data, self.ctx)
        label = nd.array(label, self.ctx)
        for i, angle in enumerate(label[0]):
            label[0][i] = np.argmax(linear_bin(angle.asscalar(), self.num_classes))
        return data, label.transpose()

    def load(self, path):
        """
        Loads the parameters found in the directory to the NN
        :param path: Directory to load from.
        :return: None
        """
        print('Loading model %s..' % (path))
        if os.path.isdir(path):
            p, folder_name = os.path.split(path)
            param_path = path + '/' + folder_name
            self.load_parameters(param_path, self.ctx)
            print('\tSucessfully loaded.', folder_name)
        else:
            print('\tFolder not found.')
            exit(1)

    def save(self, path):
        """
        Saves the network to a newly made directory. If the directory exists, adjust the directory name and retry.
        :param path: Directory to create and save parameters to.
        :return: None
        """
        print('Saving model...')
        while os.path.exists(path):
            print("\tExisting folder found at %s, creating a new one ..." % path)

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

        print("\tNew folder made at: ", path)
        os.makedirs(path)
        self.save_parameters(path + '/' + os.path.basename(path))


class GluonRes(GluonPilot):
    def __init__(self):
        super(GluonRes, self).__init__()

    def create_model(self):
        with self.name_scope():
            self.base = gluon.model_zoo.vision.resnet18_v2(pretrained=True, ctx=self.ctx).features
            self.create_output()

    def compile_model(self, loss=None, optimizer='adam', learning_rate=1e-3):
        """
        Initializes the net and instantiates the loss and optimization parameters
        :param loss: The gluon.loss.Loss() class to instantiates
        :param optimizer: A String to define the optimzation
        :param learning_rate: A float to set the Trainer's learning rate
        :return: None
        """
        self.angle_output.collect_params().initialize(mx.init.Uniform(.05), ctx=self.ctx)
        self.throttle_output.collect_params().initialize(mx.init.Uniform(.05), ctx=self.ctx)
        self.loss = self.hybrid_loss if loss is None else loss
        self.optimizer = mx.gluon.Trainer(self.collect_params(), optimizer,
                                          {'learning_rate': learning_rate,
                                           'wd': 0.001
                                           })
        self.hybridize()


def linear_bin(a, num_classes):
    a = a + 1
    b = a / (2 / (num_classes-1))
    b = 0 if b < 1 else b
    b = num_classes-1 if b > 5 else b
    b = round(b)
    arr = np.zeros(num_classes)
    arr[int(b)] = 1
    return arr

def linear_unbin(arr):
    b = np.argmax(arr)
    a = b * (2 / (len(arr)-1)) - 1
    return a

def format_img_arr(img_arr):
    """
    Formats the PyCamera's Numpy image data and into an ND array to feed through the NN.
    :param numpy.array img_arr: A Numpy array of the form (Height, Width, Channel)
    :return: The data and label in ND Array format, with data being the form (BCHW)
    """
    height_crop = int(img_arr.shape[0] / 3)
    img_arr = img_arr[height_crop:]
    img_arr = np.transpose(img_arr, axes=(2, 0, 1))
    return img_arr

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

    train_dataloader = gluon.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=cpu_count())
    test_dataloader = gluon.data.DataLoader(test_dataset, batch_size = batch_size, num_workers=cpu_count())

    return train_dataloader, test_dataloader


class GluonDataSet(gluon.data.Dataset):
    def __init__(self, record_list, load_in_memory=True):
        super(GluonDataSet, self).__init__()
        self.in_memory = load_in_memory
        if self.in_memory:
            self.json_records = []
            for record in record_list:
                self.json_records.append(self.get_record_data(record))
        else:
            self.json_records = record_list

    def __getitem__(self, item):
        if self.in_memory:
            return self.json_records[item]
        else:
            return self.get_record_data(self.json_records[item])

    def get_record_data(self, path):
        with open(path,'r') as fp:
            json_data = json.load(fp)

        base_path, file = os.path.split(path)
        img_path = json_data["cam/image_array"]
        image = np.array(Image.open(base_path + '/' + img_path))
        image = format_img_arr(image)
        throttle = json_data["user/throttle"]
        angle = json_data["user/angle"]
        data = image.astype('float32')
        label = np.array([angle, throttle]).astype('float32')
        return data, label

    def __len__(self):
        return len(self.json_records)