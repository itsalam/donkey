"""
The Gluon Model for the MxNet Variation for the DonkeyCar.
@Author Vincent Lam - @vlamai
"""

import numpy as np
import re
import os
from mxnet import nd, sym, gluon
from mxnet.ndarray import NDArray
from mxnet.symbol import Symbol
from donkeycar.util.data import linear_unbin
import mxnet as mx
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss, L1Loss


class GluonCategorical(gluon.nn.HybridBlock):
    """
    A Model with a categorical output for the angle and a float value for throttle. The run() function allows
    a 120 x 160 image to be fed through the trained network. Also has a static method specifying the loss method.
    """
    def __init__(self, num_classes=15):
        """
        The base constructor. Creates the base network before calling the method that creates the output layer.
        :param int num_classes: Specifies the number of classes for the angle classification layer.
        """
        super(GluonCategorical, self).__init__()
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        self._num_classes = num_classes
        with self.name_scope():
            self._create_base()
            self._create_output()

    def _create_base(self):
        """
        Function that specifies the base network. Allows child classes to overwrite and modify the base.
        """
        self._base = gluon.nn.HybridSequential()
        with self._base.name_scope():
            self._base.add(gluon.nn.Conv2D(channels=24, kernel_size=5, strides=(2, 2), activation='relu'))
            self._base.add(gluon.nn.Conv2D(channels=32, kernel_size=5, strides=(2, 2), activation='relu'))
            self._base.add(gluon.nn.Conv2D(channels=64, kernel_size=5, strides=(2, 2), activation='relu'))
            self._base.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(2, 2), activation='relu'))
            self._base.add(gluon.nn.Conv2D(channels=64, kernel_size=3, strides=(1, 1), activation='relu'))

            self._base.add(gluon.nn.Flatten())
            self._base.add(gluon.nn.Dense(100, activation='relu'))
            self._base.add(gluon.nn.Dropout(.1))
            self._base.add(gluon.nn.Dense(50, activation='relu'))
            self._base.add(gluon.nn.Dropout(.1))

    def _create_output(self):
        """
        Function that specifies the output layers. Allows child classes to overwrite and modify the output layers.
        """
        self._angle_output = gluon.nn.Dense(self._num_classes)
        self._throttle_output = gluon.nn.Dense(1, activation='relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        """
        Defines the handling on the input data and the output
        :param nd or sym F: NDarray or Sym module to preform operations with
        :param NDArray or Symbol x: NDarray of input data
        :return: Tuple of sym or nd
        """
        x = self._base(x)
        return self._angle_output(x), self._throttle_output(x)

    @staticmethod
    def loss():
        """
        The static method that specifies the corresponding loss function to apply.
        """
        return HybridLoss(angle_loss_function=SoftmaxCrossEntropyLoss(weight=.9),
                               throttle_loss_function=L1Loss(weight=.01))

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

    def load(self, path):
        """
        Loads the parameters found in the directory to the NN
        :param path: Directory to load from.
        :return: None
        """
        print('Loading model %s..' % path)
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


class ResGluonCategorical(GluonCategorical):
    """
    A Resnet Implementation of GluonCategorical class. Uses a ResNet18 premade model as the base network.
    Made to compare preformance of the simpler base network of the GluonCategorical class.
    """
    def __init__(self, ctx):
        super(ResGluonCategorical, self).__init__(ctx)

    def _create_base(self):
        """
        Overwritten function to set the base as ResNet18.
        """
        self.base = gluon.model_zoo.vision.resnet18_v2(pretrained=True, ctx=self.ctx).features

    def collect_params(self, select=None):
        """
        Overwritten function for the Trainer to only instaniate the angle and throttle output layers.
        """
        params = gluon.ParameterDict()
        params.update(self._angle_output)
        params.update(self._throttle_output)
        return params


class HybridLoss(gluon.loss.Loss):
    """
    The loss function in the that accepts two loss functions for angle and throttle.
    """

    def __init__(self, angle_loss_function, throttle_loss_function):
        """
        Default constructor, accepts two gluon.loss.Loss objects to be fed through.
        :param gluon.loss.Loss angle_loss_function: The loss function for the angle class
        :param gluon.loss.Loss throttle_loss_functionL The loss function for the throttle class
        """
        super(HybridLoss, self).__init__(weight=None, batch_axis=0)
        with self.name_scope():
            self.throttle_loss_func = throttle_loss_function
            self.angle_loss_func = angle_loss_function

    def hybrid_forward(self, F, angle, throttle, angle_label, throttle_label):
        """
        Defines the handling on the predicted angle and throttle with the label
        :param nd or sym F: NDarray or Sym module to preform operations with
        :param nd angle: predicted angle classes
        :param nd throttle: predicted throttle float
        :param nd angle_label: actual angle class's index
        :param nd throttle_label: actual throttle float
        :return: Tuple of sym or nd
        """
        angel_loss = self.angle_loss_func(angle, angle_label)
        throttle_loss = self.throttle_loss_func(throttle, throttle_label)
        return angel_loss + throttle_loss


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
