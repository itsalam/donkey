"""
The Gluon Trainer for the MxNet Variation for the DonkeyCar.
@Author Vincent Lam - @vlamai
"""

import mxnet as mx
from mxnet import nd, autograd, gluon


class GluonTrainer:
    """
    A Pilot that runs a Neural Network with a categorical output for angle and linear output for throttle
    """
    def __init__(self, model):
        """
        Base constructor for the Gluon Trainer. Instantiates the model given in the parameters, along with
        constant values for the training.
        :param donkeycar.parts.gluon_model.GluonCategorical model:
        """
        super(GluonTrainer, self).__init__()
        self._throttle_acc_threshold = .1
        self._epoch_retries = 5
        self._net = model
        self._ctx = model.ctx
        self.compile_model(model.loss())

    def compile_model(self, loss, optimizer='adam', learning_rate=1e-3):
        """
        Initializes the net and instantiates the loss and optimization parameters
        :param gluon.loss.Loss loss: The gluon.loss.Loss() class to instantiates
        :param optimizer: A String to define the optimzation
        :param learning_rate: A float to set the Trainer's learning rate
        :return None:
        """
        self._net.collect_params().initialize(mx.init.Uniform(.05), ctx=self._ctx)
        self.optimizer = mx.gluon.Trainer(self._net.collect_params(), optimizer,
                                          {'learning_rate': learning_rate,
                                           'wd': 0.001
                                           })
        self._net.hybridize()
        self.loss = loss

    def train(self, train_gen, val_gen, saved_model_path, epochs=50):
        """
        Trains a Neural Network, and saves the results.
        :param train_gen: The training data generator. Yields Numpy arrays with the data and label.
        :param val_gen: The validation(test) data generator. Yields Numpy arrays with the data and label.
        :param saved_model_path: Directory to save the model to.
        :param epochs: Number of epochs to run
        :return None: Saves a Neural Network to the stated directory.
        """
        smoothing_constant = .01
        best_loss = float('inf')
        epoch_retries = 0
        for epoch_index in range(epochs):
            for steps, (data, label) in enumerate(train_gen):
                data = data.as_in_context(self._ctx)
                angle_label, throttle_label = label.as_in_context(self._ctx).transpose()
                with autograd.record(train_mode=True):
                    angle, throttle = self._net(data)
                    loss = self.loss(angle, throttle, angle_label, throttle_label)
                loss.backward()
                self.optimizer.step(data.shape[0])
                current_loss = nd.mean(loss)
                moving_loss = (current_loss if ((steps == 0) and (epoch_index == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant * current_loss))
            moving_loss = moving_loss.asscalar()
            test_angle_acc, test_throttle_acc = self.evaluate_accuracy(val_gen)
            train_angle_acc, train_throttle_acc = self.evaluate_accuracy(train_gen)
            if moving_loss > best_loss:
                epoch_retries += 1
                if epoch_retries >= self._epoch_retries:
                    break
            best_loss = moving_loss if moving_loss < best_loss else best_loss
            print("Epoch %s, Loss: %.8f, Train_acc: angle=%.4f throttle=%.4f, "
                  "Test_acc: angle=%.4f throttle=%.4f Epoch Retries: %s" % (
                      epoch_index, moving_loss,
                      train_angle_acc, train_throttle_acc,
                      test_angle_acc, test_throttle_acc,
                      epoch_retries))
        self._net.save(saved_model_path)

    def evaluate_accuracy(self, data_generator):
        """
        Evaluates the net's output to actual value.
        If the absolute difference exceeds self.accuracy_threshold, the prediction is considered inaccurate.
        Returns the percentages of outputs that are considered accurate for steering and throttle predictions.
        :param data_generator: The dataset generator, iterates data and its label shuffled in batch sizes of 128.
        :return Tuple(float, float): Two floats representing the accuracy of the steering angle and throttle.
        """
        acc = mx.metric.Accuracy()
        throttle_acc = 0
        data_count = 0

        for i, (data, label) in enumerate(data_generator):
            data = data.as_in_context(self._ctx)
            label = label.as_in_context(self._ctx).transpose()
            output = self._net(data)
            angle_output = nd.argmax(output[0], axis=1)
            acc.update(angle_output, label[0])
            for throttle__label, throttle__prediction in zip(label[1], output[1]):
                throttle_err = (throttle__label - throttle__prediction).abs().asscalar()
                throttle_acc += 1 if throttle_err < self._throttle_acc_threshold else 0
                data_count += 1
        throttle_acc /= float(data_count)
        return float(acc.get()[1]), throttle_acc

