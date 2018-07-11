import os.path
import re
import time

import mxnet as mx
from mxnet import nd, autograd, gluon


class GluonPilot:

    def __init__(self):
        self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        self.net = None
        self.throttle_output = False

    def load(self, path):
        if os.path.isdir(path):
            p, folder_name = os.path.split(path)
            path += '/' + folder_name + '-0000.params'
            self.net.load_parameters(path, self.ctx)
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

    def evalulate_accuracy(self, data_iterator, steps):
        acc = mx.metric.Accuracy()
        for i, (data, label) in zip(range(steps), data_iterator):
            data, label = self.format_gen_data(data, label)
            output = self.net(data)
            acc.update(output, label)
        return acc.get()[1]

    def predict(self, img_arr):
        output = self.net(img_arr).asnumpy()
        return output[0]

    def format_gen_data(self, data, label):
        data = nd.array(data).as_in_context(self.ctx)
        data = data[0]
        data = nd.transpose(data, axes=(0, 3, 1, 2))
        label = nd.array([label[0], label[1]]).as_in_context(self.ctx)
        label = label.swapaxes(0, 1)
        return data, label

    def train(self, train_gen, val_gen, saved_model_path, epochs=100, steps=100, train_split=0.8):

        train_steps = int(steps * (1.0 - train_split) / train_split)
        smoothing_constant = .01
        old_loss = float('Inf')
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
            test_accuracy = self.evalulate_accuracy(train_gen, train_steps)
            train_accuracy = self.evalulate_accuracy(val_gen, train_steps)
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
        img_arr = nd.array(img_arr)
        img_arr = nd.transpose(img_arr, axes=(2, 0, 1))
        img_arr = nd.expand_dims(img_arr, 0)
        output = self.predict(img_arr)
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
