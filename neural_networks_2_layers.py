# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from layer_model import relu, tanh, linear_layer, dropout, miniBatchGradientDescent
from utils import softmax_cross_entropy, add_momentum, predict_label, DataSplit
import argparse
import matplotlib.pyplot as plt

def main(main_params, optimization_type="minibatch_sgd"):
    # data processing
    data = pd.read_csv("dataset/london_merged.csv")
    data.drop(['timestamp'], inplace=True, axis=1)

    bins = [-1, 1000, 2000, 3000, 4000, 100000]
    dpy = data['cnt'].to_numpy()
    r = pd.cut(dpy, bins)
    data_y = r.codes

    data.drop(['cnt'], inplace=True, axis=1)
    data = (data - data.min()) / (data.max() - data.min())
    data = data.to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(data, data_y, test_size=0.33,
                                                  random_state=int(main_params['random_seed']))

    N_train, d = x_train.shape
    N_val, _ = x_val.shape

    trainSet = DataSplit(x_train, y_train)
    valSet = DataSplit(x_val, y_val)

    # building/defining model

    # experimental setup
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    # optimization setting: _alpha for momentum, _lambda for weight decay
    _learning_rate = float(main_params['learning_rate'])
    _step = 10
    _alpha = float(main_params['alpha'])
    _lambda = float(main_params['lambda'])
    _dropout_rate = float(main_params['dropout_rate'])
    _activation = main_params['activation']

    if _activation == 'relu':
        act = relu
    else:
        act = tanh

    # The network structure is input --> linear1 --> relu --> dropout --> linear2 --> tanh-> output_liner
    # softmax_cross_entropy loss num_L1:the hidden_layer size num_L2:the output_layer size
    model = dict()
    num_L1 = 100
    num_L2 = 5
    # create objects (modules) from the module classes
    model['L1'] = linear_layer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = act()
    model['drop1'] = dropout(r=_dropout_rate)
    model['L2'] = linear_layer(input_D=num_L1, output_D=num_L2)
    model['loss'] = softmax_cross_entropy()
    # print(model.params)

    # Momentum
    if _alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []
    train_loss_record = []
    val_loss_record = []

    # training & validation
    for t in range(num_epoch):

        if (t % _step == 0) and (t != 0):
            _learning_rate = _learning_rate * 0.1

        idx_order = np.random.permutation(N_train)

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0
        val_loss = 0.0

        for i in range(int(np.floor(N_train / minibatch_size))):
            # get a mini-batch of data
            x, y = trainSet.get_example(idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            # forward
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=True)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)

            # backward
            grad_a2 = model['loss'].backward(a2, y)
            grad_d1 = model['L2'].backward(d1, grad_a2)
            grad_h1 = model['drop1'].backward(h1, grad_d1)
            grad_a1 = model['nonlinear1'].backward(a1, grad_h1)
            grad_x = model['L1'].backward(x, grad_a1)
            # update gradient for each model
            model = miniBatchGradientDescent(model, momentum, _lambda, _alpha, _learning_rate)

        # training accuracy
        for i in range(int(np.floor(N_train / minibatch_size))):
            x, y = trainSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            # forward
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)

            train_loss += loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_loss = train_loss
        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        # validation accuracy
        for i in range(int(np.floor(N_val / minibatch_size))):
            x, y = valSet.get_example(np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            # forward
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=False)
            a2 = model['L2'].forward(d1)

            loss = model['loss'].forward(a2, y)
            val_loss += loss
            val_acc += np.sum(predict_label(a2) == y)
            val_count += len(y)

        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)
        print('At epoch ' + str(t + 1))
        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))
        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    index = [int(i+1) for i in range(num_epoch)]
    plt.figure()
    plt.grid()
    plt.plot(index, train_acc_record, color='b', label='train_acc')
    plt.plot(index, val_acc_record, color='darkorange', label='valadation_acc')
    plt.xlabel('training epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    return train_acc_record, val_acc_record


if __name__ == "__main__":

    train_acc = []
    val_acc = []
    dropouts = [0.02 * i for i in range(2)]

    for dropout_r in dropouts:
        parser = argparse.ArgumentParser()
        parser.add_argument('--random_seed', default=42)
        parser.add_argument('--learning_rate', default=0.01)
        parser.add_argument('--alpha', default=0.01)
        parser.add_argument('--lambda', default=0.0)
        parser.add_argument('--dropout_rate', default=dropout_r)
        parser.add_argument('--num_epoch', default=10)
        parser.add_argument('--minibatch_size', default=30)
        parser.add_argument('--activation', default='relu')
        args = parser.parse_args()
        main_params = vars(args)
        tmp_acc1, tmp_acc2 = main(main_params)

        train_acc.append(tmp_acc1[9])
        val_acc.append(tmp_acc2[9])

    plt.figure()
    plt.grid()
    plt.plot(dropouts, train_acc, color='b', label='train_acc')
    plt.plot(dropouts, val_acc, color='darkorange', label='val_acc')
    plt.xlabel('dropout rate')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

