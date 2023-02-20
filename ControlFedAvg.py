# -*- coding: utf-8 -*-
__author__ = 'fff_zrx'

import numpy as np
import pandas as pd
import random
import os
from keras.models import Model
from keras.layers import *
from keras.optimizers import RMSprop
import keras.backend as backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Clients Params  客户端的参数定义
epochs = 20
batch_size = 25  # 批次训练样本数目
time_step = 6
print("epochs", epochs)
print("batch_size", batch_size)
print("time_step", time_step)

# Federated Learning Params  联邦学习中心模型参数
t_round = 8
num_clients = 4
print("t_round", t_round)
print("num_clients", num_clients)
# effectParams = ['none', 'none'] # order
# effectParams = ['effect', 'none'] # order + poi
# effectParams = ['none', 'tmp0'] # order + tmp
effectParams = ['effect', 'tmp0']  # order + poi + tmp
print("effectParams", effectParams)

# client model
def get_model(input_timesteps, output_timesteps):
    main_input = Input(shape=(input_timesteps, 6, 6, 1), name="main_input")
    print(main_input.shape)  # (None, 6, 6, 6, 1)
    batch_norm_0 = BatchNormalization(name='batch_norm_0')(main_input)
    print(batch_norm_0.shape)  # (None, 6, 6, 6, 1)
    conv_lstm1 = ConvLSTM2D(name='conv_lstm_1',
                            filters=16, kernel_size=(3, 3),
                            padding='same',
                            return_sequences=True)(batch_norm_0)
    print(conv_lstm1.shape)  # (None, 6, 6, 6, 16)

    dropout_1 = Dropout(0.5, name='dropout_1')(conv_lstm1)
    print(dropout_1.shape)  # (None, 6, 6, 6, 16)
    batch_norm_1 = BatchNormalization(name='batch_norm_1')(dropout_1)
    conv_lstm_2 = ConvLSTM2D(name='conv_lstm_2',
                             filters=16, kernel_size=(3, 3),
                             padding='same',
                             return_sequences=False)(batch_norm_1)
    print(conv_lstm_2.shape)  # (None, 6, 6, 16)
    dropout_2 = Dropout(0.5, name='dropout_2')(conv_lstm_2)
    batch_norm_2 = BatchNormalization(name='batch_norm_2')(dropout_2)
    flat_1 = Flatten()(batch_norm_2)
    print(flat_1.shape)  # (None, 576)
    repeat = RepeatVector(output_timesteps)(flat_1)
    print('repeat', repeat.shape)  # (None, 1,576)
    reshape1 = Reshape((output_timesteps, 6, 6, 16))(repeat)  # (None, 1, 6, 6, 16)
    print('reshape1', reshape1.shape)
    conv_lstm_3 = ConvLSTM2D(name='conv_lstm_3',
                             filters=16, kernel_size=(3, 3),
                             padding='same',
                             return_sequences=True)(reshape1)
    dropout_3 = Dropout(0.5, name='dropout_3')(conv_lstm_3)
    batch_norm_3 = BatchNormalization(name='batch_norm_3')(dropout_3)
    conv_lstm_4 = ConvLSTM2D(name='conv_lstm_4',
                             filters=16, kernel_size=(3, 3),
                             padding='same',
                             return_sequences=True)(batch_norm_3)
    timedis1 = TimeDistributed(Dense(units=1, name='dense_1', activation='relu'))(conv_lstm_4)

    convlstm_out = Reshape((6, 6, 1))(timedis1)
    # model.add(Dense(units=1, name = 'dense_2'))
    # poi + tmp
    poi_input = Input(shape=(6, 6, 1), name='poi_input')
    weather_input = Input(shape=(6, 6, 1), name='weather_input')
    x = concatenate([convlstm_out, poi_input, weather_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    main_output = Dense(1, activation='relu', name='main_output')(x)
    model = Model(inputs=[main_input, poi_input, weather_input], outputs=[main_output])
    optimizer = RMSprop()  # lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.9)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mae'])
    return model

# 处理数据
def gener_data(inputdata):
    data_x, data_y = [], []
    for i in range(0, len(inputdata) - time_step):
        data_x.append(inputdata.iloc[i:i + time_step, 1:].copy().values)
        data_y.append(inputdata.iloc[i + time_step, 1:].copy().values)
    data_x = np.array(data_x, dtype=object).reshape([-1, time_step, 6, 6, 1])
    data_y = np.array(data_y, dtype=object).reshape([-1, 6, 6, 1])
    index = [j for j in range(len(data_x))]
    random.shuffle(index)
    data_y = data_y[index]
    data_x = data_x[index]
    return data_x, data_y

# init
def init(model):
    """In keras if you don't run a funcition of the model, the model's wight would be empty [0].
       This is only for weight initilization.
    """
    model.evaluate({'main_input': x_train[0:1, ...], 'poi_input': poi_data1, 'weather_input': weather1},
                   {'main_output': y_train[0:1, ...]}, verbose=0)

# data split all different types

# non-iid - fedshare : 0 ~ 5*144 for client A(IID) .  2-6 o'clock for client B / C ( non-iid bad),  others for clientD (non-iid good)
def split_client_fedshare(num_clients: int, len_dataset: int):
    client_data_list = []
    lenthOfDay = int(len_dataset / 144) - 5
    for client in range(num_clients):
        item = []
        ind = int(client % 4)
        if (ind == 0):
            data_list = np.arange(0, 5*144)
            item.extend(data_list)
        elif (ind == 3):
            for i in range(lenthOfDay):
                startnum1 = 144 * i + 5 * 144
                # 0 - 2点， 6-24点
                data_list = np.arange(startnum1, startnum1 + 12)
                item.extend(data_list)
                data_list2 = np.arange(startnum1 + 36, startnum1 + 144)
                item.extend(data_list2)
        else:
            for i in range(lenthOfDay):
                startnum2 = 144 * i + 5 * 144
                # 2点到6点
                data_list = np.arange(startnum2 + 12, startnum2 + 36)
                item.extend(data_list)
        print(item)
        client_data_list.append(item)

    return client_data_list


# get data  ------------- start -----------
# 不会写相对路径，请手动修改
data = pd.read_excel(r'D:\sy\conv_buy\code\inputdata_10min.xlsx')
poi_data = pd.read_excel(r'D:\sy\conv_buy\code\poi_effect.xlsx')
weather_data = pd.read_excel(r'D:\sy\conv_buy\code\weather_0.xlsx')

data['day'] = data['starttime'].apply(lambda x: x.strftime("%Y-%m-%d"))
# 2023-01-18最后几条数据，需要形成一个timestep=6的数据，有溢出导致1.19也需要放进train_list，整体数据集没有变化。
train_list = ['2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
              '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14', '2023-01-15', '2023-01-16', '2023-01-17',
              '2023-01-18', '2023-01-19']
test_list = ['2023-01-19', '2023-01-20', '2023-01-21', '2023-01-22', '2023-01-23']
train = data[data['day'].isin(train_list)]
test = data[(data['day'].isin(test_list))]
train = train.drop(['day'], axis=1)
test = test.drop(['day'], axis=1)
# 生成训练集和测试集
train_x, train_y = gener_data(train)
test_x, test_y = gener_data(test)
# x为numpy array,返回值也为numpy array但其数据类型变为floatx。
x_train = backend.cast_to_floatx(train_x)
y_train = backend.cast_to_floatx(train_y)
print("train_x.shape & train_y.shape", train_x.shape, train_y.shape)
x_test = backend.cast_to_floatx(test_x)
y_test = backend.cast_to_floatx(test_y)
print("test_x.shape & test_y.shape", x_test.shape, test_y.shape)
len_dataset = x_train.shape[0]
print("len_dataset", len_dataset)

# poi
poi_data_all = poi_data[effectParams[0]].values.reshape([6, 6, 1])
poi_data_all = np.expand_dims(poi_data_all, axis=0)
poi_data1 = np.repeat(poi_data_all, len(y_train), axis=0)
poi_data2 = np.repeat(poi_data_all, len(y_test), axis=0)
print("POI 1 & 2 ", poi_data1.shape, poi_data2.shape)

# weather
weatherall = weather_data[effectParams[1]].values
weather1, weather2 = [], []
for ix in range(0, len(y_train)):
    tempData = np.repeat(weatherall[ix + time_step], 36)
    weather1.append(tempData.reshape([6, 6, 1]))
weather1 = np.array(weather1)
for ix in range(20 * 144 - 1 - len(y_test), 20 * 144 - 1):
    tempData = np.repeat(weatherall[ix], 36)
    weather2.append(tempData.reshape([6, 6, 1]))
weather2 = np.array(weather2)
print("weather 1 & 2 ", weather1.shape, weather2.shape)

global_model = get_model(time_step, 1)
acc_list = []
weight_acc = global_model.get_weights()

# get data  ------------- end -----------

client_data_list = split_client_fedshare(num_clients, len_dataset)
print("split_client_fedshare")

# FL main execute
for round in range(t_round):
    print("Round: " + str(round + 1) + " started.")
    for i in range(len(weight_acc)):
        weight_acc[i] = np.zeros(weight_acc[i].shape)
    weight_acc = np.asarray([weight_acc[0],
                             weight_acc[1], weight_acc[2], weight_acc[3], weight_acc[4], weight_acc[5], weight_acc[6],
                             weight_acc[7]
                                , weight_acc[8], weight_acc[9], weight_acc[10], weight_acc[11],
                             weight_acc[12], weight_acc[13], weight_acc[14], weight_acc[15], weight_acc[16],
                             weight_acc[17], weight_acc[18]
                                , weight_acc[19], weight_acc[20], weight_acc[21], weight_acc[22],
                             weight_acc[23], weight_acc[24], weight_acc[25], weight_acc[26], weight_acc[27],
                             weight_acc[28], weight_acc[29]
                                , weight_acc[30], weight_acc[31], weight_acc[32], weight_acc[33], weight_acc[34],
                             weight_acc[35]
                             ], dtype=object)

    for c in range(num_clients):
        model0 = get_model(time_step, 1)
        # 全局模型复制到自己的模型
        model0.set_weights(global_model.get_weights())
        param_before = np.asarray(model0.get_weights(), dtype=object)

        # Get index list
        ind = client_data_list[c]
        c_feature = np.take(x_train, ind, axis=0)
        c_poi = np.take(poi_data1, ind, axis=0)
        c_tmp = np.take(weather1, ind, axis=0)
        c_label = np.take(y_train, ind, axis=0)

        print(c_feature.shape)
        # Train client
        history = model0.fit({'main_input': c_feature, 'poi_input': c_poi, 'weather_input': c_tmp},
                             {'main_output': c_label},
                             batch_size=batch_size,
                             epochs=epochs,
                             shuffle=False,
                             validation_data=({'main_input': c_feature, 'poi_input': c_poi, 'weather_input': c_tmp},
                                              {'main_output': c_label}),
                             verbose=2)

        param_after = np.asarray(model0.get_weights(), dtype=object)

        weight_acc += (param_after) * (1 / num_clients)
        score = model0.evaluate({'main_input': x_test, 'poi_input': poi_data2, 'weather_input': weather2},
                                {'main_output': y_test})


        def calculate_mse(data_x, poi_data, weather_data, data_y):
            data_y = data_y.reshape([-1])
            # index = list(np.nonzero(data_y)[0])
            # data_y = np.array([data_y[i] for i in index])
            predict = model0.predict([data_x, poi_data, weather_data])
            predict = predict.reshape([-1])
            # predict = np.array([predict[i] for i in index])
            MSE = np.mean(np.power((data_y - predict), 2))
            R2 = 1 - MSE / np.var(data_y)
            return MSE, R2


        mse, r2 = calculate_mse(x_test, poi_data2, weather2, y_test)

        print('round:' + str(round + 1) + 'client:' + str(c + 1) + ' output: loss / accuracy / mse / r2 ', score[0],
              score[1], mse, r2)

        acc_list.append([round + 1, score[0], score[1], mse, r2])

    global_model.set_weights(weight_acc)

    gscore = global_model.evaluate({'main_input': x_test, 'poi_input': poi_data2, 'weather_input': weather2},
                                   {'main_output': y_test})

    def calculate_mse(data_x, poi_data, weather_data, data_y):
        data_y = data_y.reshape([-1])
        # index = list(np.nonzero(data_y)[0])
        # data_y = np.array([data_y[i] for i in index])
        predict = global_model.predict([data_x, poi_data, weather_data])
        predict = predict.reshape([-1])
        # predict = np.array([predict[i] for i in index])
        MSE = np.mean(np.power((data_y - predict), 2))
        R2 = 1 - MSE / np.var(data_y)
        return MSE, R2


    gmse, gr2 = calculate_mse(x_test, poi_data2, weather2, y_test)
    # score = global_model.evaluate(x_test, y_test, verbose=0)
    print('round:' + str(round + 1) + 'output:  loss / accuracy / gmse / gr2 ', gscore[0], gscore[1], gmse, gr2)
    acc_list.append([str(round + 1), gscore[0], gscore[1], gmse, gr2])

# outputs for each round in each client
print(acc_list)
