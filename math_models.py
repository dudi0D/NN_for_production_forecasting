from keras import backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from statistics import mode
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, GRU, SimpleRNN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras


class FuzzyLayer(keras.layers.Layer):

    def __init__(self,
                 output_dim,
                 initial_centers=None,
                 initial_sigmas=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.output_dim = output_dim
        self.initial_centers = initial_centers
        self.initial_sigmas = initial_sigmas
        super(FuzzyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]

        c_init_values = []

        if self.initial_centers is None:
            c_init_values = tf.random_uniform_initializer(-1, 1)(shape=(input_shape[-1], self.output_dim),
                                                                 dtype="float32")
        else:
            c_init_values = tf.convert_to_tensor(self.initial_centers, dtype="float32")
        self.c = tf.Variable(initial_value=c_init_values, trainable=True)

        a_init_values = []
        if self.initial_sigmas is None:
            a_init_values = tf.ones_initializer()(shape=(input_shape[-1], self.output_dim), dtype="float32")
        else:
            a_init_values = tf.convert_to_tensor(self.initial_sigmas, dtype="float32")
        self.a = tf.Variable(initial_value=a_init_values, trainable=True)
        super(FuzzyLayer, self).build(input_shape)

    def call(self, x):

        aligned_x = K.repeat_elements(K.expand_dims(x, axis=-1), self.output_dim, -1)
        aligned_c = self.c
        aligned_a = self.a
        for dim in self.input_dimensions:
            aligned_c = K.repeat_elements(K.expand_dims(aligned_c, 0), dim, 0)
            aligned_a = K.repeat_elements(K.expand_dims(aligned_a, 0), dim, 0)

        xc = K.exp(-K.sum(K.square((aligned_x - aligned_c) / (2 * aligned_a)), axis=-2, keepdims=False))

        return xc

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)


data1 = pd.read_csv('Скважина 20332 (действующий фонд).csv', sep=';', decimal=",")
data2 = pd.read_csv('Скважина 20336 (действующий фонд).csv', sep=';', decimal=",")
data3 = pd.read_csv('Скважина 20343 (действующий фонд).csv', sep=';', decimal=",")
data4 = pd.read_csv('Скважина 20402 (бездействующий фонд).csv', sep=';', decimal=",")
data5 = pd.read_csv('Скважина 20422 (действующий фонд).csv', sep=';', decimal=",")
data6 = pd.read_csv('Скважина 20445 (бездействующий фонд).csv', sep=';', decimal=",")
data7 = pd.read_csv('Скважина 20454 (действующий фонд).csv', sep=';', decimal=",")
data8 = pd.read_csv('Скважина 20490 (в консервации).csv', sep=';', decimal=",")
data9 = pd.read_csv('Скважина 20491 (в консервации).csv', sep=';', decimal=",")
data10 = pd.read_csv('Скважина 20501 (действующий фонд).csv', sep=';', decimal=",")
datas = [data10, data9, data8, data7, data6, data5, data4, data3, data2, data1]
names = ['Скважина 20332 (действующий фонд)', 'Скважина 20336 (действующий фонд)', 'Скважина 20343 (действующий фонд)',
         'Скважина 20402 (бездействующий фонд)', 'Скважина 20422 (действующий фонд)',
         'Скважина 20445 (бездействующий фонд)', 'Скважина 20454 (действующий фонд)', 'Скважина 20490 (в консервации)',
         'Скважина 20491 (в консервации)', 'Скважина 20501 (действующий фонд)']
listing = [data1['Год. доб. газа, млн.м3'], data2['Год. доб. газа, млн.м3'], data3['Год. доб. газа, млн.м3'],
           data4['Год. доб. газа, млн.м3'], data5['Год. доб. газа, млн.м3'], data6['Год. доб. газа, млн.м3'],
           data7['Год. доб. газа, млн.м3'], data8['Год. доб. газа, млн.м3'],
           data9['Год. доб. газа, млн.м3'], data10['Год. доб. газа, млн.м3']]
datas_annual_oil_variety = []
datas_annual_oil_expected_value = []
datas_annual_oil_moda = []
datas_annual_oil_scope = []
datas_variety_coefficient = []
for i in datas:
    variety_coefficient = []
    for j in i:
        column_variety_coefficient = np.std(i[j]) / np.mean(i[j])
        if np.std(i[j]) != 0.0:
            variety_coefficient.append(column_variety_coefficient)
    datas_variety_coefficient.append(variety_coefficient)
for i in datas:
    datas_annual_oil_variety.append(np.std(i['Год. доб. нефти, тыс.т']))
    datas_annual_oil_expected_value.append(np.var(i['Год. доб. нефти, тыс.т']))
    datas_annual_oil_moda.append(mode(i['Год. доб. нефти, тыс.т']))
    datas_annual_oil_scope.append(np.max(i['Год. доб. нефти, тыс.т']) - np.min(i['Год. доб. нефти, тыс.т']))
name_of_predicting_well = ''
for i in range(len(datas_annual_oil_variety)):
    if datas_annual_oil_expected_value[i] == np.max(datas_annual_oil_expected_value):
        print(f'Скважина максимального математического ожидания: {names[i]}')
        name_of_predicting_well = names[i] + '.csv'
        break

data = pd.read_csv('Скважина 20445 (бездействующий фонд).csv', sep=';', decimal=',')
x = data.values
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)
print(data)
X = data.loc[:, 2]
Y = data.loc[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(Dense(64, input_shape=(14, 1)))
# model.add(GRU(300, input_shape=(14, 1), input_dim=2, return_sequences=True))
# # model.add(Dropout(0.5))
# model.add(GRU(300, input_shape=(14, 1), activation='sigmoid', input_dim=2, return_sequences=True))
# for i in range(7):
#     model.add(GRU(300, activation='tanh', input_dim=2, return_sequences=True))
# model.add(Dropout(0.5))
model.add(SimpleRNN(300, input_shape=(14, 1)))
model.add(Dropout(0.2))
# model.add(LSTM(300, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(LSTM(64))
# model.add(FuzzyLayer(300))
# model.add(GRU(64, input_dim=2, input_shape=(64, ), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(GRU(64, input_dim=2, input_shape=(64, )))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
# model.add(Dense(1, activation='tanh'))
epochs = 100
learning_rate = 0.0001
decay_rate = learning_rate / epochs
model.compile(loss='msle', optimizer=keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate))
estimation_object = model.fit(X_train, Y_train, epochs=epochs, batch_size=32, verbose=1,
                              validation_split=0.25)
predict = model.predict(X_test)
predicted = predict.flatten()
original = Y_test.values
plt.figure(1)
plt.plot(predicted, color='blue', label='Predicted data')
plt.plot(original, color='red', label='Original data')
plt.legend(loc='best')
plt.title('Actual and predicted')
model.summary()
# model.save('model_fuzzy_1.h5')
predicted_and_original_difference = []
for i, j in enumerate(original):
    predicted_and_original_difference.append(abs(predicted[i] - j))
print(np.mean(predicted_and_original_difference))
loss = []
for i in estimation_object.history['loss']:
    loss.append(i)
print(np.min(loss))
plt.figure(2)
plt.plot(loss)
plt.ylabel('MSE')
plt.xlabel('epochs')
plt.show()
# plt.figure(2)
# datas_annual_oil = []
# print(data)
# for i in data[0]:
#     datas_annual_oil.append(i)
# plt.plot(datas_annual_oil)
# plt.plot(predicted)
# plt.show()
