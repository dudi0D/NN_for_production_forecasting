import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode
import matplotlib as mp

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
plt.figure(1)
for i in datas:
    plt.scatter(i['Годы'], i['Год. доб. нефти, тыс.т'])
plt.xlabel('Years')
plt.ylabel('Annual oil prod., thd. tons')
names = [i.replace('Скважина', 'Well') for i in names]
names = [i.replace('действующий фонд', 'active fund') for i in names]
names = [i.replace('безactive fund', 'inactive fund') for i in names]
names = [i.replace('в консервации', 'in conservation') for i in names]
plt.legend(names)

mp.style.use('ggplot')
listing = [data1['Год. доб. газа, млн.м3'], data2['Год. доб. газа, млн.м3'], data3['Год. доб. газа, млн.м3'],
           data4['Год. доб. газа, млн.м3'], data5['Год. доб. газа, млн.м3'], data6['Год. доб. газа, млн.м3'],
           data7['Год. доб. газа, млн.м3'], data8['Год. доб. газа, млн.м3'],
           data9['Год. доб. газа, млн.м3'], data10['Год. доб. газа, млн.м3']]
plt.figure(2)
for i in listing:
    plt.hist(i, orientation="horizontal")
plt.legend(names)
plt.xlabel('Years')
plt.ylabel('Annual gas production, mln. m3')
datas_annual_oil_variety = []
datas_annual_oil_expected_value = []
datas_annual_oil_moda = []
datas_annual_oil_scope = []
datas_variety_coefficient = []
for i in datas:
    variety_coefficient = []
    for j in i:
        column_variety_coefficient = np.std(i[j])/np.mean(i[j])
        if np.std(i[j]) != 0.0:
            variety_coefficient.append(column_variety_coefficient)
    datas_variety_coefficient.append(variety_coefficient)
for i in datas:
    datas_annual_oil_variety.append(np.std(i['Год. доб. нефти, тыс.т']))
    datas_annual_oil_expected_value.append(np.var(i['Год. доб. нефти, тыс.т']))
    datas_annual_oil_moda.append(mode(i['Год. доб. нефти, тыс.т']))
    datas_annual_oil_scope.append(np.max(i['Год. доб. нефти, тыс.т'])-np.min(i['Год. доб. нефти, тыс.т']))

for i in range(len(datas_annual_oil_variety)):
    if datas_annual_oil_variety[i] == np.max(datas_annual_oil_variety):
        print(f'Well of maximum dispersion spread: {names[i]}')
    if datas_annual_oil_variety[i] == np.min(datas_annual_oil_variety):
        print(f'Well of minimum dispersion spread: {names[i]}')
for i in range(len(datas_annual_oil_expected_value)):
    if datas_annual_oil_expected_value[i] == np.max(datas_annual_oil_expected_value):
        print(f'Well of maximum mathematical expectation: {names[i]}')
    if datas_annual_oil_expected_value[i] == np.min(datas_annual_oil_expected_value):
        print(f'Well of minimum mathematical expectation: {names[i]}')
print(f'Wells close in terms of annual oil production values: {names[2]} & {names[3]}')
plt.figure(3)
datas_annual_oil_variety_plot = plt.subplot(2, 2, 1)
datas_annual_oil_variety_plot.plot(datas_annual_oil_variety, '--o', color='red')
datas_annual_oil_variety_plot.set_xlabel('Dispersion')
datas_annual_oil_expected_value_plot = plt.subplot(2, 2, 2)
datas_annual_oil_expected_value_plot.plot(datas_annual_oil_expected_value, '--o', color='blue')
datas_annual_oil_expected_value_plot.set_xlabel('Mathematical expectation')
datas_annual_oil_moda_plot = plt.subplot(2, 2, 3)
datas_annual_oil_moda_plot.plot(datas_annual_oil_moda, '--o', color='green')
datas_annual_oil_moda_plot.set_xlabel('Mode')
datas_annual_oil_scope_plot = plt.subplot(2, 2, 4)
datas_annual_oil_scope_plot.plot(datas_annual_oil_scope, '--o', color='peru')
datas_annual_oil_scope_plot.set_xlabel('Sample range')
for i in range(len(datas_variety_coefficient)):
    datas_variety_coefficient[i] = np.mean(datas_variety_coefficient[i])
datas_variety_coefficient = np.mean(datas_variety_coefficient)
print(f'Variation coefficient: {datas_variety_coefficient}')
plt.show()
