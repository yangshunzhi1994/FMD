# -*- coding: utf-8 -*-
#创建hdf5文件,ExpW数据集很大，如果内存不够，使用本文件分步处理。

import datetime
import os
import h5py
import numpy as np

ExpW0 = h5py.File('ExpW_data0.h5','r')
ExpW1 = h5py.File('ExpW_data1.h5','r')
ExpW2 = h5py.File('ExpW_data2.h5','r')
ExpW3 = h5py.File('ExpW_data3.h5','r')
ExpW4 = h5py.File('ExpW_data4.h5','r')
ExpW5 = h5py.File('ExpW_data5.h5','r')
ExpW6 = h5py.File('ExpW_data6.h5','r')

train_data_x = []
train_data_y = []
valid_data_x = []
valid_data_y = []


train_data_x = np.concatenate((ExpW0['train_data_pixel'],ExpW1['train_data_pixel'],ExpW2['train_data_pixel'],ExpW3['train_data_pixel'],ExpW4['train_data_pixel'],ExpW5['train_data_pixel'],ExpW6['train_data_pixel']),axis=0)
train_data_y = np.concatenate((ExpW0['train_data_label'],ExpW1['train_data_label'],ExpW2['train_data_label'],ExpW3['train_data_label'],ExpW4['train_data_label'],ExpW5['train_data_label'],ExpW6['train_data_label']),axis=0)
valid_data_x = np.concatenate((ExpW0['valid_data_pixel'],ExpW1['valid_data_pixel'],ExpW2['valid_data_pixel'],ExpW3['valid_data_pixel'],ExpW4['valid_data_pixel'],ExpW5['valid_data_pixel'],ExpW6['valid_data_pixel']),axis=0)
valid_data_y = np.concatenate((ExpW0['valid_data_label'],ExpW1['valid_data_label'],ExpW2['valid_data_label'],ExpW3['valid_data_label'],ExpW4['valid_data_label'],ExpW5['valid_data_label'],ExpW6['valid_data_label']),axis=0)


print  (train_data_y.shape)
print  (valid_data_y.shape)

datafile = h5py.File('ExpW_data_100.h5', 'w')
datafile.create_dataset("train_data_pixel", dtype = 'uint8', data=train_data_x)
datafile.create_dataset("train_data_label", dtype = 'int64', data=train_data_y)
datafile.create_dataset("valid_data_pixel", dtype = 'uint8', data=valid_data_x)
datafile.create_dataset("valid_data_label", dtype = 'int64', data=valid_data_y)
datafile.close()

print("Save data finish!!!")
ExpW100 = h5py.File('ExpW_data_100.h5','r')
print  (ExpW100['train_data_pixel'])
print  (ExpW100['train_data_label'])
print  (ExpW100['valid_data_pixel'])
print  (ExpW100['valid_data_label'])




