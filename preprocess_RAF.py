# create data and label for RAF
#  0=Anger 1=Disgust 2=Fear 3=Happiness 4=Sadness 5=Surprise 6=Neutra
# Training 705 717 281 4772 1982 1290 2524
# Test 162 160 74 1185 478 329 680

import csv
import os
import numpy as np
import h5py
import skimage.io
import cv2 as cv

train_path = 'RAF/train'
valid_path = 'RAF/valid'

train_anger_path = os.path.join(train_path, '5')
train_disgust_path = os.path.join(train_path, '2')
train_fear_path = os.path.join(train_path, '1')
train_happy_path = os.path.join(train_path, '3')
train_sadness_path = os.path.join(train_path, '4')
train_surprise_path = os.path.join(train_path, '0')
train_contempt_path = os.path.join(train_path, '6')
# # Creat the list to store the data and label information
train_data_x = []
train_data_y = []

valid_anger_path = os.path.join(valid_path, '5')
valid_disgust_path = os.path.join(valid_path, '2')
valid_fear_path = os.path.join(valid_path, '1')
valid_happy_path = os.path.join(valid_path, '3')
valid_sadness_path = os.path.join(valid_path, '4')
valid_surprise_path = os.path.join(valid_path, '0')
valid_contempt_path = os.path.join(valid_path, '6')
# # Creat the list to store the data and label information
valid_data_x = []
valid_data_y = []


# order the file, so the training set will not contain the test set (don't random)
files = os.listdir(train_anger_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_anger_path,filename))
    train_data_x.append(I.tolist())
    train_data_y.append(0)
print (1)
files = os.listdir(train_disgust_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_disgust_path,filename))
    train_data_x.append(I.tolist())
    train_data_y.append(1)
print (1)
files = os.listdir(train_fear_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_fear_path,filename))
    train_data_x.append(I.tolist())
    train_data_y.append(2)
print (1)
files = os.listdir(train_happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_happy_path,filename))
    train_data_x.append(I.tolist())
    train_data_y.append(3)
print (1)
files = os.listdir(train_sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_sadness_path,filename))
    train_data_x.append(I.tolist())
    train_data_y.append(4)
print (1)
files = os.listdir(train_surprise_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_surprise_path,filename))
    train_data_x.append(I.tolist())
    train_data_y.append(5)
print (1)
files = os.listdir(train_contempt_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(train_contempt_path,filename))
    train_data_x.append(I.tolist())
    train_data_y.append(6)

print(np.shape(train_data_x))
print(np.shape(train_data_y))


# order the file, so the valid set will not contain the test set (don't random)
files = os.listdir(valid_anger_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_anger_path,filename))
    valid_data_x.append(I.tolist())
    valid_data_y.append(0)

files = os.listdir(valid_disgust_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_disgust_path,filename))
    valid_data_x.append(I.tolist())
    valid_data_y.append(1)

files = os.listdir(valid_fear_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_fear_path,filename))
    valid_data_x.append(I.tolist())
    valid_data_y.append(2)

files = os.listdir(valid_happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_happy_path,filename))
    valid_data_x.append(I.tolist())
    valid_data_y.append(3)

files = os.listdir(valid_sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_sadness_path,filename))
    valid_data_x.append(I.tolist())
    valid_data_y.append(4)

files = os.listdir(valid_surprise_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_surprise_path,filename))
    valid_data_x.append(I.tolist())
    valid_data_y.append(5)

files = os.listdir(valid_contempt_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(valid_contempt_path,filename))
    valid_data_x.append(I.tolist())
    valid_data_y.append(6)

print(np.shape(valid_data_x))
print(np.shape(valid_data_y))



datafile = h5py.File('RAF_data_100.h5', 'w')
datafile.create_dataset("train_data_pixel", dtype = 'uint8', data=train_data_x)
datafile.create_dataset("train_data_label", dtype = 'int64', data=train_data_y)
datafile.create_dataset("valid_data_pixel", dtype = 'uint8', data=valid_data_x)
datafile.create_dataset("valid_data_label", dtype = 'int64', data=valid_data_y)
datafile.close()

print("Save data finish!!!")

'''
with open('RAF/list_patition_label.txt','r') as f:
	for line in f:
		result = line.strip('\n').split(',')[0]
		result_img = result.split(' ')[0].split('.')[0]
		result_lab = result.split(' ')[1]
		#print (result_img)
		#print (result_lab)
		for filename in os.listdir(r"RAF/aligned/"):
			if str(result_img)  in str(filename):
				I = skimage.io.imread(os.path.join('RAF/aligned/',filename))
				train_test = result_img.split('_')[0]
				if train_test  == 'train':
					train_data_x.append(I.tolist())
					train_data_y.append(int(result_lab))
				else:
					valid_data_x.append(I.tolist())
					valid_data_y.append(int(result_lab))

'''
