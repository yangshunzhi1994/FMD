''' RAF-DB Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import os
import cv2
import random
import torchvision.transforms

def sp_noise(image,prob):
  '''
  添加椒盐噪声
  prob:噪声比例 
  '''
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

class RAF(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None, student_norm=None, teacher_norm=None, noise=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.noise = noise
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))
        
        
        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            #img.show()

            img_student = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            if self.noise == 'GaussianBlur':
                img_student = cv2.GaussianBlur(img_student,(5,5),0)
            elif self.noise == 'AverageBlur':
                img_student = cv2.blur(img_student,(5,5))
            elif self.noise == 'MedianBlur':
                img_student = cv2.medianBlur(img_student,5)
            elif self.noise == 'BilateralFilter':
                img_student = cv2.bilateralFilter(img_student,10,100,100)
            elif self.noise == 'Salt-and-pepper':
                img_student = sp_noise(img_student, prob=0.05)
            else:
                pass
            #cv2.imwrite('111.png', img_student)
            img_student = Image.fromarray(cv2.cvtColor(img_student,cv2.COLOR_BGR2RGB))

            img_student = self.student_norm(img_student)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
            img = Image.fromarray(img)
            img_student = torchvision.transforms.Resize(48)(img)

            img_student = cv2.cvtColor(np.asarray(img_student),cv2.COLOR_RGB2BGR)
            if self.noise == 'GaussianBlur':
                img_student = cv2.GaussianBlur(img_student,(5,5),0)
            elif self.noise == 'AverageBlur':
                img_student = cv2.blur(img_student,(5,5))
            elif self.noise == 'MedianBlur':
                img_student = cv2.medianBlur(img_student,5)
            elif self.noise == 'BilateralFilter':
                img_student = cv2.bilateralFilter(img_student,10,100,100)
            elif self.noise == 'Salt-and-pepper':
                img_student = sp_noise(img_student, prob=0.05)
            else:
                pass
            #cv2.imwrite('111.png', img_student)
            img_student = Image.fromarray(cv2.cvtColor(img_student,cv2.COLOR_BGR2RGB))

            img_student = self.transform(img_student)

            return img_student, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)

class RAF_teacher(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None, percentage_training='100%'):
        self.transform = transform
        self.split = split  # training set or test set
        self.percentage_training = percentage_training
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))

            if self.percentage_training == '20%':
                self.train_data = self.train_data[0:2455]
            elif self.percentage_training == '40%':
                self.train_data = self.train_data[0:4909]
            elif self.percentage_training == '60%':
                self.train_data = self.train_data[0:7363]
            elif self.percentage_training == '80%':
                self.train_data = self.train_data[0:9817]
            elif self.percentage_training == '100%':
                pass
            else:
                raise Exception('Invalid percentage_training...')

        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
#         img = img[:, :, np.newaxis]
#         img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)



class RAF_student(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None, noise=None, percentage_training='100%'):
        self.transform = transform
        self.split = split
        self.noise = noise
        self.percentage_training = percentage_training
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))

            if self.percentage_training == '20%':
                self.train_data = self.train_data[0:2455]
            elif self.percentage_training == '40%':
                self.train_data = self.train_data[0:4909]
            elif self.percentage_training == '60%':
                self.train_data = self.train_data[0:7363]
            elif self.percentage_training == '80%':
                self.train_data = self.train_data[0:9817]
            elif self.percentage_training == '100%':
                pass
            else:
                raise Exception('Invalid percentage_training...')
        
        
        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = torchvision.transforms.Resize(48)(img)
            img = self.transform(img)
            
            return img, target

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
            img = Image.fromarray(img)
            img = torchvision.transforms.Resize(48)(img)

            img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            if self.noise == 'GaussianBlur':
                img = cv2.GaussianBlur(img,(5,5),0)
            elif self.noise == 'AverageBlur':
                img = cv2.blur(img,(5,5))
            elif self.noise == 'MedianBlur':
                img = cv2.medianBlur(img,5)
            elif self.noise == 'BilateralFilter':
                img = cv2.bilateralFilter(img,10,100,100)
            elif self.noise == 'Salt-and-pepper':
                img = sp_noise(img, prob=0.05)
            else:
                pass
            #cv2.imwrite('111.png', img_student)
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

            img = self.transform(img)

            return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)


