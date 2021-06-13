'''Train RAF/ExpW with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils
import losses
from datasets.RAF import RAF_teacher
from datasets.ExpW import ExpW_teacher
from datasets.CK_Plus import CK_Plus_teacher
from torch.autograd import Variable
from teacherNet import Teacher
from teacherNet1 import Teacher1
from teacherNet3 import Teacher3
from thop import profile
from tensorboardX import SummaryWriter
from utils import load_pretrained_model
from torchtoolbox.transform import Cutout
from torchtoolbox.tools import mixup_data, mixup_criterion

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--model', type=str, default="Teacher", help='Teacher,Teacher1,Teacher3')
parser.add_argument('--data_name', type=str, default="CK_Plus", help='RAF, ExpW, CK_Plus')
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='Batch size')
parser.add_argument('--test_bs', default=8, type=int, help='Batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--augmentation', default=False, type=int, help='use mixup and cutout')
parser.add_argument('--perTraining', type=str, default='100%', help='Percentage of training')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0
best_mAP = 0
best_F1 = 0
NUM_CLASSES = 7
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

total_epoch = args.epochs
total_prediction_fps = 0 
total_prediction_n = 0

path = os.path.join(args.save_root + args.data_name + '_' + args.model + '_' + str(args.augmentation) + '_' + str(args.perTraining))
writer = SummaryWriter(log_dir=path)

# Data
print ('The dataset used for training is:                '+ str(args.data_name))
print ('The training mode is:                            '+ str(args.model))
print ('Whether to use data enhancement:                 '+ str(args.augmentation))
print ('Percentage of training:                          '+ str(args.perTraining))
print('==> Preparing data..')

if args.augmentation == False and args.data_name == 'RAF':
	transform_train = transforms.Compose([
		transforms.RandomCrop(92),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5884594, 0.45767313, 0.40865755), 
			(0.25717735, 0.23602168, 0.23505741)),
		])

	transform_test = transforms.Compose([
		transforms.TenCrop(92),
		transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
			mean=[0.589667, 0.45717254, 0.40727714], std=[0.25235596, 0.23242524, 0.23155019])
			(transforms.ToTensor()(crop)) for crop in crops])),
		])

elif args.augmentation == False and args.data_name == 'ExpW':
	transform_train = transforms.Compose([
		transforms.RandomCrop(92),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.6199751, 0.46946654, 0.4103778), 
			(0.25622123, 0.22915973, 0.2232292)),
		])

	transform_test = transforms.Compose([
		transforms.TenCrop(92),
		transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
			mean=[0.6081647, 0.4579959, 0.3987486], std=[0.25485262, 0.22496806, 0.21835831])
			(transforms.ToTensor()(crop)) for crop in crops])),
		])

elif args.augmentation == False and args.data_name == 'CK_Plus':
	transform_train = transforms.Compose([
		transforms.RandomCrop(92),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5950821, 0.59496826, 0.5949638), 
			(0.2783952, 0.27837786, 0.27837303)),
		])

	transform_test = transforms.Compose([
		transforms.TenCrop(92),
		transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
			mean=[0.5283895, 0.49888685, 0.48856217], std=[0.22694704, 0.21892785, 0.22059701])
			(transforms.ToTensor()(crop)) for crop in crops])),
		])

else:
	raise Exception('Invalid ...')

if args.data_name == 'RAF':
	trainset = RAF_teacher(split = 'Training', transform=transform_train, percentage_training=args.perTraining)
	PrivateTestset = RAF_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'ExpW':
	trainset = ExpW_teacher(split = 'Training', transform=transform_train, percentage_training=args.perTraining)
	PrivateTestset = ExpW_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'CK_Plus':
	trainset = CK_Plus_teacher(split = 'Training', transform=transform_train, percentage_training=args.perTraining)
	PrivateTestset = CK_Plus_teacher(split = 'PrivateTest', transform=transform_test)
else:
	raise Exception('Invalid dataset name...')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=1)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.test_bs, shuffle=False, num_workers=1)

# Model
if args.model == 'Teacher':
    net = Teacher()
elif args.model == 'Teacher1':
    net = Teacher1()
elif args.model == 'Teacher3':
    net = Teacher3()
else:
    raise Exception('.............Invalid..............')


#tcheckpoint = torch.load(os.path.join(path,'Best_Teacher_model.t7'))
#load_pretrained_model(net, tcheckpoint['tnet'])
#print ('best_Teacher_acc is '+ str(tcheckpoint['test_acc']))  #  best_PrivateTest_acc, test_acc

#flops, params = profile(net, inputs=(torch.randn(1, 3, 92,92),))
#flops, params = profile(net, input_size=(1, 3, 92,92))
#print("The FLOS of this model is  %0.3f M" % float(flops/1024/1024))
#print("The params of this model is  %0.3f M" % float(params/1024/1024))

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat_a = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat_b = np.zeros((NUM_CLASSES, NUM_CLASSES))

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = args.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = args.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        if args.augmentation:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 0.6)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        else:
            inputs, targets = Variable(inputs), Variable(targets)
        
        _, _, _, _, outputs = net(inputs)
        
        if args.augmentation:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        
        if args.augmentation:
                conf_mat_a += losses.confusion_matrix(outputs, targets_a, NUM_CLASSES)
                acc_a = sum([conf_mat_a[i, i] for i in range(conf_mat_a.shape[0])])/conf_mat_a.sum()
                precision_a = np.array([conf_mat_a[i, i]/(conf_mat_a[i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
                recall_a = np.array([conf_mat_a[i, i]/(conf_mat_a[:, i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
                mAP_a = sum(precision_a)/len(precision_a)
                F1_score_a = (2 * precision_a*recall_a/(precision_a+recall_a + 1e-10)).mean()

                conf_mat_b += losses.confusion_matrix(outputs, targets_b, NUM_CLASSES)
                acc_b = sum([conf_mat_b[i, i] for i in range(conf_mat_b.shape[0])])/conf_mat_b.sum()
                precision_b = np.array([conf_mat_b[i, i]/(conf_mat_b[i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
                recall_b = np.array([conf_mat_b[i, i]/(conf_mat_b[:, i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
                mAP_b = sum(precision_b)/len(precision_b)
                F1_score_b = (2 * precision_b*recall_b/(precision_b+recall_b + 1e-10)).mean()

                acc = lam * acc_a  +  (1 - lam) * acc_b
                mAP = lam * mAP_a  +  (1 - lam) * mAP_b
                F1_score = lam * F1_score_a  +  (1 - lam) * F1_score_b

        else:
                conf_mat += losses.confusion_matrix(outputs, targets, NUM_CLASSES)
                acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
                precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
                mAP = sum(precision)/len(precision)

                recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
                precision = np.array(precision)
                recall = np.array(recall)
                f1 = 2 * precision*recall/(precision+recall + 1e-10)
                F1_score = f1.mean()

        #utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% | mAP: %.3f%% | F1: %.3f%%'
                #% (train_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100.* F1_score))
   
    return train_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score

def test(epoch):
    global total_prediction_fps
    global total_prediction_n
    net.eval()
    PrivateTest_loss = 0
    t_prediction = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        t = time.time()
        test_bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, _, _, _, outputs = net(inputs)
        outputs_avg = outputs.view(test_bs, ncrops, -1).mean(1)  # avg over crops
        t_prediction += (time.time() - t)
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        
        conf_mat += losses.confusion_matrix(outputs_avg, targets, NUM_CLASSES)
        acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
        precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        mAP = sum(precision)/len(precision)

        recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        precision = np.array(precision)
        recall = np.array(recall)
        f1 = 2 * precision*recall/(precision+recall + 1e-10)
        F1_score = f1.mean()

        #utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% | mAP: %.3f%% | F1: %.3f%%'
			#% (PrivateTest_loss / (batch_idx + 1), 100.*acc, 100.* mAP, 100.* F1_score))
    total_prediction_fps = total_prediction_fps + (1 / (t_prediction / len(PrivateTestloader)))
    total_prediction_n = total_prediction_n + 1
    print('Prediction time: %.2f' % t_prediction + ', Average : %.5f/image' % (t_prediction / len(PrivateTestloader)) 
         + ', Speed : %.2fFPS' % (1 / (t_prediction / len(PrivateTestloader))))
    
    return PrivateTest_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score


for epoch in range(0, total_epoch):
	# train one epoch
	train_loss, train_acc, train_mAP, train_F1 = train(epoch)
	# evaluate on testing set
	test_loss, test_acc, test_mAP, test_F1 = test(epoch)

	print("train_loss:  %0.3f, train_acc:  %0.3f, train_mAP:  %0.3f, train_F1:  %0.3f"%(train_loss, train_acc, train_mAP, train_F1))
	print("test_loss:   %0.3f, test_acc:   %0.3f, test_mAP:   %0.3f, test_F1:   %0.3f"%(test_loss, test_acc, test_mAP, test_F1))

	writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
	writer.add_scalars('epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch)
	writer.add_scalars('epoch/mAP', {'train': train_mAP, 'test': test_mAP}, epoch)
	writer.add_scalars('epoch/F1', {'train': train_F1, 'test': test_F1}, epoch)

	# save model
	if test_acc > best_acc:
		best_acc = test_acc
		best_mAP = test_mAP
		best_F1 = test_F1
		print ('Saving models......')
		print("best_PrivateTest_acc: %0.3f" % best_acc)
		print("best_PrivateTest_mAP: %0.3f" % best_mAP)
		print("best_PrivateTest_F1: %0.3f" % best_F1)
		state = {
			'tnet': net.state_dict() if use_cuda else net,
			'test_acc': test_acc,
			'test_mAP': test_mAP,
			'test_F1': test_F1,
			'test_epoch': epoch,
		} 
		if not os.path.isdir(path):
				os.mkdir(path)
		torch.save(state, os.path.join(path,'Best_Teacher_model.t7'))

print("best_PrivateTest_acc: %0.3f" % best_acc)
print("best_PrivateTest_mAP: %0.3f" % best_mAP)
print("best_PrivateTest_F1: %0.3f" % best_F1)

print("total_prediction_fps: %0.2f" % total_prediction_fps)
print("total_prediction_n: %d" % total_prediction_n)
print('Average speed: %.2f FPS' % (total_prediction_fps / total_prediction_n))
writer.close()
