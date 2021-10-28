#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
import itertools
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from datasets.RAF import RAF
from datasets.ExpW import ExpW
from datasets.CK_Plus import CK_Plus
from teacherNet import Teacher
from studentNet import CNN_RIS
from torchtoolbox.transform import Cutout
from torchtoolbox.tools import mixup_data, mixup_criterion

import utils
from utils import load_pretrained_model, count_parameters_in_MB

import losses
import other
from thop import profile
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='results/other/FT/', help='models and logs are saved here')
parser.add_argument('--t_model', type=str, default="results/CK_Plus_Teacher_False/Best_Teacher_model.t7", help='path name of teacher model')
parser.add_argument('--model', type=str, default="FT", help='Fitnet,AT,NST,PKT,AB,CCKD,RKD,SP,VID,OFD,AFDS,FT,CD,FAKD,VKD,RAD')
parser.add_argument('--stage', type=str, default="Block12", help='Block1,Block2,Block12')
parser.add_argument('--data_name', type=str, default='CK_Plus', help='ExpW,RAF,CK_Plus') 

# training hyper parameters
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='learning rate')
parser.add_argument('--test_bs', default=256, type=int, help='learning rate')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')#1e-4,5e-4
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--alpha', type=float, default=0.1, help='0.1,0.4,0.7,0.9')
parser.add_argument('--beta', type=float, default=0.1, help='0.1,0.4,0.7,0.9')
parser.add_argument('--gamma', type=float, default=0.1, help='0.1,0.4,0.7,0.9')
parser.add_argument('--delta', type=float, default=0.1, help='0.1,0.4,0.7,0.9')
parser.add_argument('--T', type=int, default=1, help='1,2,3...')
parser.add_argument('--augmentation', default=False, type=int, help='use mixup and cutout')
parser.add_argument('--noise', type=str, default='None', help='GaussianBlur,AverageBlur,MedianBlur,BilateralFilter,Salt-and-pepper') 

args, unparsed = parser.parse_known_args()

path = os.path.join(args.save_root + args.data_name+ '_CNNRIS_' + args.model+ '_' + args.stage+ '_' + str(args.augmentation))
writer = SummaryWriter(log_dir=path)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)
	cudnn.enabled = True
	cudnn.benchmark = True
snet = CNN_RIS()
#scheckpoint = torch.load(os.path.join(path,'Student_Test_model.t7'))
#load_pretrained_model(snet, scheckpoint['snet'])
#print ('best_Student_acc is '+ str(scheckpoint['test_acc']))

tnet = Teacher()
tcheckpoint = torch.load(args.t_model)
load_pretrained_model(tnet, tcheckpoint['tnet'])
try:
	print ('best_Teacher_acc is '+ str(tcheckpoint['test_acc']))  
except:
	print ('best_Teacher_acc is '+ str(tcheckpoint['best_PrivateTest_acc']))

print ('The t_model used for training is:                '+ str(args.t_model))
print ('The dataset used for training is:                '+ str(args.data_name))
print ('The training mode is:                            '+ str(args.model))
print ('Which stage features are used for training:      '+ str(args.stage))
print ('The alpha parameter used is:                     '+ str(args.alpha))
print ('The beta parameter used is:                      '+ str(args.beta))
print ('The gamma parameter used is:                     '+ str(args.gamma))
print ('The delta parameter used is:                     '+ str(args.delta))
print ('The Temperature parameter used is:               '+ str(args.T))
print ('Whether to use data enhancement:                 '+ str(args.augmentation))
print ('The type of noise used is:                       '+ str(args.noise))

#flops, params = profile(tnet, input_size=(1, 3, 92,92)) #student: input_size=(1, 3, 44,44)
#print("The FLOS of this model is  %0.3f M" % float(flops/1024/1024))
#print("The params of this model is  %0.3f M" % float(params/1024/1024))

tnet.eval()
for param in tnet.parameters():
	param.requires_grad = False

# define loss functions
if args.cuda:
	Cls_crit = torch.nn.CrossEntropyLoss().cuda()   #Classification loss
	MSE_crit = nn.MSELoss().cuda() #MSE
	KD_T_crit = losses.KL_divergence(temperature = args.T).cuda() #KL
	decoder = losses.Decoder().cuda()
	tnet.cuda()
	snet.cuda()

else:
	Cls_crit = torch.nn.CrossEntropyLoss()
	MSE_crit = nn.MSELoss()
	KD_T_crit = losses.KL_divergence(temperature = args.T)

# initialize optimizer

if args.model == 'VID': 
	VID_NET1 = other.VID(96,96).cuda()
	VID_NET2 = other.VID(160,160).cuda()
	optimizer = torch.optim.SGD(itertools.chain(snet.parameters(),VID_NET1.parameters(),VID_NET2.parameters()), \
				    lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = True)
elif args.model == 'OFD':
	OFD_NET1 = other.OFD(96,96).cuda()
	OFD_NET2 = other.OFD(160,160).cuda()
	optimizer = torch.optim.SGD(itertools.chain(snet.parameters(),OFD_NET1.parameters(),OFD_NET2.parameters()), \
				    lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = True)
elif args.model == 'AFD':
	AFD_NET1 = other.AFD(96,1.0).cuda()
	AFD_NET2 = other.AFD(160,1.0).cuda()
	optimizer = torch.optim.SGD(itertools.chain(snet.parameters(),AFD_NET1.parameters(),AFD_NET2.parameters()), \
				    lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = True)
else:
	optimizer = torch.optim.SGD(snet.parameters(),
				    lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = True)

# define transforms
if args.augmentation:
	transform_train = transforms.Compose([
		transforms.RandomCrop(92),
		Cutout(),
		transforms.RandomHorizontalFlip(),
	])
else:
	transform_train = transforms.Compose([
		transforms.RandomCrop(92),
		transforms.RandomHorizontalFlip(),
	])

if args.augmentation == False and args.data_name == 'RAF':
	transforms_teacher_Normalize = transforms.Normalize((0.5884594, 0.45767313, 0.40865755), 
                            (0.25717735, 0.23602168, 0.23505741))
	transforms_student_Normalize =  transforms.Normalize((0.58846486, 0.45766878, 0.40865615), 
                            (0.2516557, 0.23020789, 0.22939532))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.59003043, 0.4573948, 0.40749523], std=[0.2465465, 0.22635746, 0.22564183])
            (transforms.ToTensor()(crop)) for crop in crops]))

elif args.augmentation == False and args.data_name == 'ExpW':
	transforms_teacher_Normalize = transforms.Normalize((0.6199751, 0.46946654, 0.4103778), 
                            (0.25622123, 0.22915973, 0.2232292))
	transforms_student_Normalize =  transforms.Normalize((0.6202007, 0.46964768, 0.41054007), 
                            (0.2498027, 0.22279221, 0.21712679))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.60876304, 0.45839235, 0.39910695], std=[0.2478118, 0.2180687, 0.21176754])
            (transforms.ToTensor()(crop)) for crop in crops]))

elif args.augmentation == False and args.data_name == 'CK_Plus':
	transforms_teacher_Normalize = transforms.Normalize((0.5950821, 0.59496826, 0.5949638), 
                            (0.2783952, 0.27837786, 0.27837303))
	transforms_student_Normalize =  transforms.Normalize((0.59541404, 0.59529984, 0.59529567), 
                            (0.2707762, 0.27075955, 0.27075458))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.52888066, 0.4993276, 0.48900297], std=[0.21970414, 0.21182147, 0.21353027])
            (transforms.ToTensor()(crop)) for crop in crops]))

else:
	raise Exception('Invalid dataset name...')

teacher_norm = transforms.Compose([
transforms.ToTensor(),
transforms_teacher_Normalize,
])

student_norm = transforms.Compose([
transforms.Resize(44),
transforms.ToTensor(),
transforms_student_Normalize,
])

transform_test = transforms.Compose([
transforms.TenCrop(44),
transforms_test_Normalize,
])

if args.data_name == 'RAF':
	trainset = RAF(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm, noise=args.noise)
	PrivateTestset = RAF(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None, noise=args.noise)
elif args.data_name == 'ExpW':
	trainset = ExpW(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
	PrivateTestset = ExpW(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None)
elif args.data_name == 'CK_Plus':
	trainset = CK_Plus(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
	PrivateTestset = CK_Plus(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None)
else:
	raise Exception('Invalid dataset name...')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=1)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.test_bs, shuffle=False, num_workers=1)

best_acc = 0
best_mAP = 0
best_F1 = 0
NUM_CLASSES = 7
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

def train(epoch):
	print('\nEpoch: %d' % epoch)
	snet.train()
	train_loss = 0
	train_cls_loss = 0

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

	for batch_idx, (img_teacher, img_student, target) in enumerate(trainloader):

		if args.cuda:
			img_teacher = img_teacher.cuda(non_blocking=True)
			img_student = img_student.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

		optimizer.zero_grad()
		
		if args.augmentation:
			img_teacher, teacher_target_a, teacher_target_b, teacher_lam = mixup_data(img_teacher, target, 0.6)
			img_teacher, teacher_target_a, teacher_target_b = map(Variable, (img_teacher, teacher_target_a, teacher_target_b))

			img_student, student_target_a, student_target_b, student_lam = mixup_data(img_student, target, 0.6)
			img_student, student_target_a, student_target_b = map(Variable, (img_student, student_target_a, student_target_b))
		else:
			img_teacher, img_student, target = Variable(img_teacher), Variable(img_student), Variable(target)

		rb1_s, rb2_s, rb3_s, mimic_s, out_s = snet(img_student)
		rb1_t, rb2_t, rb3_t, mimic_t, out_t = tnet(img_teacher)

		if args.augmentation:
			cls_loss = mixup_criterion(Cls_crit, out_s, student_target_a, student_target_b, student_lam)
		else:
			cls_loss = Cls_crit(out_s, target)

		kd_loss = KD_T_crit(out_t, out_s)

		if args.model == 'Fitnet': 
		#FITNETS: Hints for Thin Deep Nets
			if args.stage == 'Block1':
				Fitnet1_loss = other.Fitnet(rb1_t, rb1_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * Fitnet1_loss
			elif args.stage == 'Block2':
				Fitnet2_loss = other.Fitnet(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * Fitnet2_loss
			else:
				Fitnet1_loss = other.Fitnet(rb1_t, rb1_s).cuda()
				Fitnet2_loss = other.Fitnet(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * Fitnet1_loss + args.delta * Fitnet2_loss

		elif args.model == 'AT': # An activation-based attention transfer with the sum of absolute values raised to the power of 2.
		#Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
			if args.stage == 'Block1':
				AT1_loss = other.AT(rb1_t, rb1_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * AT1_loss
			elif args.stage == 'Block2':
				AT2_loss = other.AT(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * AT2_loss
			else:
				AT1_loss = other.AT(rb1_t, rb1_s).cuda()
				AT2_loss = other.AT(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * AT1_loss + args.delta * AT2_loss

		elif args.model == 'NST': # NST (poly)
		#Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
			if args.stage == 'Block1':
				NST1_loss = other.NST(rb1_t, rb1_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * NST1_loss
			elif args.stage == 'Block2':
				NST2_loss = other.NST(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * NST2_loss
			else:
				NST1_loss = other.NST(rb1_t, rb1_s).cuda()
				NST2_loss = other.NST(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * NST1_loss + args.delta * NST2_loss

		elif args.model == 'PKT': # PKT
		#Learning Deep Representations with Probabilistic Knowledge Transfer
			if args.stage == 'Block1':
				PKT1_loss = other.PKT(rb1_t, rb1_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * PKT1_loss
			elif args.stage == 'Block2':
				PKT2_loss = other.PKT(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * PKT2_loss
			else:
				PKT1_loss = other.PKT(rb1_t, rb1_s).cuda()
				PKT2_loss = other.PKT(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * PKT1_loss + args.delta * PKT2_loss

		elif args.model == 'AB': # AB
		#Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
			if args.stage == 'Block1':
				AB1_loss = other.AB(rb1_t, rb1_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * AB1_loss
			elif args.stage == 'Block2':
				AB2_loss = other.AB(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * AB2_loss
			else:
				AB1_loss = other.AB(rb1_t, rb1_s).cuda()
				AB2_loss = other.AB(rb2_t, rb2_s).cuda()
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * AB1_loss + args.delta * AB2_loss

		elif args.model == 'CCKD': # 
		#Correlation Congruence for Knowledge Distillation
			if args.stage == 'Block1':
				CCKD1_loss = other.CCKD().cuda()(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * CCKD1_loss
			elif args.stage == 'Block2':
				CCKD2_loss = other.CCKD().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * CCKD2_loss
			else:
				CCKD1_loss = other.CCKD().cuda()(rb1_t, rb1_s)
				CCKD2_loss = other.CCKD().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * CCKD1_loss + args.delta * CCKD2_loss

		elif args.model == 'RKD': # RKD-DA
		#Relational Knowledge Disitllation
			if args.stage == 'Block1':
				RKD1_loss = other.RKD().cuda()(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * RKD1_loss
			elif args.stage == 'Block2':
				RKD2_loss = other.RKD().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * RKD2_loss
			else:
				RKD1_loss = other.RKD().cuda()(rb1_t, rb1_s)
				RKD2_loss = other.RKD().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * RKD1_loss + args.delta * RKD2_loss

		elif args.model == 'SP': # SP
		#Similarity-Preserving Knowledge Distillation
			if args.stage == 'Block1':
				SP1_loss = other.SP().cuda()(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * SP1_loss
			elif args.stage == 'Block2':
				SP2_loss = other.SP().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * SP2_loss
			else:
				SP1_loss = other.SP().cuda()(rb1_t, rb1_s)
				SP2_loss = other.SP().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * SP1_loss + args.delta * SP2_loss

		elif args.model == 'VID': # VID-I
		#Variational Information Distillation for Knowledge Transfer
			if args.stage == 'Block1':
				VID1_loss = VID_NET1(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * VID1_loss
			elif args.stage == 'Block2':
				VID2_loss = VID_NET2(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * VID2_loss
			else:
				VID1_loss = VID_NET1(rb1_t, rb1_s)
				VID2_loss = VID_NET2(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * VID1_loss + args.delta * VID2_loss

		elif args.model == 'OFD': # OFD 
		#A Comprehensive Overhaul of Feature Distillation
			if args.stage == 'Block1':
				OFD1_loss = OFD_NET1(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * OFD1_loss
			elif args.stage == 'Block2':
				OFD2_loss = OFD_NET2(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * OFD2_loss
			else:
				OFD1_loss = OFD_NET1.cuda()(rb1_t, rb1_s)
				OFD2_loss = OFD_NET2(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * OFD1_loss + args.delta * OFD2_loss

		elif args.model == 'AFDS': # 
		#Pay Attention to Features, Transfer Learn Faster CNNs
			if args.stage == 'Block1':
				AFD1_loss = AFD_NET1(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * AFD1_loss
			elif args.stage == 'Block2':
				AFD2_loss = AFD_NET2(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * AFD2_loss
			else:
				AFD1_loss = AFD_NET1(rb1_t, rb1_s)
				AFD2_loss = AFD_NET2(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * AFD1_loss + args.delta * AFD2_loss

		elif args.model == 'FT': # 
		#Paraphrasing Complex Network: Network Compression via Factor Transfer
			if args.stage == 'Block1':
				FT1_loss = other.FT().cuda()(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * FT1_loss
			elif args.stage == 'Block2':
				FT2_loss = other.FT().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.delta * FT2_loss
			else:
				FT1_loss = other.FT().cuda()(rb1_t, rb1_s)
				FT2_loss = other.FT().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * FT1_loss + args.delta * FT2_loss

		elif args.model == 'CD': # CD+GKD+CE 
		#Channel Distillation: Channel-Wise Attention for Knowledge Distillation
			if args.stage == 'Block1':
				kd_loss_v2 = other.KDLossv2(args.T).cuda()(out_t, out_s, target)
				CD1_loss = other.CD().cuda()(rb1_t, rb1_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss_v2 + args.gamma * CD1_loss
			elif args.stage == 'Block2':
				kd_loss_v2 = other.KDLossv2(args.T).cuda()(out_t, out_s, target)
				CD2_loss = other.CD().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss_v2 + args.delta * CD2_loss
			else:
				kd_loss_v2 = other.KDLossv2(args.T).cuda()(out_t, out_s, target)
				CD1_loss = other.CD().cuda()(rb1_t, rb1_s)
				CD2_loss = other.CD().cuda()(rb2_t, rb2_s)
				loss = args.alpha * cls_loss + args.beta * kd_loss_v2 + args.gamma * CD1_loss + args.delta * CD2_loss

		elif args.model == 'FAKD': # DS+TS+SA 
		#FAKD: Feature-Affinity Based Knowledge Distillation for Efficient Image Super-Resolution
			if args.stage == 'Block1':
				FAKD_DT_loss = other.FAKD_DT().cuda()(out_t, out_s, target, NUM_CLASSES)
				FAKD_SA1_loss = other.FAKD_SA().cuda()(rb1_t, rb1_s)
				loss = args.alpha * FAKD_DT_loss + args.gamma * FAKD_SA1_loss      # No T
			elif args.stage == 'Block2':
				FAKD_DT_loss = other.FAKD_DT().cuda()(out_t, out_s, target, NUM_CLASSES)
				FAKD_SA2_loss = other.FAKD_SA().cuda()(rb2_t, rb2_s)
				loss = args.alpha * FAKD_DT_loss + args.gamma * FAKD_SA2_loss
			else:
				FAKD_DT_loss = other.FAKD_DT().cuda()(out_t, out_s, target, NUM_CLASSES)
				FAKD_SA1_loss = other.FAKD_SA().cuda()(rb1_t, rb1_s)
				FAKD_SA2_loss = other.FAKD_SA().cuda()(rb2_t, rb2_s)
				loss = args.alpha * FAKD_DT_loss + args.gamma * FAKD_SA1_loss + args.delta * FAKD_SA2_loss

		elif args.model == 'VKD': # 
		#Robust Re-Identification by Multiple Views Knowledge Distillation
			if args.stage == 'Block1':
				VKD_Similarity1_loss = other.VKD_SimilarityDistillationLoss().cuda()(rb1_t, rb1_s)
				VKD_OnlineTriplet1_loss = other.VKD_OnlineTripletLoss().cuda()(rb1_s, target)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * VKD_Similarity1_loss \
				                             + args.delta * VKD_OnlineTriplet1_loss
			elif args.stage == 'Block2':
				VKD_Similarity2_loss = other.VKD_SimilarityDistillationLoss().cuda()(rb2_t, rb2_s)
				VKD_OnlineTriplet2_loss = other.VKD_OnlineTripletLoss().cuda()(rb2_s, target)
				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * VKD_Similarity2_loss \
				                             + args.delta * VKD_OnlineTriplet2_loss
			else:
				VKD_Similarity1_loss = other.VKD_SimilarityDistillationLoss().cuda()(rb1_t, rb1_s)
				VKD_OnlineTriplet1_loss = other.VKD_OnlineTripletLoss().cuda()(rb1_s, target)

				VKD_Similarity2_loss = other.VKD_SimilarityDistillationLoss().cuda()(rb2_t, rb2_s)
				VKD_OnlineTriplet2_loss = other.VKD_OnlineTripletLoss().cuda()(rb2_s, target)

				loss = args.alpha * cls_loss + args.beta * kd_loss + args.gamma * VKD_Similarity1_loss \
				           + args.delta * VKD_OnlineTriplet1_loss  + args.gamma * VKD_Similarity2_loss \
				                             + args.delta * VKD_OnlineTriplet2_loss

		elif args.model == 'RAD': # RAD:  Resolution-Adapted Distillation
		# Efficient Low-Resolution Face Recognition via Bridge Distillation
			distance = mimic_t - mimic_s
			RAD_loss = torch.pow(distance, 2).sum(dim=(0,1), keepdim=False)  
			loss = RAD_loss + cls_loss
		else:
			raise Exception('Invalid model name...')
		
		loss.backward()
		utils.clip_gradient(optimizer, 0.1)
		optimizer.step()
		train_loss += loss.item()
		train_cls_loss += cls_loss.item()

		if args.augmentation:
			conf_mat_a += losses.confusion_matrix(out_s, student_target_a, NUM_CLASSES)
			acc_a = sum([conf_mat_a[i, i] for i in range(conf_mat_a.shape[0])])/conf_mat_a.sum()
			precision_a = np.array([conf_mat_a[i, i]/(conf_mat_a[i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
			recall_a = np.array([conf_mat_a[i, i]/(conf_mat_a[:, i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
			mAP_a = sum(precision_a)/len(precision_a)
			F1_score_a = (2 * precision_a*recall_a/(precision_a+recall_a + 1e-10)).mean()

			conf_mat_b += losses.confusion_matrix(out_s, student_target_b, NUM_CLASSES)
			acc_b = sum([conf_mat_b[i, i] for i in range(conf_mat_b.shape[0])])/conf_mat_b.sum()
			precision_b = np.array([conf_mat_b[i, i]/(conf_mat_b[i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
			recall_b = np.array([conf_mat_b[i, i]/(conf_mat_b[:, i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
			mAP_b = sum(precision_b)/len(precision_b)
			F1_score_b = (2 * precision_b*recall_b/(precision_b+recall_b + 1e-10)).mean()

			acc = student_lam * acc_a  +  (1 - student_lam) * acc_b
			mAP = student_lam * mAP_a  +  (1 - student_lam) * mAP_b
			F1_score = student_lam * F1_score_a  +  (1 - student_lam) * F1_score_b

		else:
			conf_mat += losses.confusion_matrix(out_s, target, NUM_CLASSES)
			acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
			precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10 )for i in range(conf_mat.shape[0])]
			mAP = sum(precision)/len(precision)

			recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10 ) for i in range(conf_mat.shape[0])]
			precision = np.array(precision)
			recall = np.array(recall)
			f1 = 2 * precision*recall/(precision+recall+ 1e-10)
			F1_score = f1.mean()

		#utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% | mAP: %.3f%% | F1: %.3f%%'
			#% (train_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100.* F1_score))
    
	return train_cls_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score
	

def test(epoch):
	
	snet.eval()
	PrivateTest_loss = 0
	t_prediction = 0
	conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
	
	for batch_idx, (img, target) in enumerate(PrivateTestloader):
		t = time.time()
		test_bs, ncrops, c, h, w = np.shape(img)
		img = img.view(-1, c, h, w)
		if args.cuda:
			img = img.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)
		
		img, target = Variable(img), Variable(target)

		with torch.no_grad():
			rb1_s, rb2_s, rb3_s, mimic_s, out_s = snet(img)

		outputs_avg = out_s.view(test_bs, ncrops, -1).mean(1)

		loss = Cls_crit(outputs_avg, target)
		t_prediction += (time.time() - t)
		PrivateTest_loss += loss.item()

		conf_mat += losses.confusion_matrix(outputs_avg, target, NUM_CLASSES)
		acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
		precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10 ) for i in range(conf_mat.shape[0])]
		mAP = sum(precision)/len(precision)

		recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10 ) for i in range(conf_mat.shape[0])]
		precision = np.array(precision)
		recall = np.array(recall)
		f1 = 2 * precision*recall/(precision+recall+ 1e-10)
		F1_score = f1.mean()

		#utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% | mAP: %.3f%% | F1: %.3f%%'
			#% (PrivateTest_loss / (batch_idx + 1), 100.*acc, 100.* mAP, 100.* F1_score))
  
	return PrivateTest_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score

for epoch in range(1, args.epochs+1):
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
		is_best = True
		print ('Saving models......')
		print("best_PrivateTest_acc: %0.3f" % best_acc)
		print("best_PrivateTest_mAP: %0.3f" % best_mAP)
		print("best_PrivateTest_F1: %0.3f" % best_F1)
		state = {
			'snet': snet.state_dict() if args.cuda else snet,
			'test_acc': test_acc,
			'test_mAP': test_mAP,
			'test_F1': test_F1,
			'test_epoch': epoch,
		} 
		torch.save(state, os.path.join(path,'Student_Test_model.t7'))

print ('!!!!!!!!!!!!!!       Done          !!!!!!!!!!!!!!!!!!!!!!\n\n\n')

print("best_PrivateTest_acc: %0.3f" % best_acc)
print("best_PrivateTest_mAP: %0.3f" % best_mAP)
print("best_PrivateTest_F1: %0.3f" % best_F1)



