from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import scipy
from scipy.stats import norm

def Fitnet(teacher, student):

    B, C, student_H, student_W = student.shape
    teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)

    Fitnet_loss = F.mse_loss(teacher, student)

    return Fitnet_loss

def AT(teacher, student, eps=1e-6):
    """Come From: https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/at.py"""

    B, C, student_H, student_W = student.shape
    teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)

    teacher = torch.pow(torch.abs(teacher), 2)
    teacher = torch.sum(teacher, dim=1, keepdim=True)
    teacher_norm = torch.norm(teacher, dim=(2,3), keepdim=True)
    teacher = torch.div(teacher, teacher_norm+eps)

    student = torch.pow(torch.abs(student), 2)
    student = torch.sum(student, dim=1, keepdim=True)
    student_norm = torch.norm(student, dim=(2,3), keepdim=True)
    student = torch.div(student, student_norm+eps)

    at_loss = F.mse_loss(teacher, student)

    return at_loss

def poly_kernel(teacher, student):

    teacher = teacher.unsqueeze(1)
    student = student.unsqueeze(2)
    out = (teacher * student).sum(-1).pow(2)

    return out

def NST(teacher, student):
    """Come From: https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/nst.py"""

    B, C, student_H, student_W = student.shape
    teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)

    teacher = teacher.view(teacher.size(0), teacher.size(1), -1)
    teacher = F.normalize(teacher, dim=2)

    student = student.view(student.size(0), student.size(1), -1)
    student = F.normalize(student, dim=2)

    NST_loss = poly_kernel(teacher, teacher).mean() + poly_kernel(student, student).mean() - 2 * poly_kernel(student, teacher).mean()

    return NST_loss


def PKT(teacher, student, eps=1e-6):
    """Come From: https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/pkt.py"""

    B, C, student_H, student_W = student.shape
    teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)

    teacher_norm = torch.sqrt(torch.sum(teacher ** 2, dim=1, keepdim=True))
    teacher = teacher / (teacher_norm + eps)    # Normalize each vector by its norm
    teacher[teacher != teacher] = 0

    student_norm = torch.sqrt(torch.sum(student ** 2, dim=1, keepdim=True))
    student = student / (student_norm + eps)    # Normalize each vector by its norm
    student[student != student] = 0

    teacher_cos_sim = torch.mul(teacher, teacher.transpose(3, 2))  # Calculate the cosine similarity
    student_cos_sim = torch.mul(student, student.transpose(3, 2))

    teacher_cos_sim = (teacher_cos_sim + 1.0) / 2.0 # Scale cosine similarity to [0,1]
    student_cos_sim = (student_cos_sim + 1.0) / 2.0

    teacher_cond_prob = teacher_cos_sim / torch.sum(teacher_cos_sim, dim=1, keepdim=True)  # Transform them into probabilities
    student_cond_prob = student_cos_sim / torch.sum(student_cos_sim, dim=1, keepdim=True)

    # Calculate the KL-divergence
    PKT_loss = torch.mean(teacher_cond_prob * torch.log((teacher_cond_prob + eps) / (student_cond_prob + eps)))

    return PKT_loss


def AB(teacher, student, margin=2.0):
    """Come From: https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/train_kd.py"""

    B, C, student_H, student_W = student.shape
    teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)

    loss = ((student + margin).pow(2) * ((student > -margin) & (teacher <= 0)).float() +
			    (student - margin).pow(2) * ((student <= margin) & (teacher > 0)).float())
    loss = loss.mean()

    return loss


class CCKD(nn.Module):
    """Come From: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/CC.py"""
    def __init__(self):
        super(CCKD, self).__init__()

    def forward(self, teacher, student):

        B, C, student_H, student_W = student.shape
        teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)

        delta = torch.abs(student - teacher)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss

class RKD(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    """Come From: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/RKD.py"""
    def __init__(self, w_d=1, w_a=2):
        super(RKD, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_t, f_s):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

class SP(nn.Module):
    """Similarity-Preserving Knowledge Distillation"""
    """Come From: https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/sp.py"""
    def __init__(self):
        super(SP, self).__init__()

    def forward(self, f_t, f_s):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)

        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)

        loss = F.mse_loss(G_t, G_s)

        return loss


class VID(nn.Module):
    """code from: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py"""
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VID, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, target, input):

        B, C, student_H, student_W = input.shape
        target = torch.nn.functional.interpolate(target, size=[student_H, student_W], mode='nearest', align_corners=None)

        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss

class OFD(nn.Module):
	"""code from: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/ofd.py"""
	def __init__(self, in_channels, out_channels):
		super(OFD, self).__init__()
		self.connector = nn.Sequential(*[
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_channels)
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, fm_t, fm_s):

		B, C, student_H, student_W = fm_s.shape
		fm_t = torch.nn.functional.interpolate(fm_t, size=[student_H, student_W], mode='nearest', align_corners=None)

		margin = self.get_margin(fm_t)
		fm_t = torch.max(fm_t, margin)
		fm_s = self.connector(fm_s)

		mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
		loss = torch.mean((fm_s - fm_t)**2 * mask)

		return loss

	def get_margin(self, fm, eps=1e-6):
		mask = (fm < 0.0).float()
		masked_fm = fm * mask

		margin = masked_fm.sum(dim=(0,2,3), keepdim=True) / (mask.sum(dim=(0,2,3), keepdim=True)+eps)

		return margin


class AFD(nn.Module):
	"""code from: https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/afd.py"""
	def __init__(self, in_channels, att_f):
		super(AFD, self).__init__()
		mid_channels = int(in_channels * att_f)
		self.attention = nn.Sequential(*[
				nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		
	def forward(self, fm_t, fm_s, eps=1e-6):

		B, C, student_H, student_W = fm_s.shape
		fm_t = torch.nn.functional.interpolate(fm_t, size=[student_H, student_W], mode='nearest', align_corners=None)

		fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
		rho = self.attention(fm_t_pooled)
		# rho = F.softmax(rho.squeeze(), dim=-1)
		rho = torch.sigmoid(rho.squeeze())
		rho = rho / torch.sum(rho, dim=1, keepdim=True)

		fm_s_norm = torch.norm(fm_s, dim=(2,3), keepdim=True)
		fm_s      = torch.div(fm_s, fm_s_norm+eps)
		fm_t_norm = torch.norm(fm_t, dim=(2,3), keepdim=True)
		fm_t      = torch.div(fm_t, fm_t_norm+eps)

		loss = rho * torch.pow(fm_s-fm_t, 2).mean(dim=(2,3))
		loss = loss.sum(1).mean(0)

		return loss

class FT(nn.Module):
	"""code from: https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/ft.py"""
	def __init__(self):
		super(FT, self).__init__()

	def forward(self, factor_t, factor_s):

		B, C, student_H, student_W = factor_s.shape
		factor_t = torch.nn.functional.interpolate(factor_t, size=[student_H, student_W], mode='nearest', align_corners=None)

		loss = F.l1_loss(self.normalize(factor_s), self.normalize(factor_t))

		return loss

	def normalize(self, factor):
		norm_factor = F.normalize(factor.view(factor.size(0),-1))

		return norm_factor

class CD(nn.Module):
    """Channel Distillation Loss,come from:    https://github.com/zhouzaida/channel-distillation"""

    def __init__(self):
        super().__init__()

    def forward(self, tea_features: list, stu_features: list):
        loss = 0.
        for s, t in zip(stu_features, tea_features):
            s = s.mean(dim=(1, 2), keepdim=False)
            t = t.mean(dim=(1, 2), keepdim=False)
            loss += torch.mean(torch.pow(s - t, 2))
        return loss

class KDLossv2(nn.Module):
    """Guided Knowledge Distillation Loss"""

    def __init__(self, T):
        super().__init__()
        self.t = T

    def forward(self, tea_pred, stu_pred, label):
        s = F.log_softmax(stu_pred / self.t, dim=1)
        t = F.softmax(tea_pred / self.t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        mask = torch.eq(label, t_argmax).float()
        count = (mask[mask == 1]).size(0)
        mask = mask.unsqueeze(-1)
        correct_s = s.mul(mask)
        correct_t = t.mul(mask)
        correct_t[correct_t == 0.0] = 1.0

        loss = F.kl_div(correct_s, correct_t, reduction='sum') * (self.t**2) / count
        return loss

class FAKD_DT(nn.Module):
    """Modify from:     https://github.com/Vincent-Hoo/Knowledge-Distillation-for-Super-resolution"""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss().cuda()
        self.CEloss = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, tea_pred, stu_pred, label, NUM_CLASSES):

        TS_loss = self.loss(stu_pred, tea_pred)
#         label = torch.nn.functional.one_hot(label, NUM_CLASSES).float()

        DS_loss = self.CEloss(stu_pred, label)

        loss = DS_loss + TS_loss

        return loss


class FAKD_SA(nn.Module):
    """Modify from:     https://github.com/Vincent-Hoo/Knowledge-Distillation-for-Super-resolution"""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, teacher, student):

        B, C, student_H, student_W = student.shape
        teacher = torch.nn.functional.interpolate(teacher, size=[student_H, student_W], mode='nearest', align_corners=None)

        teacher = self.spatial_similarity(teacher)
        student = self.spatial_similarity(student)
        loss = self.loss(teacher, student)

        return loss

    def spatial_similarity(self, fm): 
        
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm,2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
        s = norm_fm.transpose(1,2).bmm(norm_fm)
        s = s.unsqueeze(1)
        return s


class MatrixPairwiseDistances(nn.Module):
    """Come from:     https://github.com/aimagelab/VKD"""

    def __init__(self):
        super(MatrixPairwiseDistances, self).__init__()

    def __call__(self, x: torch.Tensor, y: torch.Tensor = None):
        if y is not None:  # exact form of squared distances
            differences = x.unsqueeze(1) - y.unsqueeze(0)
        else:
            differences = x.unsqueeze(1) - x.unsqueeze(0)
        distances = torch.sum(differences * differences, -1)
        return distances

class VKD_SimilarityDistillationLoss(nn.Module):
    """Come from:     https://github.com/aimagelab/VKD"""

    def __init__(self):
        super(VKD_SimilarityDistillationLoss, self).__init__()
        self.distances = MatrixPairwiseDistances()

    def forward(self, teacher_embs: torch.Tensor, student_embs: torch.Tensor):

        B, C, student_H, student_W = student_embs.shape
        teacher_embs = torch.nn.functional.interpolate(teacher_embs, size=[student_H, student_W], mode='nearest', align_corners=None)

        teacher_distances = self.distances(teacher_embs)
        student_distances = self.distances(student_embs)

        return 0.5 * nn.MSELoss(reduction='mean')(student_distances, teacher_distances)

class VKD_OnlineTripletLoss(nn.Module):
    """Come from:     https://github.com/aimagelab/VKD"""

    def __init__(self, margin='soft', batch_hard=True, reduction='mean'):
        super(VKD_OnlineTripletLoss, self).__init__()
        self.batch_hard = batch_hard
        self.reduction = reduction
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin
        else:
            raise NotImplementedError(
                'The margin {} is not recognized in TripletLoss()'.format(margin))

    def forward(self, feat, id=None, pos_mask=None, neg_mask=None, mode='id', dis_func='eu',
                n_dis=0):

        if dis_func == 'cdist':
            feat = feat / feat.norm(p=2, dim=1, keepdim=True)
            dist = self.cdist(feat, feat)
        elif dis_func == 'eu':
            dist = self.cdist(feat, feat)
            dist = torch.mean(dist, (2, 3))

        if mode == 'id':
            if id is None:
                raise RuntimeError('foward is in id mode, please input id!')
            else:
                identity_mask = torch.eye(feat.size(0)).byte()
                identity_mask = identity_mask.cuda() if id.is_cuda else identity_mask
                same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
                negative_mask = same_id_mask ^ torch.full((same_id_mask.shape[0],same_id_mask.shape[1]), 1, dtype=torch.bool).cuda()
                positive_mask = same_id_mask ^ identity_mask.bool()
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                raise RuntimeError('foward is in mask mode, please input pos_mask & neg_mask!')
            else:
                positive_mask = pos_mask
                same_id_mask = neg_mask ^ 1
                negative_mask = neg_mask
        else:
            raise ValueError('unrecognized mode')

        if self.batch_hard:
            if n_dis != 0:
                img_dist = dist[:-n_dis, :-n_dis]
                max_positive = (img_dist * positive_mask[:-n_dis, :-n_dis].float()).max(1)[0]
                min_negative = (img_dist + 1e5 * same_id_mask[:-n_dis, :-n_dis].float()).min(1)[0]
                dis_min_negative = dist[:-n_dis, -n_dis:].min(1)[0]
                z_origin = max_positive - min_negative
                # z_dis = max_positive - dis_min_negative
            else:
                max_positive = (dist * positive_mask.float()).max(1)[0]
                min_negative = (dist + 1e5 * same_id_mask.float()).min(1)[0]
                z = max_positive - min_negative
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1, 1)
            positive = torch.gather(dist, dim=1, index=pos)
            pos = negative_mask.topk(k=1, dim=1)[1].view(-1, 1)
            negative = torch.gather(dist, dim=1, index=pos)
            z = positive - negative

        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            if n_dis != 0:
                b_loss = torch.log(1 + torch.exp(
                    z_origin)) + -0.5 * dis_min_negative  # + torch.log(1+torch.exp(z_dis))
            else:
                b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError("How do you even get here!")

        if self.reduction == 'mean':
            return b_loss.mean()

        return b_loss.sum()

    def cdist(self, a, b):
        '''
        Returns euclidean distance between a and b
        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        '''
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff ** 2).sum(2) + 1e-12).sqrt()





class DirectCapsNet(nn.Module):
    def __init__(self, in_dim=7,out_dim=7):
        super(DirectCapsNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.layer2 = nn.Linear(out_dim, out_dim)
        self.layer3 = nn.Linear(out_dim, out_dim)
        self.MSE_crit = nn.MSELoss().cuda() 
 
    def forward(self, out_t, out_s):
        
        out_s = self.layer1(out_s)
        out_s = self.layer2(out_s)
        out_s = self.layer3(out_s)
        
        reconstruction_loss = self.MSE_crit(out_t,out_s)
        
        return reconstruction_loss


def get_margin_from_BN(mean, std):
    margin = []
    
    s = std
    m = mean
    
    s = abs(s.item())
    m = m.item()
    if norm.cdf(-m / s) > 0.001:
        margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
    else:
        margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()
