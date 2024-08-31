# 005_Best1_Full_TCN_3channel_fusedlabel_fit_3ch_seperate_all_weighted_loss_230912


base_dir = "/mnt/ssd/ylyoo/intermediate_filtered_split"
# base_dir = "/mnt/hdd/unist/intermediate_filtered_split"

# patch 적용
patch_size = (64,64)  # z축 방향 patch는 일단안함.. 복잡해져서..
num_samples = 16
pos_neg_ratio = 0.5   #젤중요 > tumor 비율

batch_size = 1
# train_loader의 batch size의 경우 dataloader의 batch size 에
# num_samples가 곱해진다!mn
# 즉 실 batch_size = batch_size * num_samples
sw_batch_size = 10000  # sliding window forward시 이때의 batch size. 무제한이 좋음.
overlap = 0.5 # sliding window overlap 0.5가 좋음. 이건 무조건 고정. 

##### 빠른 training 위해 일단 전체 데이터의 10~20% 선택
train_ratio = 0.07
valid_ratio = 0.1
test_ratio = 0.1
###########
max_epochs = 100
patience = 5  # 5번안에 개선안되면 중단하는 early stopping


#https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
# MS-SSSIM은 32이상일때만 적용가능해서 32로 바꿈. 

from functools import total_ordering
from operator import index
import os
from datetime import datetime
import glob
import random

import matplotlib.pyplot as plt
import nibabel as nib


import torch
import torch.nn as nn

from tqdm import tqdm

from utils_dce2ktrans import nii_to_numpy, visualize_ktrans_patch,visualize_ktrans_final, visualize_ktrans_final_save, Weighted_L1_patch
from utils_dce2ktrans import FixedRandCropByPosNegLabel_custom_coordinates, RandCropByPosNegLabel_custom_coordinates, apply_patch_coordinates_4d, apply_patch_coordinates
from utils_dce2ktrans import TemporalConvNet_custom, Wrapper_Net_fixed, Wrapper_integrate_Net

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd, 
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandWeightedCropd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld
    
)
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.networks.nets import UNet, DynUnet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils_dce2ktrans import format_4f, nii_to_numpy,path_cutter, mriview, Weighted_L1_patch, visualize_ktrans_patch, visualize_ktrans_patch_30, visualize_ktrans_compare, nonvisualize_ktrans_compare, nonvisualize_ktrans_compare_save
from utils_dce2ktrans import TemporalBlock, tile_3d_to_60_torch

device = torch.device('cpu') # cpu 로 돌릴 때
# device = torch.device('cuda') # gpu 로 돌릴 떄
print('torch.cuda.is_available:', torch.cuda.is_available())
print('torch.cuda.device_count():',  torch.cuda.device_count())
print('torch.cuda.current_device():',  torch.cuda.current_device())
print('torch.cuda.get_device_name(0):',torch.cuda.get_device_name(0))

# CPU 과소비 하지 않게 설정 (실험 여러개 돌릴 수 있게 ) by 이한얼 선생님
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=2

import numpy as np  #이게 os.environ 다음에 와야 한다고 함

torch.set_num_threads(8)

# Pytorch lightning code 구조
# model = MyLightningModule()
# trainer = Trainer()
# trainer.fit(model, train_dataloader, val_dataloader)
# model = MyLightningModule.load_from_checkpoint(PATH)
# trainer.test(model, dataloaders=test_dataloader)

set_determinism(seed=0)

tensorboard_dir_name = "tb_logs_dsc"
experiment_name = "patch3d_lightning_unet"
test_forward = False #True시 이를 forward하는 기능
forward_name = "exp1"
forward_filename = "forwarded_"+forward_name+'.nii.gz'

gpu_list = [1]
input_filename = 'g_dce_01.nii.gz' 
label_filename = 'g_kvpve_fixed.nii.gz'
mask_filename = 'g_dce_mask.nii.gz'
seg_filename = 'g_seg.nii.gz'
# f_dsc    256 256 18 60     > 341개
# f_ktrans 256 256 18        > 341개
# seg_DSCspace.nii.gz 256 256 18  > 237개


# base_dir = "/mnt/ssd/ylyoo/intermediate_filtered_split"
# # base_dir = "/mnt/hdd/unist/intermediate_filtered_split"

# # patch 적용
# patch_size = (64,64)  # z축 방향 patch는 일단안함.. 복잡해져서..
# num_samples = 16
# pos_neg_ratio = 0.5   #젤중요 > tumor 비율

# batch_size = 1
# # train_loader의 batch size의 경우 dataloader의 batch size 에
# # num_samples가 곱해진다!mn
# # 즉 실 batch_size = batch_size * num_samples
# sw_batch_size = 10000  # sliding window forward시 이때의 batch size. 무제한이 좋음.
# overlap = 0.5 # sliding window overlap 0.5가 좋음. 이건 무조건 고정. 

# ##### 빠른 training 위해 일단 전체 데이터의 10~20% 선택
# train_ratio = 0.1
# valid_ratio = 0.07
# test_ratio = 0.07
# ###########
# max_epochs = 100
# patience = 5  # 5번안에 개선안되면 중단하는 early stopping

optimizer_class = torch.optim.Adam
learning_rate = 3e-4

class Tumorseg_Loss(nn.Module):    
    def __init__(self,alpha):
        super(Tumorseg_Loss, self).__init__()
        self.alpha = alpha
    def forward(self, predictions, target, seg):  
        L1_loss = torch.nn.L1Loss()    
        loss_L1 = L1_loss(predictions,target)     
        loss_tumor = L1_loss(predictions*seg,target*seg)         
        loss_tumor = loss_tumor * self.alpha
        loss = loss_L1 + loss_tumor
        print(loss_L1, loss_tumor)
        # alpha = 1 기준으로 0.08, 0.008 정도
        return loss

#criterion = Tumorseg_Loss(alpha = 100)

class Ktrans_Triple_Loss(nn.Module):    
    def __init__(self,epsilon, alpha):
        super(Ktrans_Triple_Loss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
    def forward(self, predictions, target, seg):  

        L1_loss = torch.nn.L1Loss()    
        loss_high = L1_loss(predictions*(target>self.epsilon),target*(target>self.epsilon))     
        loss_L1 = L1_loss(predictions,target)  
        loss_high = loss_high * self.alpha
        loss = loss_L1 + loss_high
        print(loss_L1, loss_high)
        # alpha = 1 기준으로 0.08, 0.008 정도
        return loss 

#criterion = Ktrans_Triple_Loss(epsilon =0.3,alpha = 40)


class Ktrans_weighted_L1_Loss(nn.Module):    
# L1 loss +  (y > epsilon ) * (beta*y+alpha) 꼴의 loss 

    def __init__(self,epsilon, alpha, beta):
        super(Ktrans_weighted_L1_Loss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
    def forward(self, predictions, target, seg):  
        L1_loss = torch.nn.L1Loss()    
        loss_beta = L1_loss(predictions*(target>self.epsilon)*(target*self.beta),target*(target>self.epsilon)*(target*self.beta))     
        loss_alpha = L1_loss(predictions*(target>self.epsilon)*(self.alpha),target*(target>self.epsilon)*(self.alpha))    
        loss_L1 = L1_loss(predictions,target)  
        loss = loss_L1 + loss_alpha+loss_beta
        print(loss_L1, loss_alpha,loss_beta)
        return loss
        # alpha = 1 기준으로 0.08, 0.008 정도
    def print_parameters(self):
        print("Loss - Epsilon, alpha, beta", self.epsilon, self.alpha, self.beta)


class Ktrans_weighted_L1_high_Loss(nn.Module):    
# abs(y-ktrans) * (ktrans > epsilon ) * (beta*ktrans+1) 꼴의 loss 
# ktrans < epsilon인 작은 것들은 아예 loss를 주지 않는다. 
    def __init__(self,epsilon, beta):
        super(Ktrans_weighted_L1_high_Loss, self).__init__()
        self.epsilon = epsilon
        self.beta = beta
        self.print_parameters()
    def forward(self, predictions, target, seg):  
        L1_loss = torch.nn.L1Loss()    
        loss_beta = L1_loss(predictions*(target>self.epsilon)*(target*self.beta),target*(target>self.epsilon)*(target*self.beta))     
        loss_alpha = L1_loss(predictions*(target>self.epsilon),target*(target>self.epsilon))    
        loss =  loss_alpha+loss_beta
        print(loss_alpha,loss_beta)
        return loss
        # alpha = 1 기준으로 0.08, 0.008 정도
    def print_parameters(self):
        print("Loss - Epsilon, alpha, beta", self.epsilon, self.beta)



class Ktrans_weighted_L1_high_exponentialLoss(nn.Module):    
# abs(y-ktrans)^m * (ktrans > epsilon ) * (ktrans^n) 꼴의 loss 
# ktrans < epsilon인 작은 것들은 아예 loss를 주지 않는다. 
    def __init__(self,epsilon,m,n):
        super(Ktrans_weighted_L1_high_exponentialLoss, self).__init__()
        self.epsilon = epsilon
        self.m = m
        self.n = n
        self.print_parameters()
    def forward(self, predictions, target, seg):  
        loss_mn = torch.mean(torch.pow(torch.abs(predictions-target),self.m)*torch.pow(target,self.n)*(target>self.epsilon))
        return loss_mn
    def print_parameters(self):
        print("Loss - Epsilon, m,n ", self.epsilon, self.m, self.n)


class Ktrans_weighted_L1_and_exponentialLoss_Ktrans(nn.Module):     # Ktrans 용은 35~40이 적당한 듯
# abs(y-ktrans)^m * (ktrans > epsilon ) * (ktrans^n) 꼴의 loss 
# ktrans < epsilon인 작은 것들은 아예 loss를 주지 않는다. 
    def __init__(self,epsilon,alpha, m,n):
        super(Ktrans_weighted_L1_and_exponentialLoss_Ktrans, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.m = m
        self.n = n
        self.print_parameters()
    def forward(self, predictions, target, seg):  
        loss_mn = torch.mean(torch.pow(torch.abs(predictions-target),self.m)*torch.pow(target,self.n)*(target>self.epsilon))
        loss_l1 = torch.mean(torch.abs(predictions-target))
        loss = loss_l1 + self.alpha *loss_mn
        #print(loss_l1, self.alpha*loss_mn)
        return loss
    def print_parameters(self):
        print("Loss - Epsilon, alpha, m,n ", self.epsilon, self.alpha, self.m, self.n)


class Ktrans_weighted_L1_and_exponentialLoss_Vp(nn.Module):    
# abs(y-ktrans)^m * (ktrans > epsilon ) * (ktrans^n) 꼴의 loss 
# ktrans < epsilon인 작은 것들은 아예 loss를 주지 않는다. 
    def __init__(self,epsilon,alpha, m,n):
        super(Ktrans_weighted_L1_and_exponentialLoss_Vp, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.m = m
        self.n = n
        self.print_parameters()
    def forward(self, predictions, target, seg):  
        loss_mn = torch.mean(torch.pow(torch.abs(predictions-target),self.m)*torch.pow(target,self.n)*(target>self.epsilon))
        loss_l1 = torch.mean(torch.abs(predictions-target))
        loss = loss_l1 + self.alpha *loss_mn
        #print(loss_l1, self.alpha*loss_mn)
        return loss
    def print_parameters(self):
        print("Loss - Epsilon, alpha, m,n ", self.epsilon, self.alpha, self.m, self.n)

# 230911
class Ktrans_weighted_L1_and_exponentialLoss_Ve(nn.Module):     # Ve용으로 L1 loss만 적용해보기! m = 1, n = 1
# abs(y-ktrans)^m * (ktrans > epsilon ) * (ktrans^n) 꼴의 loss 
# ktrans < epsilon인 작은 것들은 아예 loss를 주지 않는다. 
    def __init__(self,epsilon,alpha, m,n):
        super(Ktrans_weighted_L1_and_exponentialLoss_Ve, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.m = m
        self.n = n
        self.print_parameters()
    def forward(self, predictions, target, seg):   # 
        loss_mn = torch.mean(torch.pow(torch.abs(predictions-target),self.m)*torch.pow(target,self.n)*(target>self.epsilon))
        loss_l1 = torch.mean(torch.abs(predictions-target))
        loss = loss_l1 + self.alpha *loss_mn
        print(loss_l1, self.alpha*loss_mn)
        return loss
    def print_parameters(self):
        print("Loss - Epsilon, alpha, m,n ", self.epsilon, self.alpha, self.m, self.n)



class DiceLoss(nn.Module):
    # 엄밀히는 loss가 아님. loss로 하려면 
    #  (x/epsilon)으로 나눠준다음에 real prediction 기준으로 해야
    def __init__(self,epsilon):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, inputs_raw, targets_raw, smooth=1):       
        inputs = (inputs_raw>self.epsilon)
        targets = (targets_raw>self.epsilon)
        """ 
        inputs = inputs.view(-1)
        targets = targets.view(-1)       """
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return dice 

##### Loss weight 조절 ####
criterion_Ktrans = Ktrans_weighted_L1_and_exponentialLoss_Ktrans(epsilon =0,alpha = 35, m =1,n=1)
criterion_Vp = Ktrans_weighted_L1_and_exponentialLoss_Vp(epsilon =0,alpha = 35, m =1,n=1)
criterion_Ve = Ktrans_weighted_L1_and_exponentialLoss_Ve(epsilon =0,alpha = 5, m =1,n=1)

# self.criterion_Ktrans(y_hat, y, y_seg) + self.criterion_Vp(y_hat, y, y_seg) + self.criterion_Ve(y_hat, y, y_seg)

dice_01 = DiceLoss(epsilon=0.1)
dice_02 = DiceLoss(epsilon=0.2)

#criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()
#epsilon = 0.007
#criterion = Weighted_L1_patch(epsilon = epsilon)

use_auto_lr = False  # 잘안되서 끔.. 값만 참조
precision = 32  # 16이 더 빠른거같은데.. torch 1.16인가 설치해야 cpu에서 할수있다고 한다.
output_scale = 1

# valid epoch 마다 비교할거
#visualize_list_valid = [2,6]
#visualize_slice_list_valid = [[8],[5]]  
visualize_list_valid = [2,6]
visualize_slice_list_valid = [[15,18,29],[13,15,29]]  



print("patch size :", patch_size)
print("number of patch samples per slices",num_samples)
print("batch size for dataloader", batch_size)
print("effective batch size = num_samples*batch_size", batch_size*num_samples)
print("PosNegRatio ", pos_neg_ratio)
print("max_epochs", max_epochs)
print("patience",patience)
print("optimizer_class",optimizer_class)
print("learning rate",learning_rate)
print("loss function_Ktrans", criterion_Ktrans)
print("loss function_Vp", criterion_Vp)
print("loss function_Ve", criterion_Ve)

print("train_ratio :", train_ratio)
print("valid_ratio :", valid_ratio)
print("test_ratio :", test_ratio)


#criterion = SqrtWeighted_L1(epsilon = 0.01)
#criterion = SquareWeighted_L1_fixed(epsilon = 0.01)
#criterion = torch.nn.L1Loss()
#criterion = SSIM_Loss()
#criterion = wL1_SSIM(lamda=100)
    # 괄호 안붙이면 아래와 같이 에러난다.  
    # Boolean value of Tensor with more than one value is ambiguous
    # loss = nn.MSELoss(pred,target)하면 안되고
    # loss_func = nn.MSELoss() 후 loss = loss_func(pred,target)으로 해야한다함.


os.makedirs(tensorboard_dir_name,exist_ok=True) # 없을 때만 만든다. 

def make_nii_path_dataset(data_dir,input_filename,label_filename,seg_filename,mask_filename,ratio_select = 1):
    
    # seg있는데 image 없는것도 있으니 seg기준으로 만들고 replace로 image, label path만듬
    path_segs = sorted(glob.glob(data_dir+"/*/"+seg_filename))
    path_images = [x.replace(seg_filename,input_filename) for x in path_segs]
    path_labels = [x.replace(seg_filename,label_filename) for x in path_segs]
    path_masks = [x.replace(seg_filename,mask_filename) for x in path_segs]

    path_dataset = [
    {"image": image_name, "label": label_name, "seg" : seg_name, "mask" : mask_name, "path": image_name[:-len(input_filename)-1]}
    for image_name, label_name,seg_name,mask_name in zip(path_images, path_labels, path_segs,path_masks)
    ]
    if ratio_select==1:
        print("created path dataset of size: ",len(path_dataset)," from ",data_dir)
        print("containing input: ", input_filename, " label:", label_filename, "seg:",seg_filename, "mask:" , mask_filename)
    if ratio_select!=1:
        num_select = int(ratio_select*len(path_dataset))       
        print("Using only a subset for fast training:")
        print("Original path dataset of size: ",len(path_dataset)," from ",data_dir)
        print("containing input: ", input_filename, " label:", label_filename,"seg:",seg_filename, "mask:" , mask_filename)
        print("Selected dataset of size", num_select)
        path_dataset = path_dataset[:num_select]    
    return path_dataset          


train_files = make_nii_path_dataset(base_dir+"/train",input_filename,label_filename,seg_filename,mask_filename, train_ratio)
valid_files = make_nii_path_dataset(base_dir+"/valid",input_filename,label_filename,seg_filename,mask_filename, valid_ratio)
test_files = make_nii_path_dataset(base_dir+"/test",input_filename,label_filename,seg_filename,mask_filename, test_ratio)

""" 
Todo할거
LoadImaged
EnsureCHannelFirstD
CropForegroundd  (using mask)
RandCropByPosNegLabeld  (using seg)
여기까지 필수
추가로 해볼수있는거
RandFLipq
ScaleIntensityRanged
RandAffined >이건느릴듯 """

# 결론 - nonramdomizable transform까지 다 cache 를 entire dataset을 하고
# non random만 dataloader사용해 접근하자.

class DiffusionCacheDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files
        self.cache_files = []
        for i in range(len(data_files)):
            vol_image = nii_to_numpy(self.data_files[i]['image'])   
            vol_label = nii_to_numpy(self.data_files[i]['label'])
            vol_seg = nii_to_numpy(self.data_files[i]['seg'])
            vol_mask = nii_to_numpy(self.data_files[i]['mask'])
            vol_path = self.data_files[i]['path']
            sample = {'image': vol_image,'label': vol_label, 'seg': vol_seg, 'mask': vol_mask, 'path': vol_path}
            self.cache_files.append(sample)
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.cache_files[idx]
        return sample



class train_transform(object): 
     # return patches of size N 60 16 16 from 'image', 'label', 'seg'
    def __init__(self, num_samples,patch_size,pos_neg_ratio):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.pos_neg_ratio = pos_neg_ratio
    def __call__(self, sample):
        image = torch.from_numpy(sample['image']).float() # 256 256 40 60
        label = torch.from_numpy(sample['label']) # 256 256 40 3
        seg = torch.from_numpy(sample['seg']) # 256 256 40
        mask = torch.from_numpy(sample['mask']) # 256 256 40
        patch_size = torch.tensor([self.patch_size[0],self.patch_size[1],1]) # torch.tensor([16,16,1])
        num_patch_samples = self.num_samples
        pos_neg_ratio = self.pos_neg_ratio
        #print(sample['path'])
        #rint(image.shape,label.shape,seg.shape,mask.shape,patch_size.shape,num_patch_samples)
        coords = RandCropByPosNegLabel_custom_coordinates(seg,patch_size,num_patch_samples,pos_neg_ratio)
        #print(coords.shape)
        patches_image = apply_patch_coordinates_4d(image,coords) # 1000 60 16 16
        patches_label = apply_patch_coordinates_4d(label,coords)   # 1000 60 16 16
        patches_seg = apply_patch_coordinates(seg,coords)      # 1000 60 16 16
        # 1000 60 16 16 꼴로 정리
        return patches_image, patches_label, patches_seg



class valid_transform(object): 
     # return patches of size N 60 16 16 from 'image', 'label', 'seg'
    def __init__(self, num_samples,patch_size,pos_neg_ratio):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.pos_neg_ratio = pos_neg_ratio
    def __call__(self, sample):
        image = torch.from_numpy(sample['image']).float() # 256 256 18 60
        label = torch.from_numpy(sample['label']) # 256 256 18
        seg = torch.from_numpy(sample['seg']) # 256 256 18
        mask = torch.from_numpy(sample['mask']) # 256 256 18
        path = sample['path']
        #print(path)
        x = hash(path)%(2**32)
        patch_size = torch.tensor([self.patch_size[0],self.patch_size[1],1]) # torch.tensor([16,16,1])
        num_patch_samples = self.num_samples
        pos_neg_ratio = self.pos_neg_ratio
        coords = FixedRandCropByPosNegLabel_custom_coordinates(x, seg,patch_size,num_patch_samples,pos_neg_ratio)
        patches_image = apply_patch_coordinates_4d(image,coords) # 1000 60 16 16
        patches_label = apply_patch_coordinates_4d(label,coords)   # 1000 60 16 16
        patches_seg = apply_patch_coordinates(seg,coords)      # 1000 60 16 16
        # 1000 60 16 16 꼴로 정리
        return patches_image, patches_label, patches_seg


class test_transform(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = torch.from_numpy(sample['image']).float() # 256 256 18 60
        label = torch.from_numpy(sample['label']) # 256 256 18
        seg = torch.from_numpy(sample['seg']) # 256 256 18       
        return image, label, seg


train_cache = DiffusionCacheDataset(train_files)
print("finished caching train files")
train_dataset = Dataset(data=train_cache, transform=train_transform(num_samples = num_samples, patch_size = patch_size, pos_neg_ratio=pos_neg_ratio))
train_loader = DataLoader(train_dataset, batch_size=batch_size)
generator = iter(train_loader)
print(f'len: {len(generator)}')
patches_image_example, patches_label_example, patches_seg_example = next(generator)

# fig, axs = plt.subplots(4,4)
# print("example train patches : input")
# for i in range(4):
#     for j in range(4):
#         axs[i,j].axis('off')
#         axs[i,j].imshow(patches_image_example[0,4*i+j,30,:,:],cmap='jet',vmin=0,vmax=1)
# plt.show()
# fig, axs = plt.subplots(4,4)
# print("example train patches : label=channel 0")
# for i in range(4):
#     for j in range(4):
#         axs[i,j].axis('off')
#         axs[i,j].imshow(patches_label_example[0,4*i+j,0,:,:],cmap='jet',vmin=0,vmax=0.1)
# plt.show()
# fig, axs = plt.subplots(4,4)
# print("example train patches : label=channel 1")
# for i in range(4):
#     for j in range(4):
#         axs[i,j].axis('off')
#         axs[i,j].imshow(patches_label_example[0,4*i+j,1,:,:],cmap='jet',vmin=0,vmax=0.1)
# plt.show()
# fig, axs = plt.subplots(4,4)
# print("example train patches : label=channel 2")
# for i in range(4):
#     for j in range(4):
#         axs[i,j].axis('off')
#         axs[i,j].imshow(patches_label_example[0,4*i+j,2,:,:],cmap='jet',vmin=0,vmax=0.1)
# plt.show()


# print("example train patches : seg")
# fig, axs = plt.subplots(4,4)
# for i in range(4):
#     for j in range(4):
#         axs[i,j].axis('off')
#         axs[i,j].imshow(patches_seg_example[0,4*i+j,0,:,:])
# plt.show()



test_cache = DiffusionCacheDataset(test_files)
print("finished caching test files")
test_dataset = Dataset(data=test_cache, transform=test_transform())
test_loader = DataLoader(test_dataset, batch_size=batch_size)
#test_image_example.shape : torch.Size([1, 256, 256, 18, 60])
#test_label_example.shape : torch.Size([1, 256, 256, 18])
generator = iter(test_loader)
test_image_example, test_label_example, test_seg_example = next(generator)


test_dataset_visualize_valid = test_dataset[visualize_list_valid]
test_vis_loader_valid = DataLoader(test_dataset_visualize_valid, batch_size=batch_size)




class Unet_2D_dynamic_patch(nn.Module):  # [N,1,a,256,256] > [N,1,1,256,256]
    #https://docs.monai.io/en/stable/_modules/monai/networks/nets/dynunet.html#DynUNet
    #https://github.com/gift-surg/MONAIfbs/blob/main/monaifbs/src/train/monai_dynunet_training.py

    def __init__(self,num_input_channels):
        super(Unet_2D_dynamic_patch, self).__init__()
        self.unet = monai.networks.nets.DynUnet(
        spatial_dims=2,
        in_channels=num_input_channels,
        out_channels=1,  # ktrans
       # num_res_units = 7,  #  channel이 stride들보다 하나 더 많아야 함.
        #res_block = True,, 
        deep_supervision=False,
        kernel_size=[3,3,3,3,3,3,3,3],
        upsample_kernel_size=[2,2,2,2,2,2,2],
        strides=[[1,1],2,2,2,2,2,2,2])        
       # strides=(2,2,2,2,2,2,2)) # stride 개수 + 1 = depth 로 2^(stride개수+1)=256 >7개       
    def forward(self, input):   # 5D > 5D를 4D > 4D로   input N 1 60 256 256 
        """Standard forward"""
        input_4d = input[:,0,:,:,:]  #   N 60 256 256
        output_4d = self.unet.forward(input_4d) # N 1 256 256       
        output_5d = torch.unsqueeze(output_4d,1) # N 1 1 256 256
        output_5d_final = 0.01* output_5d
        return output_5d_final



from utils_dce2ktrans import TemporalConvNet_custom_norelu_3output

num_timeframes = 60
list_channels = [1,32,32,1]
kernel = 7
dropout = 0
net = TemporalConvNet_custom_norelu_3output(num_timeframes,list_channels,kernel,dropout=dropout)
net.forward_test()


zslice_index = 21  # 18 slice중 9번
mriview(test_seg_example[0,:,:,zslice_index])  # 32 32 60
mriview(test_image_example[0,:,:,zslice_index,30],cmap='jet',vmin=0,vmax=1)  # 32 32 60
mriview(test_label_example[0,:,:,zslice_index,0],cmap='jet',vmin=0,vmax=0.1)  # 32 32 60
mriview(test_label_example[0,:,:,zslice_index,1],cmap='jet',vmin=0,vmax=0.1)  # 32 32 60
mriview(test_label_example[0,:,:,zslice_index,2],cmap='jet',vmin=0,vmax=0.1)  # 32 32 60


input = test_image_example[:,:,:,zslice_index,:]  # torch.Size([1, 256, 256, 60])
input_permuted = input.permute((0,3,1,2)) # 1 60 256 256

# 아직 sliding_window_inference GPU쓰면 오히려 느린거 해결이 안됨... 일단 진행
print("Sliding window inference example")
print("shape of input", input_permuted.shape)
print("size of input (bytes)",input_permuted.nelement()*input_permuted.element_size())
test_output = net(input_permuted)  # sliding window에 비해 1분에서 20초로 줄긴했는데
# 단점이, 여전히 메모리 43g먹음. 

mriview(test_output[0,0,...].detach(),cmap='jet')
mriview(test_output[0,1,...].detach(),cmap='jet')
mriview(test_output[0,2,...].detach(),cmap='jet')

#net_d = net.to(device)
#input_permuted_d = input_permuted.to(device)z
#test_output_d = net_d(input_permuted_d)


valid_cache = DiffusionCacheDataset(valid_files)
print("finished caching valid files")

valid_dataset = Dataset(data=valid_cache, transform=valid_transform(num_samples = num_samples, patch_size = patch_size, pos_neg_ratio=pos_neg_ratio))
batch_size = 1
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
generator = iter(valid_loader)
valid_image_example, valid_label_example, valid_seg_example = next(generator)
#valid_image_example.shape : torch.Size([1, 256, 60, 16, 16])
#valid_label_example.shape : torch.Size([1, 256, 60, 16, 16])


class DSC_to_Ktrans_Patch_Model(pl.LightningModule):
    def __init__(self, net, criterion_Ktrans, criterion_Vp, criterion_Ve, learning_rate, optimizer_class,patch_size,sw_batch_size,overlap,test_vis_loader_valid,visualize_slice_list_valid, visualize_list_valid ):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion_Ktrans = criterion_Ktrans
        self.criterion_Vp = criterion_Vp
        self.criterion_Ve = criterion_Ve
        self.optimizer_class = optimizer_class
        self.log_every_epoch = [] 
        self.patch_size = patch_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.test_vis_loader_valid = test_vis_loader_valid
        self.visualize_slice_list_valid = visualize_slice_list_valid
        self.visualize_list_valid = visualize_list_valid

    
    def configure_optimizers(self):
        #optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        optimizer = self.optimizer_class(self.parameters(), lr=(self.lr or self.learning_rate))
        # Auto lr 위해 수정 https://pytorch-lightning.readthedocs.io/en/1.4.5/advanced/lr_finder.html
        return optimizer
    
    def forward(self,batch): # train loop의 forward
        patches_image, patches_label, patches_seg = batch      # 각각 1 256 60 16 16
        patches_forward = self.net(patches_image[0,...])   # 256 60 16 16
        return patches_forward, patches_label[0,...], patches_seg[0,...]  # 셋다 256 60 16 16
    
    def training_step(self, batch, batch_idx):
        y_hat, y , y_seg = self(batch) # forward 함수 적용됨.

    # 230911
        # loss = self.criterion_Ktrans(y_hat[:,0,:,:], y[:,0,:,:], y_seg[:,0,:,:]) + self.criterion_Vp(y_hat[:,1,:,:], y[:,1,:,:], y_seg[:,1,:,:]) + self.criterion_Ve(y_hat[:,2,:,:], y[:,2,:,:], y_seg[:,2,:,:])
        loss1_Ktrans = self.criterion_Ktrans(y_hat[:,0,:,:], y[:,0,:,:], y_seg[:,0,:,:]) 
        loss2_Vp = self.criterion_Vp(y_hat[:,1,:,:], y[:,1,:,:], y_seg[:,1,:,:])
        loss3_Ve = self.criterion_Ve(y_hat[:,2,:,:], y[:,2,:,:], y_seg[:,2,:,:])
 
        loss = loss1_Ktrans + loss2_Vp + loss3_Ve
        #print('yhat,y',y_hat.shape,y.shape)
        self.log('train_loss_batch', loss, on_epoch=True, prog_bar=True)
        self.log(f'train_loss_batch_Ktrans', loss1_Ktrans, on_epoch=True, prog_bar=True)
        self.log(f'train_loss_batch_Vp', loss2_Vp, on_epoch=True, prog_bar=True)
        self.log(f'train_loss_batch_Ve', loss3_Ve, on_epoch=True, prog_bar=True)
        return loss        
    
    # 230911
    def validation_step(self, batch, batch_idx):    
        y_hat, y, y_seg = self(batch) # forward 함수 적용됨.
        loss1_Ktrans = self.criterion_Ktrans(y_hat[:,0,:,:], y[:,0,:,:], y_seg[:,0,:,:]) 
        loss2_Vp = self.criterion_Vp(y_hat[:,1,:,:], y[:,1,:,:], y_seg[:,1,:,:])
        loss3_Ve = self.criterion_Ve(y_hat[:,2,:,:], y[:,2,:,:], y_seg[:,2,:,:])
 
        loss = loss1_Ktrans + loss2_Vp + loss3_Ve
        # loss = self.criterion(y_hat, y, y_seg)
        loss2 = dice_01(y_hat,y)
        loss3 = dice_02(y_hat,y)


        # self.log(f'val_loss_batch - Ktrans: {loss1_Ktrans} | Vp: {loss2_Vp} | Ve: {loss3_Ve}', f'val_loss_batch - Ktrans: {loss1_Ktrans} | Vp: {loss2_Vp} | Ve: {loss3_Ve}', on_epoch=True, prog_bar=True)
        self.log(f'val_loss_batch', loss, on_epoch=True, prog_bar=True)
        self.log(f'val_loss_batch_Ktrans', loss1_Ktrans, on_epoch=True, prog_bar=True)
        self.log(f'val_loss_batch_Vp', loss2_Vp, on_epoch=True, prog_bar=True)
        self.log(f'val_loss_batch_Ve', loss3_Ve, on_epoch=True, prog_bar=True)
        # self.log(f'val_loss_batch_Ktrans', loss1_Ktrans, on_epoch=True, prog_bar=True)
        # self.log(f'val_loss_batch_Vp', loss2_Vp, on_epoch=True, prog_bar=True)
        # self.log(f'val_loss_batch_Ve', loss3_Ve, on_epoch=True, prog_bar=True)
        return {'loss':loss,'loss2':loss2,'loss3':loss3, 'loss1_Ktrans':loss1_Ktrans, 'loss2_Vp': loss2_Vp, 'loss3_Ve':loss3_Ve }

    def validation_epoch_end(self,validation_step_outputs): #https://learnopencv.com/tensorboard-with-pytorch-lightning/
        # print("val_outputs",validation_step_outputs)
        
        avg_loss2 = torch.stack([x['loss2'] for x in validation_step_outputs]).mean()   
        avg_loss3 = torch.stack([x['loss3'] for x in validation_step_outputs]).mean()   
        
    # 230911
        avg_loss1_Ktrans = torch.stack([x['loss1_Ktrans'] for x in validation_step_outputs]).mean()  
        avg_loss2_Vp = torch.stack([x['loss2_Vp'] for x in validation_step_outputs]).mean()  
        avg_loss3_Ve = torch.stack([x['loss3_Ve'] for x in validation_step_outputs]).mean()  
        avg_loss = (avg_loss1_Ktrans + avg_loss2_Vp + avg_loss3_Ve) / 3.
        print("finished epoch:",self.current_epoch)

        self.log("average_val_loss",avg_loss)  # early stopping callback 위해
    # 230911
        self.log(f'average_val_loss_batch', avg_loss)
        self.log(f'average_val_loss_batch_Ktrans', avg_loss1_Ktrans)
        self.log(f'average_val_loss_batch_Vp', avg_loss2_Vp)
        self.log(f'average_val_loss_batch_Ve', avg_loss3_Ve)
    # 230911
        print("epoch", self.current_epoch,"average valid loss",format_4f(avg_loss.item())) 
        print("epoch", self.current_epoch,f'Avg ... Ktrans | Vp | Ve: : {avg_loss1_Ktrans.item()} | {avg_loss2_Vp.item()} | {avg_loss3_Ve.item()}') 
        print("epoch", self.current_epoch,"average dice>0.1 loss (including nontumor slice)",format_4f(avg_loss2.item())) 
        print("epoch", self.current_epoch,"average dice>0.2 loss (including nontumor slice)",format_4f(avg_loss3.item())) 

        self.logger.experiment.add_scalar("average_valid_loss",avg_loss,self.current_epoch) #tensorboard에 log
        tensorboard_logs = {'loss': avg_loss}
        epoch_dictionary={ 
             #required
            'loss' : avg_loss,
            'log' : tensorboard_logs}

        self.log_every_epoch.append(avg_loss)
        loss_list = [x.item() for x in model.log_every_epoch]
        plt.title("loss of last 5 epochs")
        plt.plot(np.arange(len(loss_list))[-5:], loss_list[-5:])
        plt.show()

        generator = iter(self.test_vis_loader_valid)
        self.net.eval()
        
        if os.path.exists(f'{logger.log_dir}/figs')==False:
            os.makedirs(f'{logger.log_dir}/figs')
        with torch.no_grad():
            for i in range(len(self.test_vis_loader_valid)):  # [1,2] 면 0, 1      
                val_idx= self.visualize_list_valid[i] # [15], [15]
                test_vis_image,test_vis_label,test_vis_seg = next(generator)      
                
                for slice in self.visualize_slice_list_valid[i]: #
                    input = test_vis_image[:,:,:,slice,:] # 1 256 256 60 
                    input_permuted = input.permute((0,3,1,2))
                    output = net(input_permuted)
                    #output = sliding_window_inference(input_permuted,patch_size,sw_batch_size,net,overlap)
                    input_img = input[0,:,:,30].detach() # t=30일때 input visualize로 바꿈.
                    #seg_img = test_vis_seg[0,:,:,slice].detach()
                    output_img_0 = output[0,0,...].detach()
                    label_img_0 = test_vis_label[0,:,:,slice,0].detach()
                    output_img_1 = output[0,1,...].detach()
                    label_img_1 = test_vis_label[0,:,:,slice,1].detach()
                    output_img_2 = output[0,2,...].detach()
                    label_img_2 = test_vis_label[0,:,:,slice,2].detach()
                    vis_image = visualize_ktrans_final_save(input_img, label_img_0,output_img_0,"Ktrans", id = f'{logger.log_dir}/figs/Ep{self.current_epoch}_Ktrans_{val_idx}_slice_{slice}')
                    vis_image = visualize_ktrans_final_save(input_img, label_img_1,output_img_1,"Vp", id= f'{logger.log_dir}/figs/Ep{self.current_epoch}_Vp_{val_idx}_slice_{slice}')
                    vis_image = visualize_ktrans_final_save(input_img, label_img_2,output_img_2,"Ve", id= f'{logger.log_dir}/figs/Ep{self.current_epoch}_Ve_{val_idx}_slice_{slice}')



                    # import pdb
                    # pdb.set_trace()

                    # {logger.log_dir}/

        self.net.train()    

        
        return epoch_dictionary
    

# {logger.log_dir}/




model = DSC_to_Ktrans_Patch_Model(
    net=net,
    criterion_Ktrans=criterion_Ktrans,
    criterion_Vp=criterion_Vp,
    criterion_Ve=criterion_Ve, 
    learning_rate=learning_rate,
    optimizer_class=optimizer_class,
    patch_size = patch_size,
    sw_batch_size = sw_batch_size,
    overlap = overlap,
    test_vis_loader_valid = test_vis_loader_valid,
    visualize_slice_list_valid = visualize_slice_list_valid,
    visualize_list_valid = visualize_list_valid
)

logger = TensorBoardLogger(tensorboard_dir_name, name=experiment_name)
checkpoint_callback = ModelCheckpoint(save_top_k=5,monitor="average_val_loss",mode="min")
early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
     monitor="average_val_loss", patience=patience
)


trainer = pl.Trainer(
    #accelerator='ddp', #multii-GPU precessing. dp보다 좋은 방식.
    accelerator = "cpu",
    gpus = gpu_list,
#    gpus = gpu_list, #[2]
    precision = precision,
    callbacks = [checkpoint_callback, early_stopping_callback],
    max_epochs = max_epochs,
#    default_root_dir = checkpoint_path, 
    profiler = "simple",     
    logger = logger,
    num_sanity_val_steps=0,              #  이거하는게 결국 더 나은것같다......... early stopping callback너무 복잡함...
    auto_lr_find=True
)
trainer.logger._default_hp_metric = False #이거 해야 hp_metric끌수 이씀

# 시작

start = datetime.now()

print('Training started at', start)
print("Max epochs",max_epochs)
print('Training duration:', datetime.now() - start)  

print(model.net)
print("Model's state_dict:")
for param_tensor in model.net.state_dict():
    print(param_tensor, "\t", model.net.state_dict()[param_tensor].size())



#run learning rate finder
if use_auto_lr:
    lr_finder = trainer.tuner.lr_find(model=model,train_dataloader=train_loader)
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    # update hparams of the model
    print("lr from lr range test", new_lr)
    model.hparams.lr = new_lr

trainer.fit(model, train_loader, valid_loader)



loss_list = [x.item() for x in model.log_every_epoch]
best_epoch = np.argmin(loss_list)
# 이건 좀 복잡한데, checkpoint가 epoch 1,2,3,4,5,6찍히면 실제론 loss는 7개가 찍힌다. 
# 이중에 
# pytorch lightning의 num_sanity_val_steps=0을 안한다면, 
# validation loop을 시작전에 돌리는데, 이러면 loss고 또 하나 틀어지니 주의ㅋ    !
# 귀찮아서 이 validation loop 돌리는걸 안함.        
best_epoch_path =  glob.glob(logger.log_dir+'/checkpoints/epoch='+str(best_epoch)+'-*')[0]
print("best epoch, model loaded from: ",best_epoch, best_epoch_path)

#best_epoch_path =  tb_logs_dsc/patch3d_lightning_unet/version_118/checkpoints/epoch=18-step=322.ckpt

model_best = DSC_to_Ktrans_Patch_Model.load_from_checkpoint(best_epoch_path,
    net=net,
    criterion_Ktrans=criterion_Ktrans,
    criterion_Vp=criterion_Vp,
    criterion_Ve=criterion_Ve, 
    learning_rate=learning_rate,
    optimizer_class=optimizer_class,
    test_dataset = test_dataset,  # 출력용
    patch_size = patch_size,
    sw_batch_size = sw_batch_size,
    overlap = overlap,
    test_vis_loader_valid = test_vis_loader_valid,
    visualize_slice_list_valid = visualize_slice_list_valid
)

# full visualize
# 한 subject당 2분 정도 걸림. 일단 3명만 해놓음.


test_full_vis_loader = DataLoader(test_dataset, batch_size=batch_size)

image_list = []
output_img_0_list = []
label_img_0_list = []
output_img_1_list = []
label_img_1_list = []
output_img_2_list = []
label_img_2_list = []

seg_img_list = []
generator = iter(test_full_vis_loader)
net.eval()

#num_subjects = len(test_full_vis_loader)

visualize_subject_list = [1,2,3,4,5,6]

with torch.no_grad():
    for i in tqdm(range(max(visualize_subject_list))):       
        test_vis_image,test_vis_label,test_vis_seg = next(generator) 
        if i in visualize_subject_list:     
            for slice in [15,22,29]:
                input = test_vis_image[:,:,:,slice,:] # 1 256 256 60 
                input_permuted = input.permute((0,3,1,2))
                output = net(input_permuted)
                #output = sliding_window_inference(input_permuted,patch_size,sw_batch_size,net,overlap)
                output_img_0 = output[0,0,...].detach()
                label_img_0 = test_vis_label[0,:,:,slice,0].detach()
                output_img_1 = output[0,1,...].detach()
                label_img_1 = test_vis_label[0,:,:,slice,1].detach()
                output_img_2 = output[0,2,...].detach()
                label_img_2 = test_vis_label[0,:,:,slice,2].detach()
                seg_img = test_vis_seg[0,:,:,slice].detach()

                output_img_0_list.append(output_img_0)
                label_img_0_list.append(label_img_0)
                output_img_1_list.append(output_img_1)
                label_img_1_list.append(label_img_1)
                output_img_2_list.append(output_img_2)
                label_img_2_list.append(label_img_2)

                seg_img_list.append(seg_img)
                vis_image_0 = nonvisualize_ktrans_compare_save(label_img_0,output_img_0,id = f'{logger.log_dir}/test_id_{i}-slice_{slice}_0')
                vis_image_1 = nonvisualize_ktrans_compare_save(label_img_1,output_img_1,id = f'{logger.log_dir}/test_id_{i}-slice_{slice}_1')
                vis_image_2 = nonvisualize_ktrans_compare_save(label_img_2,output_img_2,id = f'{logger.log_dir}/test_id_{i}-slice_{slice}_2')
                vis_image_fuse = np.concatenate((vis_image_0,vis_image_1,vis_image_2),axis=1)
                image_list.append(vis_image_fuse)

for image in image_list:
    plt.imshow(np.swapaxes(image,0,2))
    plt.gcf().set_dpi(500)
    plt.ylabel('Prediction    |  Label')
    plt.xlabel('Ktrans    |     Vp      |  Ve')
    plt.show()
    


# def show_six_slices(a,b,c,d,e,f):
#     image_1st_row = np.concatenate((a,b,c),axis=1)
#     image_2nd_row = np.concatenate((d,e,f),axis=1)
#     image_grid = np.concatenate((image_1st_row,image_2nd_row),axis=2)
#     plt.imshow(np.swapaxes(image_grid,0,2))
#     plt.gcf().set_dpi(500)
#     plt.axis('off')
#     plt.show()

# vis_image_fuse = np.concatenate((vis_image_0,vis_image_1,vis_image_2),axis=1)
# plt.imshow(np.swapaxes(vis_image_fuse,0,2))
# plt.gcf().set_dpi(500)
# plt.ylabel('Prediction    |  Label')
# plt.xlabel('Ktrans    |     Vp      |  Ve')
# plt.show()

#    plt.imshow(np.swapaxes(vis_image_0,0,2))
# """ for i in range(int(len(image_list)/6)):
#     show_six_slices(image_list[6*i],image_list[6*i+1],image_list[6*i+2],image_list[6*i+3],image_list[6*i+4],image_list[6*i+5])
#  """
# for i in [1,4,7]:
#     show_six_slices(image_list[6*i],image_list[6*i+1],image_list[6*i+2],image_list[6*i+3],image_list[6*i+4],image_list[6*i+5])


# output_img_stack = torch.stack(output_img_list)
# label_img_stack = torch.stack(label_img_list)
# seg_img_stack = torch.stack(seg_img_list)

# output_img_stack = torch.reshape(torch.stack(output_img_list),(-1,))
# label_img_stack = torch.reshape(torch.stack(label_img_list),(-1,))
# seg_img_stack = torch.reshape(torch.stack(seg_img_list),(-1,))

# threshold = 0
# x = label_img_stack[label_img_stack>threshold]
# y = output_img_stack[label_img_stack>threshold]
# seg = seg_img_stack[label_img_stack>threshold]
# limit = max(max(x),max(y))
# #limit = 1
# plt.scatter(x, y,alpha=0.5)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.title("ktrans vs predict ktrans for all voxels #"+str(x.shape[0])+ "(3 subjects) \n"+"corrcoeff:"+str(np.corrcoef(x,y)[0][1]))
# plt.show()

# x_seg = x[seg==1]
# y_seg = y[seg==1]
# limit_seg = max(max(x_seg),max(y_seg))
# #limit_seg = 1
# plt.scatter(x_seg, y_seg,alpha=0.5)
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.title("ktrans vs predict ktrans for tumor voxels #"+str(x_seg.shape[0])+"(3 subjects) \n"+"corrcoeff:"+str(np.corrcoef(x_seg,y_seg)[0][1]))
# plt.show()


# print("average dice >0.1 score for slices containing tumor ( 3 subjects):", format_4f(np.mean(dice01_list)))
# print("average dice >0.2 score for slices containing tumor ( 3 subjects):", format_4f(np.mean(dice02_list)))
# print("correlation coeff all voxels (3 subjects)",format_4f(np.corrcoef(x,y)[0][1]))
# print("correlation coeff tumor voxels (3 subjects)",format_4f(np.corrcoef(x_seg,y_seg)[0][1]))



class inference_transform(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = torch.from_numpy(sample['image']).float() # 256 256 18 60
        label = torch.from_numpy(sample['label']) # 256 256 18
        seg = torch.from_numpy(sample['seg']) # 256 256 18       
        path = sample['path'] 
        return image, label, seg, path

forward_name = "tcn_3fuse_baseline_seperate_loss_240830"
forward_filename = "forwarded_"+forward_name+'.nii.gz'

# train_files_full = make_nii_path_dataset(base_dir+"/train",input_filename,label_filename,seg_filename,mask_filename, 1)
# valid_files_full = make_nii_path_dataset(base_dir+"/valid",input_filename,label_filename,seg_filename,mask_filename, 1)
# test_files_full = make_nii_path_dataset(base_dir+"/test",input_filename,label_filename,seg_filename,mask_filename, 1)

# net.eval()
# with torch.no_grad():
#     for selected_files in [train_files_full,valid_files_full,test_files_full]:   
#         selected_files_cache = DiffusionCacheDataset(selected_files)
#         selected_dataset = Dataset(data = selected_files_cache, transform=inference_transform())
#         selected_dataloader = DataLoader(selected_dataset, batch_size = 1)       
#         generator = iter(selected_dataloader)
#         for i in tqdm(range(len(selected_dataloader))):
#             input,label,seg,path = next(generator) # input.shape [1, 256, 256, 16, 60])
#             num_slices = input.shape[-2]
#             refvol = nib.load(path[0]+'/g_kvpve_fixed.nii.gz')
#             output_vol = np.zeros((256,256,num_slices,3))        # 256 256 40 3   
#             for slice in range(num_slices):
#                 input_slice = input[:,:,:,slice,:]  # 3 256 256 60
#                 input_slice_permuted = input_slice.permute((0,3,1,2)) # 3 60 256 256
#                 output_slice = net(input_slice_permuted) # 3 60 256 256
#                 output_slice_permuted = output_slice.permute((0,2,3,1)) # 1 256 256 3
#                 output_vol[:,:,slice,:] = output_slice_permuted[0,:,:,:].detach().numpy()
#             save_path = path[0]+'/'+forward_filename
#             nft_img = nib.Nifti1Image(output_vol,refvol.affine,refvol.header)
#             nib.save(nft_img, save_path)


# # Forward하는데 한 subject당 1분
# # 전체 237 subject에 5시간정도 걸릴 예정
    
# #     best epoch, model loaded from:  18 tb_logs_dsc/patch3d_lightning_unet/version_118/checkpoints/epoch=18-step=322.ckpt

# # g_kvpve_fixed 256 256 40 3 
# # g_ktrans 256 256 40 1

# # #test_vis_image.shape torch.Size([1, 256, 256, 40, 60])
# # #test_vis_label.shape torch.Size([1, 256, 256, 40, 3])
# # #test_vis_seg.shape  torch.Size([1, 256, 256, 40]


# # )#input.shape  torch.Size([1, 256, 256, 60])
# # #input_permuted.shape  torch.Size([1, 60, 256, 256])
# # #output.shape torch.Size([1, 3, 256, 256])


# # 최종저장도 g_kvpve_forwarded로 저장하자 256 256 40 3

# x = glob.glob("/mnt/ssd/ylyoo/intermediate_filtered_split/train/*/forwarded_tcn_3fuse_baseline.nii.gz")
# y = glob.glob("/mnt/ssd/ylyoo/intermediate_filtered_split/train/*/g_kvpve_fixed.nii.gz")
# x2 = [a.split('/')[-2] for a in x]
# y2 = [a.split('/')[-2] for a in y]
# z = [item for item in y2 if item not in x2] # 40264244


# pth = "/mnt/ssd/ylyoo/intermediate_filtered_split/train/40264244/"
# print(pth + input_filename)

# temp_dataset = [{"image":pth+input_filename , "label":pth+label_filename, "seg" :pth+seg_filename , "mask": pth+mask_filename, "path":pth}]

# input_filename,label_filename,seg_filename,mask_filename
