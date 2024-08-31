############################## # 003_DCE_Final_3dpatch_lightning_50_tcn_norelu_prob_unet_baseline_0418_pipeline_inference_try001_10.py
## Control Parameters
folder_path = 'results_3output_240830_align' # folder_path = 'results_3output_230415_align_4_try5' # folder_path = 'results_3output_230412_align_4_try0001'
hash_dict = dict({0:'1_try1_confirm'})# , 1:'1_try2', 2:'1_try3', 3:'4_try1', 4:'4_try_2', 5:'9_try1'})  
N_iter_infer = 1 # N_iter_infer = 4
N_list = list(range(1)) # 저장해둔 N개 중 1개 추출
N_iter_infer_list_list = [list(range(1))] # 1번만 inference 한 것 대비 10번 한 것의 우수성 보려면 1번 샘플 VS 10번 샘플 평균치 통계 비교 필요
best_epoch_path = '/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/240605_dce2ktrans/tb_logs_dsc/patch3d_lightning_unet/version_0/checkpoints/epoch=35-step=6227.ckpt' # 'tb_logs_dsc/patch3d_lightning_unet/version_237_selected_0412_18/checkpoints/epoch=35-step=6227.ckpt'
##################### Setting #############################

##### library import #####
verbose = False # Print Flag for Debugging || True: print on, False: print off
#https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
# # MS-SSSIM은 32이상일때만 적용가능해서 32로 바꿈. 
from functools import total_ordering
from operator import index
import os
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=2

import torch
torch.set_num_threads(8)
import torch.nn as nn

import numpy as np
from datetime import datetime
import glob
import random
random.seed(random.randint(1, 100000))
import matplotlib.pyplot as plt
import nibabel as nib

from tqdm import tqdm

from utils_dce2ktrans import nii_to_numpy, visualize_ktrans_patch, Weighted_L1_patch
from utils_dce2ktrans import FixedRandCropByPosNegLabel_custom_coordinates, RandCropByPosNegLabel_custom_coordinates, apply_patch_coordinates_4d, apply_patch_coordinates, apply_patch_coordinates_ch1
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

from utils_dce2ktrans import format_4f, nii_to_numpy,path_cutter, mriview, Weighted_L1_patch, visualize_ktrans_patch, visualize_ktrans_patch_30, visualize_ktrans_patch_30_uncertainty, visualize_ktrans_compare, nonvisualize_ktrans_compare
from utils_dce2ktrans import TemporalBlock, tile_3d_to_60_torch

device = torch.device('cpu')
# print('torch.cuda.is_available:', torch.cuda.is_available())
# print('torch.cuda.device_count():',  torch.cuda.device_count())
# print('torch.cuda.current_device():',  torch.cuda.current_device())
# print('torch.cuda.get_device_name(0):',torch.cuda.get_device_name(0))

import pdb
import glob
from PIL import Image
import matplotlib.pyplot as plt
import copy

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#######  너무 오래 걸리니 꺼두기*   ###### 
#This code is based on: https://github.com/SimonKohl/probabilistic_unet
from unet_blocks import *
from unet import Unet
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from probabilistic_unet import *

############################## Probabilistic U-Net implement! ####################################
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
# from probabilistic_unet import add_ex
# from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation

import glob
from datetime import datetime
import time

set_determinism(seed=0)
tensorboard_dir_name = "tb_logs_dsc"
experiment_name = "patch3d_lightning_unet" 
test_forward = False # True시 이를 forward하는 기능
forward_name = "exp1"
forward_filename = "forwarded_"+forward_name+'.nii.gz'

gpu_list = [0]
input_filename = 'forwarded_dce_baseline_2.nii.gz' #'forwarded_dce_baseline.nii.gz' # 'g_dce_01.nii.gz' # 'g_dce_relax.nii.gz'  # input_filename = 'g_dce_01.nii.gz' 
# 'forwarded_dce_baseline.nii.gz' (256, 256, 40)
# 'g_dce_01.nii.gz' (256, 256, 40, 60)

label_filename = 'g_kvpve_fixed.nii.gz' # 'g_ktrans.nii.gz' # Ktans: (256, 256, 40) #  kvpve (256, 256, 40, 3)
mask_filename = 'g_dce_mask.nii.gz' # (256, 256, 40)
seg_filename = 'g_seg.nii.gz' # (256, 256, 40)

# f_dsc    256 256 18 60     > 341개
# f_ktrans 256 256 18        > 341개
# seg_DSCspace.nii.gz 256 256 18  > 237개

base_dir = "/mnt/ssd/ylyoo/intermediate_filtered_split"

# g_seg.nii.gz
# cd /mnt/ssd/ylyoo/intermediate_filtered_split/input_filename
# input, label, mask, seg  sample dimension check!
# import nibabel as nib
# from glob import glob
# folder_dir = f'{base_dir}/train'
# imgs_dir = glob(f'{folder_dir}/*/{label_filename}')
# print(imgs_dir)
# print(f'imgs_dir: {imgs_dir}')

# img_path = imgs_dir[0]

# label_filename = 'g_kvpve_fixed.nii.gz' # (256, 256, 40)
# mask_filename = 'g_dce_mask.nii.gz' # (256, 256, 40)
# seg_filename = 'g_seg.nii.gz' # (256, 256, 40)

# img = nib.load(img_path)
# img_data = img.get_fdata()

# print(img_data.shape)
# nib.load(glob(f'{folder_dir}/*/g_kvpve_fixed.nii.gz')[0]).get_fdata().shape # (256, 256, 40)  # ..(256, 256, 40)
# nib.load(glob(f'{folder_dir}/*/{label_filename}')[0]).get_fdata().shape # (256, 256, 40)  # ..(256, 256, 40)
# nib.load(glob(f'{folder_dir}/*/g_dce_mask.nii.gz')[0]).get_fdata().shape # (256, 256, 40)
# nib.load(glob(f'{folder_dir}/*/g_seg.nii.gz')[0]).get_fdata().shape # (256, 256, 40)

'''
# # nifti 이미지 읽기
# import nibabel as nib
# img = nib.load('image path')
# img_data = img.get_fdata()

# # nifti 이미지 저장하기
# nii_img = nib.Nifti1Image(img_data, img.affine, img.header)
# nib.save(nii_img, 'path to save')

# # dicom 이미지 읽기
# import pydicom
# img = pydicom.dcmread('image path')
# img_data = img.pixel_array()
'''

###################### full_image로 변경 필요 - 이하 부분 ##########################
# patch 적용
patch_size = (64,64)  # z축 방향 patch는 일단 안함. 복잡해짐.
num_samples = 16
pos_neg_ratio = 0.5   # 중요 > tumor 비율

batch_size = 1
# train_loader의 batch size의 경우 dataloader의 batch size 에
# num_samples가 곱해진다!
# 즉 실 batch_size = batch_size * num_samples
sw_batch_size = 10000  # sliding window forward시 이때의 batch size. 무제한이 좋음.
overlap = 0.5 # sliding window overlap 0.5가 좋음. 이건 무조건 고정. 

##### 빠른 training 위해 일단 전체 데이터의 10~20% 선택
train_ratio = 0.07# 0.1 # 1.0    
valid_ratio = 0.07# 0.2 # 1.0 
test_ratio = 1.0
###########
max_epochs = 100
patience = 13  # 5번안에 개선안되면 중단하는 early stopping

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

#################  여기 dimension  주의! #230117 ######################
class Ktrans_weighted_L1_and_exponentialLoss(nn.Module):    
# abs(y-ktrans)^m * (ktrans > epsilon ) * (ktrans^n) 꼴의 loss 
# ktrans < epsilon인 작은 것들은 아예 loss를 주지 않는다. 
    def __init__(self,epsilon,alpha, m,n):
        super(Ktrans_weighted_L1_and_exponentialLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.m = m
        self.n = n
        self.print_parameters()
    def forward(self, predictions, target, seg):  
        # print(f'loss pred_dim:{predictions[0].shape}')
        # print(f'loss target_dim: {target.shape}')
        loss_mn = torch.mean(torch.pow(torch.abs(predictions-target),self.m)*torch.pow(target,self.n)*(target>self.epsilon))
        loss_l1 = torch.mean(torch.abs(predictions-target))
        loss = loss_l1 + self.alpha *loss_mn
        #print(loss_l1, self.alpha*loss_mn)
        return loss
    def print_parameters(self):
        print("Loss - Epsilon, alpha, m,n ", self.epsilon, self.alpha, self.m, self.n)

#$3$#$230310
class DiceLoss(nn.Module):
    # 엄밀히는 loss가 아님. loss로 하려면 
    #  (x/epsilon)으로 나눠준다음에 real prediction 기준으로 해야
    def __init__(self,epsilon):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, inputs_raw, targets_raw, smooth=1):    
        ## inputs_raw, targets_raw dim 체크!
        # if 

        inputs = (inputs_raw>self.epsilon)
        targets = (targets_raw>self.epsilon)
        """ 
        inputs = inputs.view(-1)
        targets = targets.view(-1)       """
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        return dice 

criterion = Ktrans_weighted_L1_and_exponentialLoss(epsilon =0,alpha = 40, m =1,n=1) # alpha = 40 추천!
dice_01 = DiceLoss(epsilon=0.1)
dice_02 = DiceLoss(epsilon=0.2)

#criterion = torch.nn.MSELoss()
#criterion = torch.nn.L1Loss()
#epsilon = 0.007
#criterion = Weighted_L1_patch(epsilon = epsilon)

use_auto_lr = False  # 잘안되서 끔.. 값만 참조
precision = 32  #16이 더 빠른거같은데.. torch 1.16인가 설치해야 cpu에서 할수있따한다.
output_scale = 1

# valid epoch 마다 비교할거
visualize_list_valid = [2,6]
visualize_slice_list_valid = [[8],[5]]  

print("patch size :", patch_size)
print("number of patch samples per slices",num_samples)
print("batch size for dataloader", batch_size)
print("effective batch size = num_samples*batch_size", batch_size*num_samples)
print("PosNegRatio ", pos_neg_ratio)
print("max_epochs", max_epochs)
print("patience",patience)
print("optimizer_class",optimizer_class)
print("learning rate",learning_rate)
print("loss function", criterion)

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

base_dir = "/mnt/ssd/ylyoo/intermediate_filtered_split"
label_filename = 'g_kvpve_fixed.nii.gz' # 'g_ktrans.nii.gz' # Ktans: 256, 256, 40 #  kvpve (256, 256, 40, 3)
mask_filename = 'g_dce_mask.nii.gz' # (256, 256, 40)
seg_filename = 'g_seg.nii.gz' # (256, 256, 40)

os.makedirs(tensorboard_dir_name,exist_ok=True) # 없을 때만 만든다. 

def make_nii_path_dataset(data_dir,input_filename,label_filename,seg_filename,mask_filename,ratio_select = 1):

###### IF1. TCN Output이 input으로 들어오는 경우! ######

###### IF2. DCE_input이 input으로 들어오는 경우! ######
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


train_files = make_nii_path_dataset(base_dir+"/train",input_filename,label_filename,seg_filename,mask_filename, train_ratio) # 
valid_files = make_nii_path_dataset(base_dir+"/valid",input_filename,label_filename,seg_filename,mask_filename, valid_ratio)
test_files = make_nii_path_dataset(base_dir+"/test",input_filename,label_filename,seg_filename,mask_filename, test_ratio)


def make_nii_path_dataset_load(data_output_dir, data_dir,input_filename,label_filename,seg_filename,mask_filename,ratio_select = 1):

###### IF1. TCN Output이 input으로 들어오는 경우! ######
###### IF2. DCE_input이 input으로 들어오는 경우! ######
    # seg있는데 image 없는것도 있으니 seg기준으로 만들고 replace로 image, label path만듬
    path_segs = sorted(glob.glob(data_dir+"/*/"+seg_filename))
    path_images = [x.replace(seg_filename,input_filename) for x in path_segs]

    # path = sorted(glob.glob(data_output_dir +  ))

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

# 결론 - nonramdomizable transform까지 다 cache 를 entire dataset을 하고
# non random만 dataloader사용해 접근하자.

class DiffusionCacheDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files
        self.cache_files = []
        for i in range(len(data_files)):
            vol_image = nii_to_numpy(self.data_files[i]['image']) # 230308 TCN_output 넣음      # relaxivity라 - 넣어준거조심!!
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

# ############### check now !!! - patches_image  input: 60 channel ####################
# patches_image: torch.Size([1, 16, 60, 64, 64])
# patches_label: torch.Size([1, 16, 60, 64, 64])
# patches_image.shape : torch.Size([1, 16, 60, 64, 64])


# ############### check now !!! - patches_image  input: 1 channel ####################
# patches_image: torch.Size([1, 16, 60, 64, 64])
# patches_label: torch.Size([1, 16, 60, 64, 64])
# patches_image.shape : torch.Size([1, 16, 60, 64, 64])

class train_transform(object): 
     # return patches of size N 60 16 16 from 'image', 'label', 'seg'
    def __init__(self, num_samples,patch_size,pos_neg_ratio):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.pos_neg_ratio = pos_neg_ratio

        self.verbose = False # Debugging 용도!

    def __call__(self, sample):
        if sample['image'].ndim == 4:
            image = torch.from_numpy(sample['image']).float() # 256 256 18 60  # 256 256 18 1
        elif sample['image'].ndim == 3:
            image = torch.from_numpy(sample['image']).unsqueeze(-1).float() # 256 256 18 60  # 256 256 18 1
                                                        # 60 channel: 256 256 18 60
                                                        # 1 channel: torch.Size([256, 256, 40])
        if sample['label'].ndim == 4:
            label = torch.from_numpy(sample['label']) # 256 256 18 60  # 256 256 18 1
        elif sample['label'].ndim == 3:
            label = torch.from_numpy(sample['label']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        # label = torch.from_numpy(sample['label']) # 256 256 18

        # if sample['seg'].ndim == 4:
        #     seg = torch.from_numpy(sample['seg']) # 256 256 18 60  # 256 256 18 1
        # elif sample['seg'].ndim == 3:
        #     seg = torch.from_numpy(sample['seg']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        seg = torch.from_numpy(sample['seg']) # 256 256 18 ## 뒤에  RandCropByPosNegLabel_custom_coordinates 여기서 처리되므로 건들면 안됨

        # if sample['mask'].ndim == 4:
        #     mask = torch.from_numpy(sample['mask']) # 256 256 18 60  # 256 256 18 1
        # elif sample['mask'].ndim == 3:
        #     mask = torch.from_numpy(sample['mask']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        mask = torch.from_numpy(sample['mask'])

            # 230309 kvpve.nii.gz 기준, image: torch.Size([256, 256, 40, 1])
            # label: torch.Size([256, 256, 40, 3])
            # seg: torch.Size([256, 256, 40, 1])

# dimension 맞춰주는 건 대부분 여기서 하자!
        patch_size = torch.tensor([self.patch_size[0],self.patch_size[1],1]) # torch.Size([3])   # torch.tensor([16,16,1])  
        num_patch_samples = self.num_samples
        pos_neg_ratio = self.pos_neg_ratio
        #print(sample['path'])
        #print(image.shape,label.shape,seg.shape,mask.shape,patch_size.shape,num_patch_samples)        
        coords = RandCropByPosNegLabel_custom_coordinates(seg,patch_size,num_patch_samples,pos_neg_ratio)
        #print(coords.shape)
        # patches_image = apply_patch_coordinates_ch1(image,coords) # 230307 # channel == 1 
        patches_image = apply_patch_coordinates_4d(image,coords)  # 230307  [16, 1, 64, 64] ### channel > 1. e.g. 60) ### 1000 60 16 16
        patches_label = apply_patch_coordinates_4d(label,coords)  # 230307  [16, 3, 64, 64] ### channel > 1. e.g. 3)  ### 1000 60 16 16
        patches_seg = apply_patch_coordinates_ch1(seg,coords)   # 1000 60 16 16
        # patches_seg = apply_patch_coordinates_4d(seg,coords)     # 230307  [16, 1, 64, 64] ### 1000 60 16 16
        # patches_image = apply_patch_coordinates(image,coords) # 230307 # channel == 1 
        # # patches_image = apply_patch_coordinates_4d(image,coords) # 1000 60 16 16 # 230307 # channel > 1. e.g. 60)
        # patches_label = apply_patch_coordinates(label,coords)   # 1000 60 16 16
        # patches_seg = apply_patch_coordinates(seg,coords)      # 1000 60 16 16
    # channel == 60
        # 1000 60 16 16 꼴로 정리
        # self.verbose = False
        # if self.verbose == True:        
        #     print('\n\ntrain_transform!!!!! 518 line: \n\n')
        #     print(f'image: {image.shape}')
        #     print(f'label: {label.shape}')
        #     print(f'seg: {seg.shape}')
        #     print(f'train_coords:{coords}')
        return patches_image, patches_label, patches_seg

# def apply_patch_coordinates_4d(torch_array_4d, coords):
#     patches = [apply_patch_coordinates_4d_single(torch_array_4d,coord) for coord in coords]
#     patches_stacked = torch.stack(patches) # 1000 16 16 1 60 
#     patches_stacked_final = patches_stacked[:,:,:,0,:].permute((0,3,1,2)) # 1000 60 16 16
#     return patches_stacked_final

class valid_transform(object): 
     # return patches of size N 60 16 16 from 'image', 'label', 'seg'
    def __init__(self, num_samples,patch_size,pos_neg_ratio):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.pos_neg_ratio = pos_neg_ratio
        
        self.verbose = False # Debugging 용도
        
    def __call__(self, sample):
        if sample['image'].ndim == 4:
            image = torch.from_numpy(sample['image']).float() # 256 256 18 60  # 256 256 18 1
        elif sample['image'].ndim == 3:
            image = torch.from_numpy(sample['image']).unsqueeze(-1).float() # 256 256 18 60  # 256 256 18 1
                                                        # 60 channel: 256 256 18 60
                                                        # 1 channel: torch.Size([256, 256, 40])
        if sample['label'].ndim == 4:
            label = torch.from_numpy(sample['label']) # 256 256 18 60  # 256 256 18 1
        elif sample['label'].ndim == 3:
            label = torch.from_numpy(sample['label']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        # label = torch.from_numpy(sample['label']) # 256 256 18

        # if sample['seg'].ndim == 4:
        #     seg = torch.from_numpy(sample['seg']) # 256 256 18 60  # 256 256 18 1
        # elif sample['seg'].ndim == 3:
        #     seg = torch.from_numpy(sample['seg']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        seg = torch.from_numpy(sample['seg']) # 256 256 18

        if sample['mask'].ndim == 4:
            mask = torch.from_numpy(sample['mask']) # 256 256 18 60  # 256 256 18 1
        elif sample['mask'].ndim == 3:
            mask = torch.from_numpy(sample['mask']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        
        path = sample['path']
        x = hash(path)%(2**32)
        patch_size = torch.tensor([self.patch_size[0],self.patch_size[1],1]) # torch.tensor([16,16,1])
        num_patch_samples = self.num_samples
        pos_neg_ratio = self.pos_neg_ratio
        coords = FixedRandCropByPosNegLabel_custom_coordinates(x, seg,patch_size,num_patch_samples,pos_neg_ratio)

        # patches_image = apply_patch_coordinates_ch1(image,coords) # 230307 # channel == 1 
        patches_image = apply_patch_coordinates_4d(image,coords)  # 230307  [16, 1, 64, 64] ### channel > 1. e.g. 60) ### 1000 60 16 16
        patches_label = apply_patch_coordinates_4d(label,coords)  # 230307  [16, 3, 64, 64] ### channel > 1. e.g. 3)  ### 1000 60 16 16
        # patches_label = apply_patch_coordinates_ch1(label,coords)   # 1000 60 16 16
###########*#*#*#*##########
        patches_seg = apply_patch_coordinates_ch1(seg,coords)     # 230307 [16, 64, 64] > [16, 1, 64, 64] ### 1000 60 16 16 ####### 230309 이게 최선인가?
        # patches_image = apply_patch_coordinates(image,coords) # 230307 # channel == 1 
        # # patches_image = apply_patch_coordinates_4d(image,coords) # 1000 60 16 16 # 230307 # channel > 1. e.g. 60)
        # patches_label = apply_patch_coordinates(label,coords)   # 1000 60 16 16
        # patches_seg = apply_patch_coordinates(seg,coords)      # 1000 60 16 16
    # channel == 60
        # 1000 60 16 16 꼴로 정리

        # self.verbose = False
        # if self.verbose == True:
        #     print('\n\nval_transform!!!!! 518 line: \n\n')
        #     print(f'image: {image.shape}')
        #     print(f'label: {label.shape}')
        #     print(f'seg: {seg.shape}')
        #     print(f'mask: {mask.shape}')  
        #     print(f'valid_coords:{coords}')
        #     print(f'patches_seg:{patches_seg.shape}')
        return patches_image, patches_label, patches_seg

''' 230309 내용 정리 '''
# image: torch.Size([256, 256, 40, 1])
# label: torch.Size([256, 256, 40, 3])
# seg: torch.Size([256, 256, 40])
# mask: torch.Size([256, 256, 40, 1])
# valid_coords:[{'lower': tensor([181,   9,  14]), 'upper': tensor([245,  73,  15])}, {'lower': tensor([ 89, 189,  17]), 'upper': tensor([153, 253,  18])}, {'lower': tensor([ 95, 100,  24]), 'upper': tensor([159, 164,  25])}, {'lower': tensor([106,  93,  27]), 'upper': tensor([170, 157,  28])}, {'lower': tensor([ 31, 113,  12]), 'upper': tensor([ 95, 177,  13])}, {'lower': tensor([30,  0,  4]), 'upper': tensor([94, 64,  5])}, {'lower': tensor([114,  92,  29]), 'upper': tensor([178, 156,  30])}, {'lower': tensor([121,  95,  27]), 'upper': tensor([185, 159,  28])}, {'lower': tensor([131,  99,  28]), 'upper': tensor([195, 163,  29])}, {'lower': tensor([114,  88,  31]), 'upper': tensor([178, 152,  32])}, {'lower': tensor([127,  67,  13]), 'upper': tensor([191, 131,  14])}, {'lower': tensor([69, 57,  8]), 'upper': tensor([133, 121,   9])}, {'lower': tensor([ 92, 108,  25]), 'upper': tensor([156, 172,  26])}, {'lower': tensor([110,  86,  32]), 'upper': tensor([174, 150,  33])}, {'lower': tensor([164, 126,   0]), 'upper': tensor([228, 190,   1])}, {'lower': tensor([154,   0,   0]), 'upper': tensor([218,  64,   1])}]
# patches_seg:torch.Size([16, 1, 64, 64])
# finished epoch: 7
# epoch 7 average valid loss 0.8589
# epoch 7 average dice>0.1 loss (including nontumor slice) 0.2537
# epoch 7 average dice>0.2 loss (including nontumor slice) 0.2320

class test_transform(object): # 왜 또 test_transform만 format이 다르지?ㅠ 230309
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        self.verbose = False # Debugging 용도!
        
        if sample['image'].ndim == 4:
            image = torch.from_numpy(sample['image']).float() # 256 256 18 60  # 256 256 18 1
        elif sample['image'].ndim == 3:
            image = torch.from_numpy(sample['image']).unsqueeze(-1).float() # 256 256 18 60  # 256 256 18 1
                                                        # 60 channel: 256 256 18 60
                                                        # 1 channel: torch.Size([256, 256, 40])
        if sample['label'].ndim == 4:
            label = torch.from_numpy(sample['label']) # 256 256 18 60  # 256 256 18 1
        elif sample['label'].ndim == 3:
            label = torch.from_numpy(sample['label']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        # label = torch.from_numpy(sample['label']) # 256 256 18

        # if sample['seg'].ndim == 4:
        #     seg = torch.from_numpy(sample['seg']) # 256 256 18 60  # 256 256 18 1
        # elif sample['image'].ndim == 3:
        #     seg = torch.from_numpy(sample['seg']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        seg = torch.from_numpy(sample['seg']) # 256 256 18

        if sample['mask'].ndim == 4:
            mask = torch.from_numpy(sample['mask']) # 256 256 18 60  # 256 256 18 1
        elif sample['mask'].ndim == 3:
            mask = torch.from_numpy(sample['mask']).unsqueeze(-1) # 256 256 18 60  # 256 256 18 1
        
        path = sample['path'] 
        # if self.verbose == True:
        #     print('\n\ntest_transform!!!!! 518 line: \n\n')
        #     print(f'image: {image.shape}')
        #     print(f'label: {label.shape}')
        #     print(f'seg: {seg.shape}')
        return image, label, seg, path # helee
'''
image: torch.Size([256, 256, 40, 1])
label: torch.Size([256, 256, 40, 3])
seg: torch.Size([256, 256, 40])
'''

train_cache = DiffusionCacheDataset(train_files)
print("finished caching train files")
train_dataset = Dataset(data=train_cache, transform=train_transform(num_samples = num_samples, patch_size = patch_size, pos_neg_ratio=pos_neg_ratio))
train_loader = DataLoader(train_dataset, batch_size=batch_size)
generator = iter(train_loader)
patches_image_example, patches_label_example, patches_seg_example = next(generator)

# patches_image_example: torch.Size([1, 16, 1, 64, 64])
# patches_label_example: torch.Size([1, 16, 3, 64, 64])
# patches_seg_example: torch.Size([1, 16, 1, 64, 64])

fig, axs = plt.subplots(4,4)
print("example train patches : input")
for i in range(4):
    for j in range(4):
        axs[i,j].axis('off')
        axs[i,j].imshow(patches_image_example[0,4*i+j,0,:,:]) # cmap='bwr' # cmap='jet',vmin=0.0,vmax=1.0
plt.show()
fig, axs = plt.subplots(4,4)
print("example train patches : label")
for i in range(4):
    for j in range(4):
        axs[i,j].axis('off')
        axs[i,j].imshow(patches_label_example[0,4*i+j,0,:,:])
plt.show()
print("example train patches : seg")
fig, axs = plt.subplots(4,4)
for i in range(4):
    for j in range(4):
        axs[i,j].axis('off')
        axs[i,j].imshow(patches_seg_example[0,4*i+j,0,:,:])
plt.show()

#################################################################
# Dataset-TEST
#################################################################

test_cache = DiffusionCacheDataset(test_files)
print("finished caching test files")
test_dataset = Dataset(data=test_cache, transform=test_transform())
test_loader = DataLoader(test_dataset, batch_size=batch_size)
#test_image_example.shape : torch.Size([1, 256, 256, 18, 60])
#test_label_example.shape : torch.Size([1, 256, 256, 18])
generator = iter(test_loader)

test_image_example, test_label_example, test_seg_example, path = next(generator)

test_dataset_visualize_valid = test_dataset[visualize_list_valid]
test_vis_loader_valid = DataLoader(test_dataset_visualize_valid, batch_size=batch_size)

print(f'path: {path}')

##### 230308 ##### TCN output인 경우 1 channel
from utils_dce2ktrans import TemporalConvNet_custom_norelu
num_timeframes = 1
##### 230308 ##### DCE input인 경우 60 channel
# num_timeframes = 60

list_channels = [1,32,32,1]
kernel = 7
dropout = 0

net = TemporalConvNet_custom_norelu(num_timeframes,list_channels,kernel,dropout=dropout)
# net.forward_test()
# input dim [1, 60, 256, 256]
# import 잘 안되는 문제가 있어서 파일 자체에 불러옴!
# from probabilistic_unet import add_ex
# add_ex(2,3)

    
class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False, segm_dim=3):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters # ??

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            # self.input_channels += 1 # 230117
            self.input_channels += segm_dim # 230309: 원래는 ktrans 하나라 +1인데, Vp, Ve 추가 되면서 +3 됨 

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim # 처음만 input_dim 나머지는 output_dim
            output_dim = num_filters[i] # 0, 1, 2, ---- self.num_filters-1
            
            if i != 0: # 처음 input dl dkslaus..
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False, segm_dim=3):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior # False, True
        if self.posterior:  # True인 경우 name 설정 Posterior
            self.name = 'Posterior'
        else:               # False인 경우 name 설정 Prior
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior, segm_dim=segm_dim) # 여기 posterior는 True, False

        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu') ### Q1 fan_in ??
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):
        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None: ### Q2) segm? segment?
            # if segm.ndim == 3:
            self.show_img = input # 2303: torch.Size([1, 1, 256, 256]) , 1ch output  #@@ 16, 60, 64, 64])
            self.show_seg = segm # 2303: torch.Size([1, 3, 256, 256])   #@@ 16, 60, 64, 64]) # 왜 여기는 segm dim [:,*,:,:]에서 *dl 1이 아니라 60?
            # elif segm.ndim==4:
            #     self.show_img = input 
            #     self.show_seg = segm[:,] 

        ############ segm dimention 맞춰주기! ############### 
            # if segm 
            input = torch.cat((input, segm), dim=1) # #@@ [16,61,64,64]
                # torch.Size([16, 60, 64, 64])
                # torch.Size([1, 16, 1, 64, 64]) #$#$#$#$
            # print(f'torch.cat(input,segm).shape: {input.shape}') #@@ [16,61,64,64]
            self.show_concat = input # torch.Size([1, 4, 256, 256]) 
            self.sum_input = torch.sum(input) # 상수항이 나오는 게 맞음! torch.Size([])

        encoding = self.encoder(input)
        self.show_enc = encoding
        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True) # torch.Size([1, 192, 32, 32]) > torch.Size([1, 192, 1, 32]) : H 축에서 평균
        encoding = torch.mean(encoding, dim=3, keepdim=True) #  torch.Size([1, 192, 1, 32]) > torch.Size([1, 192, 1, 1]) : W 축에서 평균

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding) # : 1,4,1,1


        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2) # torch.Size([1, 4, 1])
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2) # torch.Size([1, 4])
        

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178

        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        
        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis. # channel 축을 따라 concatenate함. UNet의 output과 latent space에서 취한 sample을 결합하는...  1x1 conv layers* 
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1 # 16, 60, 64, 64 이런 식으로 실제로 그러함*
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        self.verbose = False # On/Off of print! # 230110_helee

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[0]+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile): # tf.tile과 동일! 지금으로 따지면 torch.tile 인 듯?
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim) # a.dim(): 1 
        repeat_idx = [1] * a.dim() # a가 뭐지? a.dim? # [1, 1, 1] or [1, 1, 1, 1]
        # print(f'a.dim 158 line of p U-Net.py: {a.dim()}') # a: 3 or 4
        # print(f'repeat_idx:{repeat_idx}')
        # print(f'init_dim: {init_dim}')

        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))# .to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_size x latent_dim and feature_map is batch_size x no_channels x H x W.
        So broadcast Z to batch_size x latent_dim x H x W. Behavior is exactly the same as tf.tile (verified)

        Z: B x l_dim  -- Broad Casting --> B x l_dim x H x W
        F_map: B x C x H x W

        z ... B x l_dim x 1 x ?
        z ... B x l_dim x 1 x 1 ..

        z ... B x l_dim x H x 1 ..
        z ... B x l_dim x H x W ..

        -> feature_map, z =concat=> B x (C + l_dim) x H x W 
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]]) # H 만큼 확대 broadcasting
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]]) # W 만큼 확대 broadcasting

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis) # C, l_dim .. 1 axis 채널에서 feature_map 과 z와 concat!
            output = self.layers(feature_map)

            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """
# AxisAlignedConvGaussian , input_channels = 60
    def __init__(self, input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=4, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device) # 62가 되는 문제?.. 이거 그럼  + 1 빼줘야 하나? OO.. # self.input_channels = 60, 60+1=61 # Posterior option여부만 True, False
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

        self.verbose = False # Debugging

    def forward(self, patch, segm, seg, training=True):        
        # segm: target_image,
        # seg: 기존 baseline에서 쓰던 segment mask
        # 여기에 patch: [1, 16, 60, 256, 256] || segm: [1, 16, 1, 256, 256] 넣어주니 OK
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        # segm = segm.unsqueeze(0)
        if training:
            # self.verbose=True
            if self.verbose == True:
                print('################ ProbabilisticUnet  .forward() - 여기 dim은 잘 맞음! 261 line ################')
                print('patch.shape 269', patch.shape ) #@@ 1, 60, 256, 256]
                print('segm.shape 270', segm.shape ) #@@@ # 1, 1, 256, 256  #@@ XXX 1, 60, 256, 256]
            
            ## poseterior forward에서 dimension 2D, 3D 모두 가능하게 처리해줄 것!
            self.posterior_latent_space = self.posterior.forward(patch, segm) #@@ 30 대신 뭘 넣어도 상관 없음

        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch,False) # False는 어디에 해당되나?
        # print('self.unet_features.shape 287! line',self.unet_features.shape ) ##@@@ 1, 32, 256, 256  self.unet_features.shape 287! line torch.Size([1, 32, 256, 256])
        return self.prior_latent_space, self.posterior_latent_space, self.unet_features

###########################################
# Independent(Normal(loc: torch.Size([1, 2]), scale: torch.Size([1, 2])), 1)
# Independent(Normal(loc: torch.Size([1, 2]), scale: torch.Size([1, 2])), 1)
# torch.Size([1, 32, 256, 256]) # 마지막 out channel이 32로 설정 되어있음.
        ## input: 16, 60, 256, 256   ||  
        ### self.prior_latent_space:  [16, 2]
        ### self.posterior_latent_space:  [16, 2]
        ### self.unet_features: [32,16,64,64]
###########################################
    # def forward(self, patch, segm, training=True):
    #     """
    #     Construct prior latent space for patch and run patch through UNet,
    #     in case training is True also construct posterior latent space
    #     """
    #     if training:
    #         self.posterior_latent_space = self.posterior.forward(patch, segm)
    #     self.prior_latent_space = self.prior.forward(patch)
    #     self.unet_features = self.unet.forward(patch,False)
    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            for _ in range(list(random.randint(5,15))):
                z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc 
            for _ in range(random.randint(5,15)):
                z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        # print(f'self.z_prior_sample.shape (317line): {self.z_prior_sample.shape}')
        output = self.fcomb.forward(self.unet_features,z_prior)
        # pdb.set_trace() #$#$#$ helee
        return output

    # def rsample(self, sample_shape=torch.Size()):
    #     shape = self._extended_shape(sample_shape)
    #     eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
    #     return self.loc + eps * self.scale

    # def sample(self, sample_shape=torch.Size()):
    #     shape = self._extended_shape(sample_shape)
    #     with torch.no_grad():
    #         return torch.bernoulli(self.probs.expand(shape))


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc # .loc??? #$#$#$
            # print(f'self.posterior_latent_space.loc 344 line: {self.posterior_latent_space.loc}') # 안에 있는 instance 변수!
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

############################ 1 ###########################
        z_posterior = self.posterior_latent_space.rsample() # posterior 결과 출력 # rsample?
    ## kl-divergence 계산 해줌.
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)

############################ 2 ###########################
        criterion = nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None) # BCE logits loss 계산을 위한 객체 생성        
        reconstruction_loss = criterion(input=self.reconstruction, target=segm) # recon, GT 사이에서 Loss계산!
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

############################ 3 ###########################
        # pdb.set_trace()
        return self.reconstruction, -(self.reconstruction_loss + self.beta * self.kl)




    def elbo_custom(self, segm, seg, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

############################ 1 ###########################
        for _ in range(random.randint(3,6)):
            z_posterior = self.posterior_latent_space.rsample() # posterior 결과 출력 
        print(f'z_posterior:{z_posterior}')
        # print(f'z_posterior:{z_posterior.shape}') # 
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))
        # kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space) 에서 수정

############################ 2 ###########################
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior) # torch.Size([1, 1, 256, 256]), # z_posterior : tensor([[-0.0215,  0.6054]], grad_fn=<AddBackward0>)
        # Here we use the posterior sample sampled above
        ############## 이 부분 loss를 고쳐야 함.
        return self.reconstruction, self.beta * self.kl # -(self.beta * self.kl) -가 왜 붙었지?


############################ 3 ########################### : 바깥으로 빼기!
        # criterion = Ktrans_weighted_L1_and_exponentialLoss(epsilon = 0, alpha = 40, m = 1, n = 1) # nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None) # BCE logits loss 계산을 위한 객체 생성 # nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None) # BCE logits loss 계산을 위한 객체 생성
        # reconstruction_loss = criterion(self.reconstruction, segm, seg )
        # # self.reconstruction: torch.Size([1, 1, 256, 256]) # segm: torch.Size([1, 1, 256, 256])
        # self.reconstruction_loss = torch.sum(reconstruction_loss) # 전체 값 summation!
        # self.mean_reconstruction_loss = torch.mean(reconstruction_loss) # value 하나로 나옴: 전체 pixel 값들 평균*
        # self.reconstruction_loss

########################### 결과 출력 ###############################
        # reconstruction:  [1, 1, 256, 256]
        # self.reconstruction_loss:  tensor(0.0251, grad_fn=<SumBackward0>)
        # self.beta: 10.0
        # self.kl: tensor(1.8848, grad_fn=<MeanBackward0>)
        # criterion: Ktrans_weighted_L1_and_exponentialLoss()
########################### 결과 출력 ###############################

##### 230307 TCN output - input == 1 channel #####
net = ProbabilisticUnet(input_channels=1, num_classes=3, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=1.0)

##### 230307 DCE input == 60 channel #####
# net = ProbabilisticUnet(input_channels=60, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=1.0)


net.to(device)
if verbose == True:
    print( f"shape of input test_image_example[:,:,:,:,:]: {test_image_example[:,:,:,:].shape}")  #@ 1, 256, 256,40, 60 # shape of input test_image_example[:,:,:,:,:]: torch.Size([1, 256, 256, 40, 60]) # =>> channel == 1" torch.Size([1, 1, 256, 256])
    print("shape of label test_label_example[:,:,:, :]", test_label_example[:,:,:,:].shape)
    #@ 1, 256,256,40 # shape of label test_label_example[:,:,:, :] torch.Size([1, 256, 256, 40])
    print("shape of seg test_seg_example[:,:,:, :]", test_seg_example[:,:,:,:].shape)

zslice_index = 9  # 18 slice중 9번
mriview(test_image_example[0,:,:,zslice_index]) # 230307 channel == 1
# mriview(test_image_example[0,:,:,zslice_index,30]) # 230307 channel == 60   # vmin=0, vmax=1.0 # ,cmap='jet',vmin=-0.5,vmax=0.5  # 32 32 60 # cmp : 'bwr'->'jet'

# 1ch output (ktrans) -> 3ch output( ktrans, ve, vp)
print('label_0')
mriview(test_label_example[0,:,:,zslice_index,0],cmap='jet',vmin=0,vmax=0.1)  # 32 32 60
print('label_1')
mriview(test_label_example[0,:,:,zslice_index,1],cmap='jet',vmin=0,vmax=0.1)  # 32 32 60
print('label_2')
mriview(test_label_example[0,:,:,zslice_index,2],cmap='jet',vmin=0,vmax=0.1)  # 32 32 60
print('seg')
mriview(test_seg_example[0,:,:,zslice_index],cmap='jet',vmin=0,vmax=0.1)  # 32 32 60

if test_image_example.dim() == 5:
    input = test_image_example[:,:,:,zslice_index,:] # 230307 channel == 1 # torch.Size([1, 256, 256])
elif test_image_example.dim() == 4:
    input = test_image_example[:,:,:,zslice_index].unsqueeze(-1)  # 230307 channel == 1 # torch.Size([1, 256, 256])
# input = test_image_example[:,:,:,zslice_index,:] # 230307 channel == 60 # torch.Size([1, 256, 256, 60])
# input_permuted = input #.permute((0,1,2)) # 1 60 256 256 # # 230307 channel == 1
input_permuted = input.permute((0,3,1,2)) # 1 60 256 256 # # 230307 channel == 1

if test_label_example.dim() == 5:
    label = test_label_example[:,:,:,zslice_index,:] # 1, 256, 256
elif test_label_example.dim() == 4:
    label = test_label_example[:,:,:,zslice_index].unsqueeze(-1) # 1, 256, 256

if test_seg_example.dim() == 5:
    seg = test_seg_example[:,:,:,zslice_index,:]
elif test_seg_example.dim() == 4:
    seg = test_seg_example[:,:,:,zslice_index].unsqueeze(-1).tile(1,1,1,test_label_example.shape[-1]) # 1, 256, 256, 3 # 동일한 posision 정보이고, domain만 다르므로, 동일 input에 대한 vp, ve, ktrans 값**  # .tile(1,1,1,1)  # 1, 256, 256

label_permuted = label.permute((0,3,1,2))   #  .unsqueeze(1) # 어차피 dimension 하나 없으므로 unsqueeze해주기 1,256,256 -> 1, 1*, 256,256
seg_permuted = seg.permute((0,3,1,2))  # .tile(1,3,1,1) # 어차피 dimension 하나 없으므로 unsqueeze해주기 1,256,256 -> 1, 1*, 256,256

# 아직 sliding_window_inference GPU쓰면 오히려 느린거 해결이 안됨... 일단 진행
print("Sliding window inference example")
print("shape of input", input_permuted.shape)
print("size of input (bytes)",input_permuted.nelement()*input_permuted.element_size())

print(f'test_image_example.shape:{test_image_example.shape}') #$#$#$#$
print(f'test_label_example.shape:{test_label_example.shape}') #$#$#$#$
print(f'seg.shape:{seg.shape}') #$#$#$#$
print('------------------------------------')
print(f'input_permuted.shape:{input_permuted.shape}') #$#$#$#$
print(f'label_permuted.shape:{label_permuted.shape}') #$#$#$#$
print(f'seg_permuted.shape:{seg_permuted.shape}') #$#$#$#$

''' # input dimension!!:
test_image_example.shape:torch.Size([1, 256, 256, 40])
test_label_example.shape:torch.Size([1, 256, 256, 40, 3])
seg.shape:torch.Size([1, 256, 256, 3])
------------------------------------
input_permuted.shape:torch.Size([1, 1, 256, 256])
label_permuted.shape:torch.Size([1, 3, 256, 256])
seg_permuted.shape:torch.Size([1, 3, 256, 256])
'''
################ => 여기 수정했음

# test_label_example[0,:,:,zslice_index]
# print(f'patch.shape lin251:{patch.shape}')
# print(f'segm.shape: {segm.shape}')

######################### # 1**. Total Key Forward #############################
############# input_permuted: 1,1,60,256,256  |  segm: 1,1, 3, 256,256      |||
prior_latent_space, posterior_latent_space, unet_features = net(input_permuted, segm=label_permuted, seg=seg_permuted)  # .to(device)  .to(device) 삭제! # sliding window에 비해 1분에서 20초로 줄긴했는데
'''
prior_latent_space: Independent(Normal(loc: torch.Size([1, 2]), scale: torch.Size([1, 2])), 1)
posterior_latent_space: Independent(Normal(loc: torch.Size([1, 2]), scale: torch.Size([1, 2])), 1)
unet_features: torch.Size([1, 32, 256, 256])
'''

############################## Original VS 새로 개발한 것 input 자체가 다름!
# test_output = net.elbo(label_permuted)  # test_label_example
test_output, KL_loss = net.elbo_custom(label_permuted, seg_permuted) # test_output: [1, 1, 256, 256]
# test_output: torch.Size([1, 3, 256, 256])
# KL_loss: torch.Size([])


######################### # 1**. Total Key Forward #############################
criterion = Ktrans_weighted_L1_and_exponentialLoss(epsilon = 0, alpha = 40, m = 1, n = 1) # nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None) # BCE logits loss 계산을 위한 객체 생성 # nn.BCEWithLogitsLoss(size_average = False, reduce=False, reduction=None) # BCE logits loss 계산을 위한 객체 생성
reconstruction_loss = criterion(test_output, label_permuted, seg_permuted ) # pred, label, seg_mask 순서! 
# test_output: torch.Size([1, 3, 256, 256])
# label_permuted: torch.Size([1, 3, 256, 256])
# seg_permuted: torch.Size([1, 3, 256, 256])

# self.reconstruction: torch.Size([1, 1, 256, 256]) # segm: torch.Size([1, 1, 256, 256])
reconstruction_loss = torch.sum(reconstruction_loss) # 전체 값 summation!
mean_reconstruction_loss = torch.mean(reconstruction_loss) # value 하나로 나옴: 전체 pixel 값들 평균*

# 여기걸림!
print(f'################ for sample printing! 256,256으로 일부러 맞춰준 것 \n\n\ntest_output[0][0,...].detach() : {test_output[0,0,...].detach().shape}') # [256,256]
# 단점이, 여전히 메모리 43g먹음.
print('Ktrans:') 
mriview(test_output[0,0,...].detach()) # <- [0] 없애줘도 되게 위에서 수정함! # test_output이 tuple로 나와서 test_output[0] 해줘야 함
print('Vp:') 
mriview(test_output[0,1,...].detach())
print('Ve:') 
mriview(test_output[0,2,...].detach())
################ => tuple이 결과라서 위에 [0] 추가 수정했음
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

############################## 4 ################################

from PIL import Image
import matplotlib.pyplot as plt
import copy

def nonvisualize_ktrans_compare_new_label(y,y_hat,vmax=0.1): # y,y_hat이 들어오면.. axis=1방향으로.. concat
    plt.figure()

    if y.ndim == 3:
        y = y.permute((2,0,1))
        y = y.reshape(y.shape[0]*y.shape[1],-1)
    if y_hat.ndim == 3:
        y_hat = y_hat.permute((2,0,1))
        y_hat = y_hat.reshape(y_hat.shape[0]*y_hat.shape[1],-1)        


    concat_image = np.concatenate((y,y_hat),axis=0) # torch.size.. ([1792, 256])  = ([512,256] + [1280, 256]) 
    random_num = str(np.random.randint(99999999))
    plt.imsave(random_num+'temp.png',concat_image,cmap='jet',vmin=0,vmax=vmax,format="png")
    figure_np = np.array(Image.open(random_num+'temp.png')) # 1835008 ??  (1792, 256, 4)

    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2) #$#$ (3, 256, 1792)  4->3으로 하나 줄어듦 :3] slicing함. ->  # [3 X Y] 로 바꿈. Tensorboard용    
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2) # (3, 1792, 256)  # [3 X Y] 로 바꿈. Tensorboard용  (좌우flip안되게) -> 언제 flip 되는 편인가? 
    os.remove(random_num+'temp.png')
    # plt.cla()

    return figure_np_3channel


def nonvisualize_ktrans_compare_new(y,y_hat,vmax=0.1): # y,y_hat이 들어오면.. axis=1방향으로.. concat*
    plt.figure()
    concat_image = np.concatenate((y,y_hat),axis=0) # 0으로 붙이는 게 맞을 듯! # torch.size.. ([1792, 256])  = ([512,256] + [1280, 256]) 
    random_num = str(np.random.randint(99999999))
    plt.imsave(random_num+'temp.png',concat_image,cmap='jet',vmin=0,vmax=vmax,format="png")
    figure_np = np.array(Image.open(random_num+'temp.png')) # 1835008 ??  (1792, 256, 4)

    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2) #$#$ (3, 256, 1792)  4->3으로 하나 줄어듦 :3] slicing함. ->  # [3 X Y] 로 바꿈. Tensorboard용    
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2) # (3, 1792, 256)  # [3 X Y] 로 바꿈. Tensorboard용  (좌우flip안되게) -> 언제 flip 되는 편인가? # 
    os.remove(random_num+'temp.png')
    # plt.cla()

    return figure_np_3channel



def visualize_ktrans_patch_30_uncertainty_3output(x,y,y_hat, seg, uncertainty_map, vmax=0.1):
    fig, axes = plt.subplots(3, 4, figsize=(15,10))
    # 1. input
    verbose = False
    if verbose ==True:
        print(f'x:{x.shape}')
        print(f'y:{y.shape}')
        print(f'y_hat:{y_hat.shape}')
    #     print(f'seg:{seg.shape}')

    '''x:torch.Size([256, 256])
    y:torch.Size([3, 256, 256])
    y_hat:torch.Size([3, 256, 256])
    seg:torch.Size([256, 256])'''
    ## input들어오는 것에 따라 약간 수정 필요! 3#

    im1 = axes[0,0].imshow(x,cmap = 'jet')
    axes[0,0].set_title("Input") # 지금은 relaxivity map이 아니라 DCE input 이므로 cmap bwr이 아니라 jet로 똑같이 맞춰주기! 
    plt.colorbar(im1,ax=axes[0,0])
    # 2. label 1
    im2 = axes[0,1].imshow(y[0],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im2,ax=axes[0,1])
    axes[0,1].set_title(" Label 1")
    # 3. label 2
    im3 = axes[0,2].imshow(y[1],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im3,ax=axes[0,2])
    axes[0,2].set_title(" Label 2")
    # 4. label 3
    im4 = axes[0,3].imshow(y[2],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im4,ax=axes[0,3])
    axes[0,3].set_title(" Label 3")

    # 5. seg
    im5 = axes[1,0].imshow(seg,cmap = 'jet', vmin=0, vmax=vmax)
    axes[1,0].set_title("Tumor_segment")
    plt.colorbar(im5,ax=axes[1,0])

    # 6. output 1
    im6 = axes[1,1].imshow(y_hat[0],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im6,ax=axes[1,1])
    axes[1,1].set_title(" Output 1")
    # 7. output 2
    im7 = axes[1,2].imshow(y_hat[1],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im7,ax=axes[1,2])
    axes[1,2].set_title(" Output 2")
    # 8. output 3
    im8 = axes[1,3].imshow(y_hat[2],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im8,ax=axes[1,3])
    axes[1,3].set_title(" Output 3")

    # 9. Uncertainty Map
    im9 = axes[2,0].imshow(y[0],cmap = 'jet', vmin=0)
    axes[2,0].set_title("UncertaintyMap(..Ktrans_max)")
    plt.colorbar(im9,ax=axes[2,0])

    # 10. output 1
    im10 = axes[2,1].imshow(y_hat[0],cmap = 'jet',vmin=0)
    plt.colorbar(im10,ax=axes[2,1])
    axes[2,1].set_title("Output 1 max")
    # 11. output 2
    im11 = axes[2,2].imshow(y_hat[1],cmap = 'jet',vmin=0)
    plt.colorbar(im11,ax=axes[2,2])
    axes[2,2].set_title("Output 2 max")
    # 12. output 3
    im12 = axes[2,3].imshow(y_hat[2],cmap = 'jet',vmin=0)
    plt.colorbar(im12,ax=axes[2,3])
    axes[2,3].set_title("Output 3 max")

    # criterion = Custom_L1()
    random_num = str(np.random.randint(99999999))
    plt.savefig(random_num+'temp.png',bbox_inches='tight')
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    os.remove(random_num+'temp.png')
    plt.show()
    return figure_np_3channel

############################## 5 ################################

class DSC_to_Ktrans_Patch_Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class,patch_size,sw_batch_size,overlap,test_vis_loader_valid,visualize_slice_list_valid ):
        super().__init__()
        self.lr = learning_rate
        self.net = net # 그냥 대입해준 것
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.log_every_epoch = [] 
        self.patch_size = patch_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.test_vis_loader_valid = test_vis_loader_valid
        self.visualize_slice_list_valid = visualize_slice_list_valid

        self.verbose = False

    def configure_optimizers(self):
        #optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        # optimizer = self.optimizer_class(self.parameters(), lr=(self.lr or self.learning_rate))

        ########################### 최적화 필요1 ###############################
        # optimizer = self.optimizer_class(net.parameters(), lr=1e-4, weight_decay=0)
        optimizer = self.optimizer_class(self.parameters(), lr=(self.lr or self.learning_rate))
        ########################### 최적화 필요1 ###############################
        # optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
        # Auto lr 위해 수정 https://pytorch-lightning.readthedocs.io/en/1.4.5/advanced/lr_finder.html
        return optimizer
    
    def forward(self,batch): # train loop의 forward
    # 1. 받아온 배치에서 patches_image, patches_label, patches_seg 추출
        patches_image, patches_label, patches_seg = batch 
        patch_label = patches_label 
        patch_seg = patches_seg       
        prior_latent_space, posterior_latent_space, unet_features = self.net(patches_image[0,...], patch_label[0,...], patch_seg[0,:,0,...]) 

        patches_forward = self.net.elbo_custom(patch_label[0,...], patch_seg[0,...]) # [16, 60, 64, 64]
        # patches_forward[0]: 1) self.reconstruction -> torch.Size([16, 3, 64, 64]), 
        # patches_forward[1]: 2) self.beta * self.kl -> torch.Size([])
        return patches_forward, patches_label[0,...], patches_seg[0,...]   # 셋 다 256 6

    
    def training_step(self, batch, batch_idx):
        #$#$ loss수정!
        weight_elbo_loss = 1.0 # 1 # 100
        # weight_kl_loss =  

        y_hat, y, y_seg = self(batch) # forward 함수 적용됨. # 여기서 probUNet은 (y_hat_feature, kl_Loss) tuple로 구성됨
        y_hat, kl_Loss = y_hat # feature, loss값 분리해주기*
        
        # y_hat : torch.Size([16, 3, 64, 64])
        # y : torch.Size([16, 3, 64, 64])
        # y_seg : torch.Size([16, 1, 64, 64])

        weighted_recon_loss = self.criterion(y_hat, y, y_seg)
        loss = weighted_recon_loss + weight_elbo_loss * kl_Loss       # print(y_hat) # 1) reconmap: torch.Size([16, 1, 64, 64]), 2) kl_Loss: 값 1개
 
        self.log('train_weighted_recon_loss_batch', weighted_recon_loss, on_epoch=True, prog_bar=True)
        self.log('train_kl_loss', kl_Loss, on_epoch= True, prog_bar=True)
        self.log('train_loss_batch', loss, on_epoch=True, prog_bar=True)
        return loss        
    
    def validation_step(self, batch, batch_idx):    
        y_hat, y, y_seg = self(batch) # forward 함수 적용됨. # 여기서 probUNet은 (y_hat_feature, kl_Loss) tuple로 구성됨
        y_hat, kl_Loss = y_hat # feature, loss값 분리해주기*
        
        weighted_recon_loss = self.criterion(y_hat, y, y_seg)
        loss = weighted_recon_loss+kl_Loss

        self.log('val_weighted_recon_loss_batch', weighted_recon_loss, on_epoch=True, prog_bar=True)
        self.log('val_kl_loss', kl_Loss, on_epoch= True, prog_bar=True)
        self.log('val_loss_batch', loss, on_epoch=True, prog_bar=True)

        loss2 = dice_01(y_hat,y)
        loss3 = dice_02(y_hat,y)

        self.log('val_loss_batch', loss, on_epoch=True, prog_bar=True)
        return {'loss':loss,'loss2':loss2,'loss3':loss3}


    def validation_epoch_end(self,validation_step_outputs): #https://learnopencv.com/tensorboard-with-pytorch-lightning/
        visual_verbose = False
        # print("val_outputs",validation_step_outputs)
        
        avg_loss2 = torch.stack([x['loss2'] for x in validation_step_outputs]).mean()   
        avg_loss3 = torch.stack([x['loss3'] for x in validation_step_outputs]).mean()   
        
        avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
        print("finished epoch:",self.current_epoch)
        self.log("average_validation_loss",avg_loss)  # early stopping callback 위해

        print("epoch", self.current_epoch,"average valid loss",format_4f(avg_loss.item())) 
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
        if visual_verbose == True:
            plt.title("loss of last 5 epochs")
            plt.plot(np.arange(len(loss_list))[-5:], loss_list[-5:])
            plt.show()

        generator = iter(self.test_vis_loader_valid)
        self.net.eval()
        with torch.no_grad():
            for i in range(len(self.test_vis_loader_valid)):     # object 별 iteration   
                test_vis_image,test_vis_label,test_vis_seg, path = next(generator)      
                for slice in self.visualize_slice_list_valid[i]: # slice 별 iteration
                    if test_vis_image.dim() == 5:
                        input = test_vis_image[:,:,:,slice,:] # 1 256 256 60
                    elif test_vis_image.dim() == 4: 
                        input = test_vis_image[:,:,:,slice].unsqueeze(-1) # 1 256 256 60
                    else:
                        print(f'line 174 error!: test_vis_img.dim is not both 4 and 5 but {test_vis_image.dim()}')
                    
                    if test_vis_label.dim() == 5:
                        label = test_vis_label[:,:,:,slice,:] # 1 256 256 60
                    elif test_vis_label.dim() == 4: 
                        label = test_vis_label[:,:,:,slice].unsqueeze(-1) # 1 256 256 60
                    
                    if test_vis_seg.dim() == 5:
                        seg = test_vis_seg[:,:,:,slice,:] # 1 256 256 60
                    elif test_vis_seg.dim() == 4: 
                        seg = test_vis_seg[:,:,:,slice].unsqueeze(-1) # 1 256 256 60
                    
                    input_permuted = input.permute((0,3,1,2))
                    label_permuted = label.permute((0,3,1,2))   #  .unsqueeze(1) # 어차피 dimension 하나 없으므로 unsqueeze해주기 1,256,256 -> 1, 1*, 256,256
                    seg_permuted = seg.permute((0,3,1,2))  # .tile(1,3,1,1) # 어차피 dimension 하나 없으므로 unsqueeze해주기 1,256,256 -> 1, 1*, 256,256

                    prior, posterior, unet_feature = self.net(input_permuted, label_permuted, seg_permuted )# .detach()
                    output, kl_Loss = self.net.elbo_custom(label_permuted, seg_permuted)
                    # DSC_to_Ktrans_Patch_Model
                    # torch.Size([1, 60, 256, 256])
                    # torch.Size([1, 1, 256, 256])
                    # torch.Size([1, 1, 256, 256])
                    #output = sliding_window_inference(input_permuted,patch_size,sw_batch_size,net,overlap)s
                    # print('up - 171 line checke!')#$#$#$
                    print(f'kl_Loss: {kl_Loss}')
######################### Dim 설정! #############################
                    input_img = input_permuted[0,0,:,:].detach() # t=30일때 input visualize로 바꿈. # [256, 256]
                    output_img = output[0,...].detach() # [3, 256, 256]
                    label_img = label_permuted [0,:,:,:].detach() # [3, 256, 256]
                    seg_img = seg_permuted[0,0,:,:].detach() # [256, 256]
######################### Dim 설정! #############################
            ################# helee2320??? ####################
                    # print(i)
                    # 에러남 -> 문제 있음 # if epoch % 5 == 0:

                    # if visual_verbose == True:
                    if self.current_epoch % 5 == 0:
                        vis_image = visualize_ktrans_patch_30_uncertainty_3output(input_img, label_img, output_img, seg=seg_img, uncertainty_map=seg_img)
        self.net.train()    
        return epoch_dictionary


model = DSC_to_Ktrans_Patch_Model(
    net=net,
    criterion=criterion, 
    learning_rate=learning_rate,
    optimizer_class=optimizer_class,
    patch_size = patch_size,
    sw_batch_size = sw_batch_size,
    overlap = overlap,
    test_vis_loader_valid = test_vis_loader_valid,
    visualize_slice_list_valid = visualize_slice_list_valid
)

logger = TensorBoardLogger(tensorboard_dir_name, name=experiment_name)
checkpoint_callback = ModelCheckpoint(save_top_k=10,monitor="average_validation_loss",mode="min") # early stopping 5 -> 8 로 바꿈*
early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
     monitor="average_validation_loss", patience=patience
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


model_best = DSC_to_Ktrans_Patch_Model.load_from_checkpoint(best_epoch_path,
    net=net,
    criterion=criterion, 
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
output_img_list = []
label_img_list = []
seg_img_list = []
dice01_list = []
dice02_list = []
generator = iter(test_full_vis_loader)
net.eval()

#num_subjects = len(test_full_vis_loader)
visualize_subject_list = [7,8,10]

#$#$#$helee 230426

with torch.no_grad():
    for i in tqdm(range(len(test_dataset))):       
        test_vis_image,test_vis_label,test_vis_seg, path = next(generator) 
        if i in visualize_subject_list:     
            for slice in range(test_vis_image.shape[-2]):
                if test_vis_image.dim() == 5:
                    input = test_vis_image[:,:,:,slice,:] # 1 256 256 60
                elif test_vis_image.dim() == 4: 
                    input = test_vis_image[:,:,:,slice].unsqueeze(-1) # 1 256 256 60
                input_permuted = input.permute((0,3,1,2))
                # output = net(input_permuted)
                # print( model_best.net(input_permuted, label_permuted, seg_permuted ) )
                prior, posterior, unet_feature = model_best.net(input_permuted, label_permuted, seg_permuted )# .detach()
                output, kl_Loss = net.elbo_custom(label_permuted, seg_permuted)
                # DSC_to_Ktrans_Patch_Model
                # torch.Size([1, 60, 256, 256])
                # torch.Size([1, 1, 256, 256])
                # torch.Size([1, 1, 256, 256])
                # pdb.set_trace()
                #output = sliding_window_inference(input_permuted,patch_size,sw_batch_size,net,overlap)s
                # print('up - 171 line checke!')#$#$#$
                #print(f'kl_Loss: {kl_Loss}')

                #output = sliding_window_inference(input_permuted,patch_size,sw_batch_size,net,overlap)
                output_img = output[0,:,...].detach() # output이 channel인데 왜 [0,0,...]으로 했을까?....
                label_img = test_vis_label[0,:,:,slice].detach()
                seg_img = test_vis_seg[0,:,:,slice].detach()
                output_img_list.append(output_img)
                label_img_list.append(label_img)
                seg_img_list.append(seg_img)

                label_img = label_img.permute((2,0,1))

                # if (i % 5 == 0):
                # if (i % 5 == 0) or (): # epoch이 5주기로 돌거나 (1) loss 값이 최소로 업데이트 됐을 때:
                vis_image = nonvisualize_ktrans_compare_new_label(label_img,output_img)

                image_list.append(vis_image)
                if torch.sum(seg_img)>0:
                    dice01_list.append(dice_01(label_img,output_img))
                    dice02_list.append(dice_02(label_img,output_img))


def show_six_slices(a,b,c,d,e,f):
    image_1st_row = np.concatenate((a,b,c),axis=1)
    image_2nd_row = np.concatenate((d,e,f),axis=1)
    image_grid = np.concatenate((image_1st_row,image_2nd_row),axis=2)
    plt.imshow(np.swapaxes(image_grid,0,2))
    plt.gcf().set_dpi(500)
    plt.axis('off')
    plt.show()
    

""" for i in range(int(len(image_list)/6)):
    show_six_slices(image_list[6*i],image_list[6*i+1],image_list[6*i+2],image_list[6*i+3],image_list[6*i+4],image_list[6*i+5])
 """
# for i in [1,4,7]:
#     show_six_slices(image_list[6*i],image_list[6*i+1],image_list[6*i+2],image_list[6*i+3],image_list[6*i+4],image_list[6*i+5])


output_img_stack = torch.stack(output_img_list)
label_img_stack = torch.stack(label_img_list)
seg_img_stack = torch.stack(seg_img_list)

output_img_stack = torch.reshape(torch.stack(output_img_list),(-1,))
label_img_stack = torch.reshape(torch.stack(label_img_list),(-1,))
seg_img_stack = torch.reshape(torch.stack(seg_img_list),(-1,))

print('1813 line!')
# load_from_checkpoint
############################## 6 ################################
############################## 6 ################################

# folder_path = 'results_3output_230412_align_10_try001'
# folder_name_dict
def show_and_save_img_3output_slice(subject_number='1', objects_num=1,slice_num=1, ch=1, label=1, seg=1, mean=1, std=1, std_per_mean=1, std_per_mean_seg=1, show_flag=False, label_name='KtransVpVe'):
    global folder_path
    # global folder_name_dict # subject_number
    title_list = [f'{label_name}_label', f'{label_name}_seg', f'{label_name}_std/mean_seg' , f'{label_name}_mean', f'{label_name}_std', f'{label_name}_std/mean']
    # print(title_list)
    w = 256
    h = 256
    columns = 6
    img_list = [label, seg, std_per_mean_seg, mean, std, std_per_mean]

    rows = 1

    fig = plt.figure(figsize=(6*columns, 6*rows))
    for i in range(1, columns*rows +1):
        # img = np.random.randint(2, size=(h,w))
        
        img = img_list[(i-1)%6] # [(i-1)//5s]

        fig.add_subplot(rows, columns, i)
        if (i-1)//6==0:
            plt.title(title_list[i-1], fontsize=20)
        # elif (i-1) % 5 == 0:
        #     ax = plt.gca()
        #     ax.get_yaxis().set_visible(True)
        if i in [3,6]: #(i % 6) != 0: 
            plt.imshow(img, cmap='jet', vmin = 0., vmax=1.)            
        else:
            plt.imshow(img, cmap='jet', vmin = 0., vmax=0.1) # std/mean에서만 vmax 조절 필요!
        plt.axis('off')
    ####### save name , file path 지정 #######
    # if objects_num == str():
    
    visuaialization_slice_file_path = f'./{folder_path}/visualization/{subject_number}_{str(objects_num)}_{str(slice_num)}_{str(ch)}_{label_name}_TCN-PUnet.png'
    if os.path.exists( '/'.join((visuaialization_slice_file_path).split('/')[:-1])) == False:
        os.makedirs( '/'.join((visuaialization_slice_file_path).split('/')[:-1])) # get folder path W/O file name

    plt.savefig(visuaialization_slice_file_path, bbox_inches='tight',pad_inches = 0)
    if show_flag == True:
        plt.show()
    plt.close()
img = np.random.randint(2, size=(256,256))

show_and_save_img_3output_slice(objects_num=1, slice_num=1, ch=1, label= copy.deepcopy(img), seg= copy.deepcopy(img), mean= copy.deepcopy(img), std= copy.deepcopy(img), std_per_mean= copy.deepcopy(img), std_per_mean_seg = copy.deepcopy(img), show_flag=True, label_name='ktrans')

################################################################
######### 20230425 Inference!!! ############
#################### exp folder_path: results_3output_230320
# folder_path = 'results_3output_230415_align_4_try3'
####################
import time
import matplotlib.pyplot as plt
import nibabel as nib
# value_per_pixel_aligned_list = []
# # 1. label_img - segmented
# output_segmented = label_img * seg_img
# seg_img_flat = seg_img.view(-1)
# for i, v in enumerate(output_segmented):
#     if seg_img_flat[i] != 0:
#         value_per_pixel_aligned_list += [v] # 0은 제외하고, 0이 아닌 값들만 체크!
# X_axis = list(range(torch.count_nonzero(output_segmented))) # seg_img_flat[i]
# Y_axis = value_per_pixel_aligned_list
# print(f'X_axis:{len(X_axis)} || Y_axis:{len(Y_axis)}')

# plt.bar(X_axis, Y_axis) #, color='y')
# plt.bar(x, values, color='dodgerblue')
# plt.bar(x, values, color='C2')
# plt.bar(x, values, color='#e35f62')
# plt.xticks(x, years)
##### 230203 plot & savefig 성공!! #######

# vmax = 1.0로 설정함
label_name_list = ['Ktrans', 'Vp', 'Ve']
           
test_cache = DiffusionCacheDataset(test_files)
print("finished caching test files")
test_dataset = Dataset(data=test_cache, transform=test_transform())
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
# #test_image_example.shape : torch.Size([1, 256, 256, 18, 60])
# #test_label_example.shape : torch.Size([1, 256, 256, 18])
# generator = iter(test_loader)
# test_image_example, test_label_example, test_seg_example = next(generator)

# test_dataset_visualize_valid = test_dataset[visualize_list_valid]
# test_vis_loader_valid = DataLoader(test_dataset_visualize_valid, batch_size=batch_size)

test_full_vis_loader = DataLoader(test_dataset, batch_size=batch_size)

image_list = []
output_img_list = []
output_img_iter_list = []
label_img_list = []
seg_img_list = []
dice01_list = []
dice02_list = []
generator = iter(test_full_vis_loader)
net.eval()

#num_subjects = len(test_full_vis_loader)
visualize_subject_list = list(set(range(118))) #  [7,8,10,   11,12,13,14]
print(f'visualize_subject_list: {visualize_subject_list}')
########### visualize 기준은 ( ch, H, W ) 순서의 정렬을 기본으로 한다! 헷갈림 방지!! ###############
########### visualize 기준은 ( ch, H, W ) 순서의 정렬을 기본으로 한다! 헷갈림 방지!! ###############
########### visualize 기준은 ( ch, H, W ) 순서의 정렬을 기본으로 한다! 헷갈림 방지!! ###############
# start = time.time()
# end = time.time()
with torch.no_grad():
    for i in tqdm(visualize_subject_list): 

        test_vis_image,test_vis_label,test_vis_seg, path = next(generator) 
        if i == 77: continue      
        segMax, iMax, sliceMax, file_str_segMax_list = -1, -1, -1, [] # seg가 아무것도 없으면 -1로 설정!
                
        subject_number = path[0].split('/')[-1]
        print(f'Make {subject_number} ...')

##################### 0. Ref load! - 위 (여기)로 옮김 #####################
        refvol = nib.load(f'{base_dir}/test/{subject_number}/g_ktrans.nii.gz') # folder_name_dict[i]
        # Numpy array to Nifti
        save_folder_path = f'./{folder_path}/{subject_number}' #path[0]+'/'+forward_filename
        if os.path.exists(save_folder_path)==False :
            os.makedirs(save_folder_path)

        save_path_seg = f'{save_folder_path}/seg.nii.gz' # test_vis_seg[0]
        seg_nft = nib.Nifti1Image(test_vis_seg[0] ,refvol.affine,refvol.header)
        nib.save(seg_nft, save_path_seg)

        # Object 단위로 저장 하기* 
        if i in visualize_subject_list:
            ################ nii.gz buffer 선언! #####################
    ### list 선언해서 올린 뒤, 나중에 stack
            # nii_gz_list = []
            mean_nii_gz_list, std_per_mean_nii_gz_list, std_per_mean_seg_nii_gz_list = [], [], []
            for slice in range(test_vis_image.shape[-2]): # range(0, 40)
                input = test_vis_image[:,:,:,slice,:] # 1 256 256 60 
                input_permuted = input.permute((0,3,1,2))

                # label_img = test_vis_label[:,:,:,slice,:].detach()
                # # label_permuted = label_img.permute(2,0,1)
                # seg_img = test_vis_seg[:,:,:,slice].detach() 

                if test_vis_label.dim() == 5:
                    label_img = test_vis_label[:,:,:,slice,:] # 1 256 256 60
                elif test_vis_label.dim() == 4: 
                    label_img = test_vis_label[:,:,:,slice].unsqueeze(-1) # 1 256 256 60
                
                if test_vis_seg.dim() == 5:
                    seg_img = test_vis_seg[:,:,:,slice,:] # 1 256 256 60
                elif test_vis_seg.dim() == 4: 
                    seg_img = test_vis_seg[:,:,:,slice].unsqueeze(-1) # 1 256 256 60
                # print(f'input:{input.shape}') 
                # print(f'label_img:{label_img.shape}')
                # print(f'seg_img:{seg_img.shape}')
                if seg_img.dim() == label_img.dim(): # concat을 해주기 위해서 seg_img, label_img 의 dim이 모두 딱 같아야 함!
                    pass
                elif seg_img.dim() == (label_img.dim() - 1):
                    seg_img = seg_img.unsqueeze(0).detach() # 채널 ch는 0번 axis에 추가!!
                elif seg_img.dim() == (label_img.dim() + 1):
                    label_img = label_img.unsqueeze(0).detach()
                ## 완전알고리즘인가? 230309                    
                seg_permuted = seg_img.permute((0,3,1,2))  # .tile(1,3,1,1) # 어차피 dimension 하나 없으므로 unsqueeze해주기 1,256,256 -> 1, 1*, 256,256
                
                if label_img.dim() == 4:
                    label_permuted = label_img.permute((0,3,1,2))# ((0,3,1,2))   #  .unsqueeze(1) # 어차피 dimension 하나 없으므로 unsqueeze해주기 1,256,256 -> 1, 1*, 256,256
                elif label_img.dim() == 3:
                    label_permuted = label_img.unsqueeze(-1)
                    label_permuted = label_img.permute((0,3,1,2)) #((0,3,1,2))   #  .unsqueeze(1) # 어차피 dimension 하나 없으므로 unsqueeze해주기 1,256,256 -> 1, 1*, 256,256

                ''' 230320 : input: TCN Output
                input_permuted: torch.Size([1, 1, 256, 256])
                label_permuted: torch.Size([1, 3, 256, 256])
                seg_permuted: torch.Size([1, 1, 256, 256])
                '''
##################################################
#                 print( '''Imaginative results: \
#                 input:torch.Size([1, 256, 256, 1])\
#                 label_img:torch.Size([1, 256, 256, 3])\
#                 seg_img:torch.Size([1, 256, 256, 1])\
# \
#                 input_permuted: torch.Size([1, 1, 256, 256])\
#                 label_permuted: torch.Size([1, 3, 256, 256])\
#                 seg_permuted: torch.Size([1, 1, 256, 256])''')
##################################################
                seg_img_list.append(seg_permuted[0])
                output_img_iter_list_orgin = [] # for 문 돌때마다 한 번 씩 선언해줌! # 
                # output_img_iter_concat = torch.zeros((1,N+1+1,256,256)) # N+1+1:( N번 반복+tumor seg+label, H:256, W:256 )
                # XXX output_img_list.append( output_img_iter_list ) # [[0],[1],[2],[3],[4]]
                # output_img_iter_list # ( 10, 3, 256, 256 )

##########################################################################
###################### 1. N-iter network forward!!! ######################
##########################################################################
                # N_list = list(range(N_iter_infer+100))
                # N_infer_sample_list = random.sample(N_list, N_iter_infer)
                # random.randint() # 100개 중에 4개 뽑으면 됨.. inference저장하기
                for n in range(N_iter_infer):
                    # if n not in N_iter_infer_list:
                    #     continue
                    # output = net(input_permuted)
                    # unet_features: torch.Size([1, 32, 256, 256])
                    prior, posterior, unet_feature = net(input_permuted, label_permuted, seg_permuted) # prior, posterior: Independent(Normal(loc: torch.Size([1, 2]), scale: torch.Size([1, 2])), 1) # .detach()
                    output, kl_Loss = net.elbo_custom(label_permuted, seg_permuted) # output: torch.Size([1, 1, 256, 256]) || kl_Loss: tensor(0.1196)
                    output_img = output[0,...].detach() # inference라 학습에 관여 안하므로 detach 해버리기! # torch.Size([3, 256, 256])
                    # if n == 0:
                    #     output_img_iter_concat = output  
                    # else:
                    #     output_img_iter_concat = torch.cat( (output_img_iter_concat, output) )
                # N 번 inference한 것을 그대로 저장해두자!
                    output_img_iter_list_orgin.append(output_img) # list에 더해주나 concat해주나 똑같음!
                ##### output_img: # torch.Size([3, 256, 256])
########################### 2. network forwarding outputs save!!! ##############################
                # XXX output_img_list.append( output_img_iter_list ) # [[0],[1],[2],[3],[4]]
                                #######################################
                # for n in range(N_iter_infer):
                if not os.path.exists(f'../outputs/results_3output_240830_align_forward/{subject_number}/subject_outputs/'):
                    os.makedirs(f'../outputs/results_3output_240830_align_forward/{subject_number}/subject_outputs/')
                save_path_output = f'../outputs/results_3output_240830_align_forward/{subject_number}/subject_outputs/outputs_{subject_number}_{slice}_{n}.nii.gz' # ktrans

                # output_N_infer = nib.Nifti1Image((output_img_iter_list_orgin.numpy()).transpose((1,2,0)),refvol.affine,refvol.header) # 2, 256, 256
                if type(output_img_iter_list_orgin) == list:
                    output_img_iter_list_orgin = output_img_iter_list_orgin[0]
                output_N_infer = nib.Nifti1Image((output_img_iter_list_orgin.detach().numpy()).transpose((1,2,0)),refvol.affine,refvol.header) # 2, 256, 256 # deatach 없음!
                
                nib.save(output_N_infer, save_path_output) # numpy array의 transpose 방식이었음!
                print(f'{save_path_output}')

                ################# Dimension ######################
                # output_img_iter_list_orgin # ( N:10, 3, 256, 256 )
    ###########################  3. network forwarding outputs sampling ###########################
                N_list_temp = N_list.copy()
                ## helee control 1 line
                for key, N_iter_infer_list in enumerate(N_iter_infer_list_list): # , list(range(4)), list(range(4))  #  list(range(9)), list(range(9))
                
                    save_folder_path = f'./{folder_path}_{hash_dict[key]}/{subject_number}' #path[0]+'/'+forward_filename
                    if os.path.exists(save_folder_path)==False :
                        os.makedirs(save_folder_path)

                    N_infer_sample_list_list = []
                    N_infer_sample_list = random.sample(N_list_temp, len(N_iter_infer_list)) #                     
                    N_infer_sample_list_list += [N_infer_sample_list]
                    N_list_temp = list( set(N_list_temp) - set(N_infer_sample_list) )  # 이미 뽑은 샘플은 전체 list에서 제외하기
                    ############# inference** 이후 mean, std 등 계산 ############
                    # random.randint() # 100개 중에 4개 뽑으면 됨.. inference저장하기
                    output_img_iter_list = [] # 초기화
                    for N_s in N_infer_sample_list: # [3,5,6,7,8]
                    # for N_iter_infer_num in N_iter_infer_list: # loader 로 작성 필요
                    #해당 (1) subject 에서 (2) slice 의 (3) N_s 번째 결과 뽑기!
                    # /mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/221226/results_3output_230501_align_forward/{subject_number}/subject_outputs/outputs_{subject_number}_{slice}_{n}.nii.gz
                        sample_path = f'../outputs/results_3output_240830_align_forward/{subject_number}/subject_outputs/outputs_{subject_number}_{slice}_{N_s}.nii.gz'
                        
                        sample_output = torch.Tensor(np.array(nib.load(sample_path).dataobj).transpose(2, 0, 1)) # 256,256,3 -> 3,256,256

                        now = datetime.now()

                        with open(f'../outputs/results_3output_240830_align_forward/{subject_number}/subject_outputs/results_inference_{hash_dict[key]}.txt', 'a') as f: 
                            f.write(f"[{now.month}{now.day}_{now.hour}{now.minute}_{save_path_output}] {N_infer_sample_list}\n")
                        f.close()

                ############################# output dimension ################################
                        output_img_iter_list += [sample_output] # [N, tensor(3,256,256)]

                        output_img_iter_tensor = torch.cat(tuple(output_img_iter_list)) # torch.Size([30, 256, 256])   # [ 10, 3, 256,256 ] > .mean(dim=0) > [3, 256, 256] 
                        output_img_iter_stack = torch.stack(tuple(output_img_iter_list)) #   torch.Size([10, 3, 256, 256])
                        
                        output_mean_map = torch.mean(output_img_iter_stack, dim=0) # 0320: (10,3,256,256) || # output_img_iter_stack: ([5, 256, 256]) | output_mean_map: (256,256)
                        output_std_map = torch.std(output_img_iter_stack, dim=0) # 0320: (10,3,256,256)  || # output_std_map ([5, 256, 256])

                        output_std_per_mean_map = output_std_map / output_mean_map
                        output_std_per_mean_map_thres = output_std_map.where(output_mean_map>0.01, torch.zeros(output_mean_map.shape)) / (output_mean_map)
                        output_std_per_mean_map_thres_seg = output_std_per_mean_map_thres*torch.tile(seg_permuted[0], (3,1,1)) # 3,256,256  || 1,1,256,256->1,256,256-tile->3,256,256
                        '''
                        input_permuted: torch.Size([1, 1, 256, 256])
                        label_permuted: torch.Size([1, 3, 256, 256])
                        seg_permuted: torch.Size([1, 1, 256, 256])
                        '''
                        
                        for j, label_name in enumerate(label_name_list): # i: int, slice: int
                            show_and_save_img_3output_slice(subject_number=subject_number ,objects_num=i,slice_num=slice, ch=j, label=label_permuted[0,j,:,:], seg=seg_permuted[0,0,:,:], mean=output_mean_map[j], std=output_std_map[j], std_per_mean=output_std_per_mean_map_thres[j], std_per_mean_seg=output_std_per_mean_map_thres_seg[j], show_flag=False, label_name=label_name)
                        mean_nii_gz_list += [ output_mean_map ]
                        std_per_mean_nii_gz_list += [ output_std_per_mean_map_thres ]
                        std_per_mean_seg_nii_gz_list += [output_std_per_mean_map_thres_seg]            

                    mean_stack = torch.stack(tuple(mean_nii_gz_list) , dim=1 )
                    std_per_mean_stack = torch.stack(tuple(std_per_mean_nii_gz_list) , dim=1 )
                    std_per_mean_seg_stack = torch.stack(tuple(std_per_mean_seg_nii_gz_list), dim=1)
            
                    # Tensor to Numpy
                    mean_stack = mean_stack.numpy()
                    std_per_mean_stack = std_per_mean_stack.numpy()
                    std_per_mean_seg_stack = std_per_mean_seg_stack.numpy() 

            ######### 1. mean ##########
                    save_path_1 = f'{save_folder_path}/ktrans_mean.nii.gz' # ktrans
                    save_path_2 = f'{save_folder_path}/vp_mean.nii.gz' # vp
                    save_path_3 = f'{save_folder_path}/ve_mean.nii.gz' # ve

                    # mean_nft_1 = nib.Nifti1Image(np.transpose(mean_stack[0]) ,refvol.affine,refvol.header)
                    # mean_nft_2 = nib.Nifti1Image(np.transpose(mean_stack[1]) ,refvol.affine,refvol.header)
                    # mean_nft_3 = nib.Nifti1Image(np.transpose(mean_stack[2]) ,refvol.affine,refvol.header)
                    # print(f'mean_stack[0]:{mean_stack[0].shape}')
                    # print(f'mean_stack[0]:{np.transpose(mean_stack[0], ).shape}')
                    
                    mean_nft_1 = nib.Nifti1Image(mean_stack[0].transpose((1,2,0)) ,refvol.affine,refvol.header ,refvol.affine,refvol.header)
                    mean_nft_2 = nib.Nifti1Image(mean_stack[1].transpose((1,2,0)) ,refvol.affine,refvol.header)
                    mean_nft_3 = nib.Nifti1Image(mean_stack[2].transpose((1,2,0)) ,refvol.affine,refvol.header)

                    nib.save(mean_nft_1, save_path_1)
                    nib.save(mean_nft_2, save_path_2)
                    nib.save(mean_nft_3, save_path_3)

            ######### 2. mean per std ##########
                    save_path_1_spm = f'{save_folder_path}/ktrans_spm.nii.gz' # ktrans
                    save_path_2_spm = f'{save_folder_path}/vp_spm.nii.gz' # vp
                    save_path_3_spm = f'{save_folder_path}/ve_spm.nii.gz' # ve

                    spm_nft_1 = nib.Nifti1Image(std_per_mean_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
                    spm_nft_2 = nib.Nifti1Image(std_per_mean_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
                    spm_nft_3 = nib.Nifti1Image(std_per_mean_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

                    nib.save(spm_nft_1, save_path_1_spm)
                    nib.save(spm_nft_2, save_path_2_spm)
                    nib.save(spm_nft_3, save_path_3_spm)

            ######### 3. mean per std_seg ##########

                    save_path_1_spm_seg = f'{save_folder_path}/ktrans_spm_seg.nii.gz' # ktrans
                    save_path_2_spm_seg = f'{save_folder_path}/vp_spm_seg.nii.gz' # vp
                    save_path_3_spm_seg = f'{save_folder_path}/ve_spm_seg.nii.gz' # ve

                    spm_nft_1 = nib.Nifti1Image(std_per_mean_seg_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
                    spm_nft_2 = nib.Nifti1Image(std_per_mean_seg_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
                    spm_nft_3 = nib.Nifti1Image(std_per_mean_seg_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

                    nib.save(spm_nft_1, save_path_1_spm_seg)
                    nib.save(spm_nft_2, save_path_2_spm_seg)
                    nib.save(spm_nft_3, save_path_3_spm_seg)

                    print(f'save_folder_path ... {save_folder_path}')
