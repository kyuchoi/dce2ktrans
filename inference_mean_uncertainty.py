##### Work space path 지정 #####
base_path = '/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/240605_dce2ktrans'

##### Mean & Uncertainty map 에 사용될 Samples 지정 #####
### 01 VS 67
### 0167  VS  2345

N_11 = 0
N_12 = 1

N_21 = 6
N_22 = 7

N_idx = str(N_11)+str(N_12)+str(N_21)+str(N_22)
N_list = [N_11, N_12, N_21, N_22]



##### library import #####
verbose = False # Print Flag for Debugging || True: print on, False: print off
#https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
# # MS-SSSIM은 32이상일때만 적용가능해서 32로 바꿈. 
from functools import total_ordering
from operator import index
import os
# CPU 과소비 하지 않게 설정 (실험 여러개 돌릴 수 있게 ) by 이한얼 선생님
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=2

import numpy as np
from datetime import datetime
import glob
import random

import matplotlib.pyplot as plt
import nibabel as nib

import torch
torch.set_num_threads(4)
import torch.nn as nn

from tqdm import tqdm

from utils_dce2ktrans import nii_to_numpy, visualize_ktrans_patch,visualize_ktrans_final, visualize_ktrans_final_save, visualize_ktrans_final_save_extend, Weighted_L1_patch

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

import pdb
import glob

from PIL import Image
import matplotlib.pyplot as plt
import copy

device = torch.device('cpu')
# device = torch.device('cuda')
print('torch.cuda.is_available:', torch.cuda.is_available())
print('torch.cuda.device_count():',  torch.cuda.device_count())
print('torch.cuda.current_device():',  torch.cuda.current_device())
print('torch.cuda.get_device_name(0):',torch.cuda.get_device_name(0))

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from unet_blocks import *
from unet import Unet
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from probabilistic_unet import *
############################## Probabilistic U-Net implement! ####################################
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from load_LIDC_data import LIDC_IDRI
# from probabilistic_unet import add_ex
# from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
from glob import glob
import pandas as pd
############################

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
    visuaialization_slice_file_path = f'{base_path}/231101_figs_show/visualization/{subject_number}_{str(objects_num)}_{str(slice_num)}_{str(ch)}_{label_name}_TCN-PUnet.png'# f'./{folder_path}/visualization/{subject_number}_{str(objects_num)}_{str(slice_num)}_{str(ch)}_{label_name}_TCN-PUnet.png'
    # visuaialization_slice_file_path = f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/231101_figs_show/visualization/{subject_number}_{str(objects_num)}_{str(slice_num)}_{str(ch)}_{label_name}_TCN-PUnet.png'# f'./{folder_path}/visualization/{subject_number}_{str(objects_num)}_{str(slice_num)}_{str(ch)}_{label_name}_TCN-PUnet.png'
    if os.path.exists( '/'.join((visuaialization_slice_file_path).split('/')[:-1])) == False:
        os.makedirs( '/'.join((visuaialization_slice_file_path).split('/')[:-1])) # get folder path W/O file name

    plt.savefig(visuaialization_slice_file_path, bbox_inches='tight',pad_inches = 0)
    if show_flag == True:
        plt.show()
    plt.close()
img = np.random.randint(2, size=(256,256))

show_and_save_img_3output_slice(objects_num=1, slice_num=1, ch=1, label= copy.deepcopy(img), seg= copy.deepcopy(img), mean= copy.deepcopy(img), std= copy.deepcopy(img), std_per_mean= copy.deepcopy(img), std_per_mean_seg = copy.deepcopy(img), show_flag=True, label_name='ktrans')




############################

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


# # train_files = make_nii_path_dataset(base_dir+"/train",input_filename,label_filename,seg_filename,mask_filename, train_ratio) # 
# valid_files = make_nii_path_dataset(base_dir+"/valid",input_filename,label_filename,seg_filename,mask_filename, valid_ratio)
# test_files = make_nii_path_dataset(base_dir+"/test",input_filename,label_filename,seg_filename,mask_filename, test_ratio)


from pytorch_ssim import *
# 
ssim = SSIM()
# Numpy 안됨 torchTensor만 됨


# import numpy as np
# print(np.ones((1,3, 256,256)).shape)
# ssim = SSIM()
# ssim(np.ones((1,1,256,256)), np.ones((1,1,256,256)))
import torch
import numpy as np
# print(np.ones((1,3, 256,256)).shape)

img1 = Variable(torch.rand(1, 1, 256, 256))
img2 = Variable(torch.rand(1, 1, 256, 256))

# img1 = np.random.rand(1,1,256,256 )
# img2 = np.random.rand(1,1,256,256 )

# img1 = np.random.rand(1,1,256,256 )
# print(img1.shape)

# torch.tensor((1,1,256,256)), torch.tensor((1,1,256,256)) 
# ssim = SSIM()
# print(np.random.rand(1,1,256,256 ).shape)

# ssim( torch.ones((1,1,256,256)), torch.ones((1,1,256,256)) )  # B C H W)


# torch.tensor여야 함!
ssim(img1, img2)

from einops import rearrange



# slice 별로 불러오기
from glob import glob


slice_num_list = [ 'slice_'+str(ii).zfill(2) for ii in list(range(40))] # [18:22]

# print(slice_num_list) # ([1])
# import pdb
# pdb.set_trace()
###############################
ssim_avg_list = []
ssim_1_avg_list = []
ssim_2_avg_list = []
ssim_3_avg_list = []

mse_list = []
rmse_1_avg_list = []
rmse_2_avg_list = []
rmse_3_avg_list = []

nrmse_1_avg_list = []
nrmse_2_avg_list = []
nrmse_3_avg_list = []
################################


total_rmse_1_list = []
total_rmse_2_list = []
total_rmse_3_list = []

total_nrmse_1_list = []
total_nrmse_2_list = []
total_nrmse_3_list = []



######################################### DataFrame에 통계 데이터 저장하도록 구축하기* ############################
# df = pd.dataframe()
# row : 98*40 + 1
# col :


'''
with pd.ExcelWriter('sample.xlsx') as writer:
	df[df.a == 'A'].to_excel(writer, sheet_name='A')
	df[df.a == 'B'].to_excel(writer, sheet_name='B')
	df[df.a == 'C'].to_excel(writer, sheet_name='C')
    
with pd.ExcelWriter('sample.xlsx') as writer:
	for a in df.a:
		df[df.a == f"{a}"].to_excel(writer, sheet_name=f"{a}")  
'''



ssim_list = []
ssim_1_avg_list = []
ssim_2_avg_list = []
ssim_3_avg_list = []

rmse_1_avg_list = []
rmse_2_avg_list = []
rmse_3_avg_list = []

nrmse_1_avg_list = []
nrmse_2_avg_list = []
nrmse_3_avg_list = []



ssim_1_seg_avg_list = []
ssim_2_seg_avg_list = []
ssim_3_seg_avg_list = []

rmse_1_seg_avg_list = []
rmse_2_seg_avg_list = []
rmse_3_seg_avg_list = []

nrmse_1_seg_avg_list = []
nrmse_2_seg_avg_list = []
nrmse_3_seg_avg_list = []

ssim_seg_percentile_total_list = []

######################################### ############################# #########################################


############# ~~~ add list 
pass_tuple_list = [ ('50148231','slice_00'), ('48262855','slice_00'), \
############# ~~~ add list 
('20252076','slice_00'), ('40892562','slice_00'), ('50935567','slice_00'), \
('40892562','slice_01'), \
('40892562','slice_02'), ('40892562','slice_03'), \
('40892562','slice_03'), \
('40892562','slice_04'), \
('48505615','slice_36'), \
('48505615','slice_37'), \
('29899179','slice_38'), \
('29899179','slice_39'), ('50225501','slice_39'),('50275847','slice_39'), ('47983337','slice_39'), ('51390840','slice_39'), ('48284196','slice_39'),('48505615','_slice_39')]

pass_list = []

#  glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/*/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat0.nii.gz')

# N_11 = 0
# N_12 = 1

# N_21 = 6
# N_22 = 7

# ### 01 VS 67

# ### 0167  VS  2345


# N_idx = str(N_11)+str(N_12)+str(N_21)+str(N_22)

# N_list = [N_11, N_12, N_21, N_22]


flag = True
continue_flag = False
####### obj 단위
object_num_path_list = glob('/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/*')
object_num_list = []
for object_num_path in object_num_path_list:
    object_num = object_num_path.split('/')[-1]
    if os.path.isdir(object_num_path):
        object_num_list += [object_num]
# print(object_num_list)
object_num_list.sort()
print(object_num_list)

del object_num_path, object_num_path_list



for object_num in object_num_list:
# for ii in slice_num_list: #['00','01','02','03','04','05','06','07','08','30','31','32','33','34','35','36','37','38','39']:
    

#### 11-1. For문으로 바꿀 수 있을 듯! N_ii 이쪽만 정리해서 for으로 돌리면!

#### 10-1.  N = 2개 파일들 불러오기test_files_full
  
    # PVDM은 분리가 안 됐긴 한데

    # test_files_full = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/g_kvpve_fixed_clipped_0_1.nii.gz') # 98 개 잘 나옴

    test_files_full = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_11}.nii.gz') # 98 개 잘 나옴

    test_files_full2 = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_12}.nii.gz') # 98 개 잘 나옴

    test_files_full21 = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_21}.nii.gz') # 98 개 잘 나옴

    test_files_full22 = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_22}.nii.gz') # 98 개 잘 나옴


    test_files_full.sort()
    test_files_full2.sort()
    test_files_full21.sort()
    test_files_full22.sort()
    # test_files_full = glob('/mnt/ssd/ylyoo/intermediate_filtered_split/test/*/slice_*/slice_forwarded_tcn_3fuse_baseline_KtransVpVe_clipped_0_1_230918.nii.gz')


    # /mnt/ssd/ylyoo/intermediate_filtered_split/test/*/slice_*/slice_g_kvpve_fixed_clipped_0_1.nii.gz
    # g_kvpve_fixed_clipped_0_1.nii.gz

    # /mnt/ssd/ylyoo/intermediate_filtered_split/test/20252076/slice_00/slice_forwarded_tcn_3fuse_baseline_KtransVpVe_clipped_0_1_230918.nii.gz
    # 230726/006_statistical_test_SSIM_MRI_v74_231013_seg_region.py
    # len(test_files_full)/40 # 116

    batch_size = 1

    ssim_list = []
    ssim_1_list = []
    ssim_2_list = []
    ssim_3_list = []

    rmse_1_list = []
    rmse_2_list = []
    rmse_3_list = []

    nrmse_1_list = []
    nrmse_2_list = []
    nrmse_3_list = []

    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0

    ssim_seg_percentile_list = []

################################ seg region
    ssim_1_seg_list = []
    ssim_2_seg_list = []
    ssim_3_seg_list = []

    rmse_1_seg_list = []
    rmse_2_seg_list = []
    rmse_3_seg_list = []

    nrmse_1_seg_list = []
    nrmse_2_seg_list = []
    nrmse_3_seg_list = []



    # slice 마다 결정하는 것임*
    seg_cnt_flag = False # seg_percentil이 일정 이상 넘겼을 때 발동하게 하기 ?  seg_percentil > 0.01~0.03 
    seg_cnt = 0 # 유효한 seg 갯수를 내서 나중에 평균 내줄 것임

# slice_num_list
# test_file_path
# i

# slice_num_object_num  ,'ii'

    # for i, test_file_path in enumerate(tqdm(test_files_full)):
    
    # aligned가 맞는지 체크 필요!

#### 11-2.
    # for i, test_file_path in enumerate(test_files_full):
    #     for test_files_full_kk in test_files_full_list:
    #         # ~~~ 반복 ~~~
#### 11-2.
    # 231025
    ### Object 단위로 모아서 저장하기**
    test_forward_mean_1_list = []
    test_forward_mean_2_list = []
    test_forward_mean_3_list = []

    test_forward_std_1_list = []
    test_forward_std_2_list = []
    test_forward_std_3_list = []


##################################### object 별로 저장하기!! ##############################
    std_per_mean_seg_stack = []
    std_per_mean_stack = []
    mean_stack = []
    std_stack = []

    g_mean_stack_1 = []
    g_mean_stack_2 = [] #[test_gt_2]
    g_mean_stack_3 = [] #[test_gt_3]
    g_mean_stack = []

#### 10-2
    for i, (test_file_path) in enumerate(zip(test_files_full)): # , test_file_path_2, test_file_path_21, test_file_path_22
        test_file_path = test_file_path[0]
        # test_file_path
        with torch.no_grad():


            # test_files_full
        # for ii in slice_num_list: 
            # test_files_full_list = []
            # test_files_full = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/*/slice_{ii}/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_11}.nii.gz') # 98 개 잘 나옴

            # test_files_full2 = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/*/slice_{ii}/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_12}.nii.gz') # 98 개 잘 나옴
                # test_files_full = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_11}.nii.gz') # 98 개 잘 나옴
                # test_files_full2 = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_12}.nii.gz
            pass
        ### 1-1. 로딩할 데이터 관련한 path 들 정리
            # object_num = test_file_path.split('/')[-3]
            slice_num = test_file_path.split('/')[-2]
            if int(slice_num.split('_')[-1]) != i:
                import pdb
                pdb.set_trace()
            slice_num_int = int(slice_num.split('_')[-1])

            test_gt_path = f'/mnt/ssd/ylyoo/intermediate_filtered_split/test/{object_num}/{slice_num}/g_kvpve_fixed_clipped_0_1.nii.gz'
            test_brain_mask_path = f'/mnt/ssd/ylyoo/intermediate_filtered_split/test/{object_num}/g_dce_mask.nii.gz'
            # g_dce_mask.nii.gz   
            # 230726/006_statistical_test_SSIM_MRI_v74_231011.py
            test_seg_path = f'/mnt/ssd/ylyoo/intermediate_filtered_split/test/{object_num}/g_seg.nii.gz'

            refvol = nib.load(f'/mnt/ssd/ylyoo/intermediate_filtered_split/test/{object_num}/g_ktrans.nii.gz') # folder_name_dict[i]
    
            pass_flag = False
            if (object_num,slice_num) in pass_tuple_list:
                pass_list += [(object_num,slice_num)]
                pass_flag = True

            # test_files_full2 = glob(f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat{N_12}.nii.gz') # 98 개 잘 나옴

            save_folder_path = f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}'
            save_folder_subfolder_path = f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/{slice_num}'
            if not( os.path.isdir(save_folder_path) ):
                os.makedirs(save_folder_path)
            # save_folder_path = f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}'
            # save_folder_subfolder_path = f'/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/{slice_num}'

            # for i_1 in range(len(test_files_full_list)):
            #     for i_2 in range(i_1+1, len(test_files_full_list)):
            #         if flag==True:
            #             print(f'{i_1}, {i_2}')
            #             flag = False
            #         if test_files_full_list[i_1] != test_files_full_list[i_2]:
            #             import pdb
            #             pdb.set_trace()
            for kk in range(len(test_files_full)):
                try:
                    if not (  test_files_full[kk].split('/')[:-1] == test_files_full2[kk].split('/')[:-1] ) : # 모두 같지 않다면 XX
                        import pdb
                        pdb.set_trace()
                except:
                    print(f'test_files_full[kk]:{test_files_full}')
                    print(f'test_files_full2[kk]:{test_files_full2}')
                    print(f'len test_files_full[kk]:{len(test_files_full)}')
                    print(f'len test_files_full2[kk]:{len(test_files_full2)}')
                    continue_flag=True
                    
            if continue_flag==True:
                continue_flag = False
                input()
            # print(test_file_path)
            #
            # brain mask 씌워줘도 이런가?
            # Ktrans가 많이 망가짐..
            
            # /mnt/ssd/ylyoo/intermediate_filtered_split/test/20252076/slice_00
            
            # /mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_73_inference/47013656/slice_00/slice_forwarded_forwarded_tcn_0_1_punet_230919.nii.gz
            
            # /mnt/ssd/ylyoo/intermediate_filtered_split/test/47013656/slice_00/slice_g_kvpve_fixed_clipped_0_1.nii.gz

            # print(test_file_path)
            # print(test_gt_path)
        ### 1-2. 로딩할 데이터 path에 따라 로딩해오고 dimension 통일해주기 정리

##################### GT 만 구할 거니까 모두 주석처리!!!!!!!!!!! ######################
    #         test_forward = torch.Tensor(nii_to_numpy(test_file_path))    
    #         test_forward = rearrange(test_forward, 'h w b c -> b c h w')
    
    # ######## 231024. 10-3. 여기를 증설!
    #         test_forward2 = torch.Tensor(nii_to_numpy(test_file_path_2))    
    #         test_forward2 = rearrange(test_forward2, 'h w b c -> b c h w')

    #         test_forward21 = torch.Tensor(nii_to_numpy(test_file_path_21))    
    #         test_forward21 = rearrange(test_forward21, 'h w b c -> b c h w')
    
    # ######## 231024. 10-3. 여기를 증설!
    #         test_forward22 = torch.Tensor(nii_to_numpy(test_file_path_22))    
    #         test_forward22 = rearrange(test_forward22, 'h w b c -> b c h w')



            test_gt = torch.Tensor(nii_to_numpy(test_gt_path))   
            test_gt = rearrange(test_gt, 'h w b c -> b c h w') # torch.Size([1, 3, 256, 256])
            
            test_seg = torch.Tensor(nii_to_numpy(test_seg_path))         # torch.Size([256, 256, 40])
            test_seg = rearrange(test_seg, 'h w (s c) -> s c h w', s=40) # torch.Size([40, 3, 256, 256])
            test_seg_slice = test_seg[slice_num_int,...].unsqueeze(0)    # torch.Size([1, 1, 256, 256])
            # print(test_seg.shape) # torch.Size([256, 256, 40])
            # input()
    ##################################### 수정할 것!!!!!!!!!! ###################################
            # ssim_list += [ssim(test_forward, test_gt)]
            # test_seg = test_seg.float()
            # print(type(test_seg)) # <class 'torch.Tensor'>
            test_seg_slice = torch.where( test_seg_slice > 0., test_seg_slice, torch.zeros(test_seg_slice.shape) ) #  test_seg_slice > 0.03
    ############### percentile 구하기 - reasonable! 
            # print(torch.unique(test_seg_slice, return_counts=True)[1][0]) # zero 갯수 카운트
            # print(f'test_seg_slice.shape: {test_seg_slice.shape}') # torch.Size([1, 1, 256, 256])

            test_seg_slice_pixels_number = (256**2) - torch.unique(test_seg_slice, return_counts=True)[1][0]
            test_seg_slice_percentile = (test_seg_slice_pixels_number) / (256**2) # nonzeors number / total number

            if test_seg_slice_percentile !=0:
                # print(f'SEG O - test_seg_slice_percentile:  {object_num} || {slice_num} : {test_seg_slice_percentile*100}%')
                ### txt 에 로그 저장!
                with open(f'./logs_mutation_segments_percentile_stats_test_230830.txt', 'a+') as f: 
                    print_str = f"test_seg_slice_percentile: {object_num} || {slice_num} : {test_seg_slice_percentile*100}%\n"
                    f.write(print_str) #f'Epoch #{self.current_epoch}-'

                # if test_seg_slice_percentile > 0.01
            if test_seg_slice_percentile > 0.00005: # 0.0001 : # 30/2500 0.0004 # test_seg_slice_pixels_number >3.xx
                seg_cnt_flag = True
                seg_cnt+=1
            else:
                seg_cnt_flag = False
            # excel sheet 로 정리해서 추출해보자*
    ############################################################################################
            # torch.where(torch.ones((4,4))>0, torch.zeros((4,4)),torch.ones((4,4)))
            # torch.unique(torch.where( test_seg > 0.01, test_seg, torch.zeros(test_seg.shape) ) , return_counts=True)
    ####################
            test_brain_mask = torch.Tensor(nii_to_numpy(test_brain_mask_path)) # torch.Size([256, 256, 40])
            test_brain_mask = rearrange(test_brain_mask, 'h w (s c) -> s c h w', s= 40) # torch.Size([40, 1, 256, 256])
            test_brain_mask_slice = test_brain_mask[slice_num_int,...].unsqueeze(0) # torch.Size([1, 1, 256, 256])
            
            del test_brain_mask, test_seg

    #         test_forward_1 = test_forward[:,0,:,:].unsqueeze(1).detach()
    #         test_forward_2 = test_forward[:,1,:,:].unsqueeze(1).detach()
    #         test_forward_3 = test_forward[:,2,:,:].unsqueeze(1).detach()
    # ######## 231024. 10-4. 여기를 증설!
    #         test_forward2_1 = test_forward2[:,0,:,:].unsqueeze(1).detach()
    #         test_forward2_2 = test_forward2[:,1,:,:].unsqueeze(1).detach()
    #         test_forward2_3 = test_forward2[:,2,:,:].unsqueeze(1).detach()
    # ######## 231024. 10-4. 여기를 증설!
    #         test_forward21_1 = test_forward21[:,0,:,:].unsqueeze(1).detach()
    #         test_forward21_2 = test_forward21[:,1,:,:].unsqueeze(1).detach()
    #         test_forward21_3 = test_forward21[:,2,:,:].unsqueeze(1).detach()

    #         test_forward22_1 = test_forward22[:,0,:,:].unsqueeze(1).detach()
    #         test_forward22_2 = test_forward22[:,1,:,:].unsqueeze(1).detach()
    #         test_forward22_3 = test_forward22[:,2,:,:].unsqueeze(1).detach()

            test_gt_1 = test_gt[:,0,:,:].unsqueeze(1)
            test_gt_2 = test_gt[:,1,:,:].unsqueeze(1)
            test_gt_3 = test_gt[:,2,:,:].unsqueeze(1)

            test_gt_1_counts = torch.unique(test_gt_1, return_counts=True)
            test_gt_2_counts = torch.unique(test_gt_2, return_counts=True)
            test_gt_3_counts = torch.unique(test_gt_3, return_counts=True)

            counts_zeors_1 = test_gt_1_counts[1][0]
            counts_zeors_2 = test_gt_2_counts[1][0]
            counts_zeors_3 = test_gt_3_counts[1][0]
            
            # import pdb
            # pdb.set_trace()
            # print(test_forward.shape)
            # print(test_gt.shape)
            # print(test_gt_3.shape) # torch.Size([1, 1, 256, 256])

            # import pdb
            # pdb.set_trace()


############ g_mean
            g_mean_stack += [test_gt]
            g_mean_stack_1 += [test_gt_1]
            g_mean_stack_2 += [test_gt_2]
            g_mean_stack_3 += [test_gt_3]


    ######## 231024. 10-5. Mean, STD 계산!
            
            # test_forward_mean_1 = torch.mean( torch.stack([test_forward_1]+[test_forward2_1]+[test_forward21_1]+[test_forward22_1], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
            # test_forward_mean_2 = torch.mean( torch.stack([test_forward_2]+[test_forward2_2]+[test_forward21_2]+[test_forward22_2], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
            # test_forward_mean_3 = torch.mean( torch.stack([test_forward_3]+[test_forward2_3]+[test_forward21_3]+[test_forward22_3], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
            # test_forward_mean = torch.cat([test_forward_mean_1,test_forward_mean_2,test_forward_mean_3],dim=1)
            # mean_stack += [test_forward_mean]


            # test_forward_std_1 = torch.std( torch.stack([test_forward_1]+[test_forward2_1]+[test_forward21_1]+[test_forward22_1], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
            # test_forward_std_2 = torch.std( torch.stack([test_forward_2]+[test_forward2_2]+[test_forward21_2]+[test_forward22_2], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
            # test_forward_std_3 =  torch.std( torch.stack([test_forward_3]+[test_forward2_3]+[test_forward21_3]+[test_forward22_3], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
            # test_forward_std = torch.cat([test_forward_std_1,test_forward_std_2,test_forward_std_3],dim=1)
            # std_stack += [test_forward_std]

            # test_forward_std_per_mean_1 = test_forward_std_1 / test_forward_mean_1 # torch.Size([1, 1, 256, 256])
            # test_forward_std_per_mean_2 = test_forward_std_2 / test_forward_mean_2  # torch.Size([1, 1, 256, 256])
            # test_forward_std_per_mean_3 = test_forward_std_3 / test_forward_mean_3  # torch.Size([1, 1, 256, 256])
            # # test_forward_std_per_mean = torch.cat([test_forward_std_per_mean_1,test_forward_std_per_mean_2,test_forward_std_per_mean_3],dim=1)
            # # std_per_mean_stack += [test_forward_std_per_mean]
            
            # test_forward_std_per_mean_thres_1 = test_forward_std_1.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_1.shape)) / (test_forward_mean_1) # torch.Size([1, 1, 256, 256])
            # test_forward_std_per_mean_thres_2 = test_forward_std_2.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_2.shape)) / (test_forward_mean_2) # torch.Size([1, 1, 256, 256])
            # test_forward_std_per_mean_thres_3 = test_forward_std_3.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_3.shape)) / (test_forward_mean_3) # torch.Size([1, 1, 256, 256])
            # test_forward_std_per_mean_thres = torch.cat([test_forward_std_per_mean_thres_1,test_forward_std_per_mean_thres_2,test_forward_std_per_mean_thres_3],dim=1)
            # std_per_mean_stack += [test_forward_std_per_mean_thres]

            # test_forward_std_per_mean_thres_seg_1 = test_forward_std_per_mean_thres_1 * test_seg_slice # torch.Size([1, 1, 256, 256])   # test_seg_slice  # ... == torch.tile(test_seg_slice[0], (3,1,1))
            # test_forward_std_per_mean_thres_seg_2 = test_forward_std_per_mean_thres_2 * test_seg_slice # torch.Size([1, 1, 256, 256])
            # test_forward_std_per_mean_thres_seg_3 = test_forward_std_per_mean_thres_3 * test_seg_slice # torch.Size([1, 1, 256, 256])
            # test_forward_std_per_mean_thres_seg = torch.cat([test_forward_std_per_mean_thres_seg_1,test_forward_std_per_mean_thres_seg_2,test_forward_std_per_mean_thres_seg_3],dim=1)
            # std_per_mean_seg_stack += [test_forward_std_per_mean_thres_seg]        
            # # B C H W                

            if pass_flag == True:
                pass_flag = False
                continue
            # std_per_mean_seg_stack = []
            # std_per_mean_stack = []
            mean_stack = []


            # for j, label_name in enumerate(label_name_list): # i: int, slice: int
            # show_and_save_img_3output_slice(subject_number=object_num ,objects_num=i,slice_num=slice, ch=j, label=label_permuted[0,j,:,:], seg=seg_permuted[0,0,:,:], mean=output_mean_map[j], std=output_std_map[j], std_per_mean=output_std_per_mean_map_thres[j], std_per_mean_seg=output_std_per_mean_map_thres_seg[j], show_flag=False, label_name=label_name)
    #         output_std_per_mean_map_thres
    #         output_std_per_mean_map_thres_seg
    # ######### 2. mean per std ##########
    #         save_path_1_spm = f'{save_folder_subfolder_path}/ktrans_spm.nii.gz' # ktrans
    #         save_path_2_spm = f'{save_folder_subfolder_path}/vp_spm.nii.gz' # vp
    #         save_path_3_spm = f'{save_folder_subfolder_path}/ve_spm.nii.gz' # ve

    #         spm_nft_1 = nib.Nifti1Image(std_per_mean_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
    #         spm_nft_2 = nib.Nifti1Image(std_per_mean_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
    #         spm_nft_3 = nib.Nifti1Image(std_per_mean_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

    #         nib.save(spm_nft_1, save_path_1_spm)
    #         nib.save(spm_nft_2, save_path_2_spm)
    #         nib.save(spm_nft_3, save_path_3_spm)
    ######### 3. mean per std_seg! ##########


            # test_forward_std_per_mean_thres_1 = test_forward_std_1.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_1.shape)) / (test_forward_mean_1)
            # test_forward_std_per_mean_thres_2 = test_forward_std_2.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_2.shape)) / (test_forward_mean_2)
            # test_forward_std_per_mean_thres_3 = test_forward_std_3.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_3.shape)) / (test_forward_mean_3)

            # test_forward_std_per_mean_thres_seg_1 = test_forward_std_per_mean_thres_1 * test_seg_slice   # test_seg_slice  # ... == torch.tile(test_seg_slice[0], (3,1,1))
            # test_forward_std_per_mean_thres_seg_2 = test_forward_std_per_mean_thres_2 * test_seg_slice
            # test_forward_std_per_mean_thres_seg_3 = test_forward_std_per_mean_thres_3 * test_seg_slice

            # test_forward_mean_1 = torch.mean( torch.stack([test_forward_1]+[test_forward2_1], dim = 0), dim=0 )
            # test_forward_mean_2 = torch.mean( torch.stack([test_forward_2]+[test_forward2_2], dim = 0), dim=0 )
            # test_forward_mean_3 = torch.mean( torch.stack([test_forward_3]+[test_forward2_3], dim = 0), dim=0 )


        # 231103
            # save_path_std = f'{save_folder_subfolder_path}/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_STD_{N_idx}.nii.gz'

            # save_path_mean = f'{save_folder_subfolder_path}/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_MEAN_{N_idx}.nii.gz'
            # # save_path_1_mean = f'{save_folder_subfolder_path}/ktrans_mean_{N_idx}.nii.gz' # ktrans
            # # save_path_2_mean = f'{save_folder_subfolder_path}/vp_mean_{N_idx}.nii.gz' # vp
            # # save_path_3_mean = f'{save_folder_subfolder_path}/ve_mean_{N_idx}.nii.gz' # ve
            # save_path_spm = f'{save_folder_subfolder_path}/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_SPM_{N_idx}.nii.gz'
            # # save_path_1_spm = f'{save_folder_subfolder_path}/ktrans_spm_{N_idx}.nii.gz' # ktrans
            # # save_path_2_spm = f'{save_folder_subfolder_path}/vp_spm_{N_idx}.nii.gz' # vp
            # # save_path_3_spm = f'{save_folder_subfolder_path}/ve_spm_{N_idx}.nii.gz' # ve
            # save_path_spm_seg = f'{save_folder_subfolder_path}/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_SPM_SEG_{N_idx}.nii.gz'
            # # save_path_1_spm_seg = f'{save_folder_subfolder_path}/ktrans_spm_seg_{N_idx}.nii.gz' # ktrans
            # # save_path_2_spm_seg = f'{save_folder_subfolder_path}/vp_spm_seg_{N_idx}.nii.gz' # vp
            # # save_path_3_spm_seg = f'{save_folder_subfolder_path}/ve_spm_seg_{N_idx}.nii.gz' # ve

            # std_nft = nib.Nifti1Image(test_forward_std,refvol.affine,refvol.header) 
            # # mean_nft_1 = nib.Nifti1Image(test_forward_mean_1,refvol.affine,refvol.header) 
            # # mean_nft_2 = nib.Nifti1Image(test_forward_mean_2,refvol.affine,refvol.header) 
            # # mean_nft_3 = nib.Nifti1Image(test_forward_mean_3,refvol.affine,refvol.header) 
            # nib.save(std_nft, save_path_std)

            # mean_nft = nib.Nifti1Image(test_forward_mean,refvol.affine,refvol.header) 
            # # mean_nft_1 = nib.Nifti1Image(test_forward_mean_1,refvol.affine,refvol.header) 
            # # mean_nft_2 = nib.Nifti1Image(test_forward_mean_2,refvol.affine,refvol.header) 
            # # mean_nft_3 = nib.Nifti1Image(test_forward_mean_3,refvol.affine,refvol.header) 
            # nib.save(mean_nft, save_path_mean)
            # # nib.save(mean_nft_1, save_path_1_mean)
            # # nib.save(mean_nft_2, save_path_2_mean)
            # # nib.save(mean_nft_3, save_path_3_mean)

            # spm_nft = nib.Nifti1Image(test_forward_std_per_mean_thres,refvol.affine,refvol.header)
            # # spm_nft_1 = nib.Nifti1Image(test_forward_std_per_mean_thres_1,refvol.affine,refvol.header)
            # # spm_nft_2 = nib.Nifti1Image(test_forward_std_per_mean_thres_2,refvol.affine,refvol.header)
            # # spm_nft_3 = nib.Nifti1Image(test_forward_std_per_mean_thres_3,refvol.affine,refvol.header)
            # nib.save(spm_nft, save_path_spm)
            # # nib.save(spm_nft_1, save_path_1_spm)
            # # nib.save(spm_nft_2, save_path_2_spm)
            # # nib.save(spm_nft_3, save_path_3_spm)

            # spm_seg_nft = nib.Nifti1Image(test_forward_std_per_mean_thres_seg,refvol.affine,refvol.header)
            # # spm_seg_nft_1 = nib.Nifti1Image(test_forward_std_per_mean_thres_seg_1,refvol.affine,refvol.header)
            # # spm_seg_nft_2 = nib.Nifti1Image(test_forward_std_per_mean_thres_seg_2,refvol.affine,refvol.header)
            # # spm_seg_nft_3 = nib.Nifti1Image(test_forward_std_per_mean_thres_seg_3,refvol.affine,refvol.header)
            # nib.save(spm_seg_nft, save_path_spm_seg)
            # # nib.save(spm_seg_nft_1, save_path_1_spm_seg)
            # # nib.save(spm_seg_nft_2, save_path_2_spm_seg)
            # # nib.save(spm_seg_nft_3, save_path_3_spm_seg)



            # test_forward_std_per_mean_thres_1 = test_forward_std_1.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_1.shape)) / (test_forward_mean_1)
            # test_forward_std_per_mean_thres_2 = test_forward_std_2.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_2.shape)) / (test_forward_mean_2)
            # test_forward_std_per_mean_thres_3 = test_forward_std_3.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_3.shape)) / (test_forward_mean_3)

            # test_forward_std_per_mean_thres_seg_1 = test_forward_std_per_mean_thres_1 * test_seg_slice   # test_seg_slice  # ... == torch.tile(test_seg_slice[0], (3,1,1))
            # test_forward_std_per_mean_thres_seg_2


            # slice_save_path_output = f'{save_slice_folder_path}/slice_{forward_filename}_repeat{nn}.nii.gz'
            # slice_nft_img = nib.Nifti1Image(rearrange(obj_forward_slice.detach().numpy(), 'B H W C -> H W B C'),refvol.affine,refvol.header) # 256 256 slice_cnt 3
            # nib.save(slice_nft_img, slice_save_path_output)
    ###########################################################################################
    ###########################################################################################
    ####################################   EVAL   #############################################
#             test_forward_mean_1_list += [test_forward_mean_1]
#             test_forward_mean_2_list += [test_forward_mean_2]
#             test_forward_mean_3_list += [test_forward_mean_3]

#             test_forward_std_1_list += [test_forward_std_1]
#             test_forward_std_2_list += [test_forward_std_2]
#             test_forward_std_3_list += [test_forward_std_3]
#     ######## 231024. 10-5. Mean, STD 계산!
#             # import pdb
#             # pdb.set_trace()
#             ssim_1_list += [ssim(test_forward_mean_1, test_gt_1)]
#             ssim_2_list += [ssim(test_forward_mean_2, test_gt_2)]
#             ssim_3_list += [ssim(test_forward_mean_3, test_gt_3)]
            
#             if seg_cnt_flag==True:
#                 ssim_1_seg_list += [ssim(test_forward_mean_1 * test_seg_slice, test_gt_1 * test_seg_slice) ]  # 231014
#                 ssim_2_seg_list += [ssim(test_forward_mean_2 * test_seg_slice, test_gt_2 * test_seg_slice) ]  # 231014
#                 ssim_3_seg_list += [ssim(test_forward_mean_3 * test_seg_slice, test_gt_3 * test_seg_slice) ]  # 231014
#                 ssim_seg_percentile_list += [test_seg_slice_percentile]
#             recon = test_forward_mean_1 # b c h w
#             gt = test_gt_1         # b c h w
            
#             recon = recon.contiguous().view(1,-1) # .reshape(...)#.view(batch_size, -1)
#             gt = gt.contiguous().view(1,-1)#reshape(1,...)# .view(batch_size, -1)
#             test_seg_slice = test_seg_slice.contiguous().view(1,-1)#reshape(1,...)# .view(batch_size, -1)

#             # print(f'recon - min,max: [{recon.min()}|{recon.max()}]')
#             # print(f'gt - min,max: [{gt.min()}|{gt.max()}]')

#             # mse = ((gt * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
#             rmse_1 =  (((gt - recon ) ** 2).mean(dim=-1))**(1/2)
#             rmse_1_list += [rmse_1]
            

            
#             rmse_1_seg = (rmse_1 * test_seg_slice) *  test_seg_slice_percentile  # 231014

#             nrmse_1 =  ( (((gt - recon ) ** 2).mean(dim=-1)) ** (1/2))  / ( gt.max()-gt.min() ) * 100

#             if torch.isnan(nrmse_1 ) == True:
#                 print(f'Vp-(((gt - recon ) ** 2).mean(dim=-1)) ** (1/2)):{(((gt - recon ) ** 2).mean(dim=-1)) ** (1/2)}')  
#                 print(f'( gt.max()-gt.min() ):{( gt.max()-gt.min() )}')
#             elif nrmse_1 > 1000:
#                 cnt_1 += 1
#                 print(f'1ch-{object_num}_{slice_num}')
#             elif nrmse_1 > 0:
#                 nrmse_1_list += [nrmse_1]


#             if test_seg_slice_pixels_number == 0:
#                 nrmse_1_seg = 0
#             else:
#                 nrmse_1_seg =  ( (( (test_seg_slice * (gt - recon )) ** 2).sum(dim=-1) / test_seg_slice_pixels_number) ** (1/2))  / ( gt.max()-gt.min() ) * 100  #$#$#$#$ 231016
#             # test_seg_slice_pixels_number

#             if seg_cnt_flag == True and test_seg_slice_pixels_number != 0:
#                 rmse_1_seg_list += [rmse_1_seg] # rmse_1_seg_list += [rmse_1 * test_seg_slice] *  test_seg_slice_percentile
#                 nrmse_1_seg_list += [nrmse_1_seg]
#             else:
#                 pass


#     ### Vp
#             recon = test_forward_mean_2 # b c h w
#             gt = test_gt_2         # b c h w
            
#             recon = recon.contiguous().view(1,-1) # .reshape(...)#.view(batch_size, -1)
#             gt = gt.contiguous().view(1,-1)#reshape(1,...)# .view(batch_size, -1)
#             test_seg_slice.contiguous().view(1,-1)
#             # print(f'recon - min,max: [{recon.min()}|{recon.max()}]')
#             # print(f'gt - min,max: [{gt.min()}|{gt.max()}]')
#             # mse = ((gt * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
#             rmse_2 =  (((gt - recon ) ** 2).mean(dim=-1))**(1/2)


#             rmse_2_seg = (rmse_2 * test_seg_slice) *  test_seg_slice_percentile #$#$#$#$

#             nrmse_2 =  ( (((gt - recon ) ** 2).mean(dim=-1)) ** (1/2))  / ( gt.max()-gt.min() ) * 100
#             # rmse_2_list += [rmse_2]
#             if torch.isnan(nrmse_2 ) == True:
#                 print(f'Vp-(((gt - recon ) ** 2).mean(dim=-1)) ** (1/2)):{(((gt - recon ) ** 2).mean(dim=-1)) ** (1/2)}')  
#                 print(f'( gt.max()-gt.min() ):{( gt.max()-gt.min() )}')
#             elif nrmse_2 > 1000:
#                 cnt_2 += 1
#                 print(f'1ch-{object_num}_{slice_num}')
#             elif nrmse_2 > 0:
#                 nrmse_2_list += [nrmse_2]

#             if test_seg_slice_pixels_number == 0:
#                 nrmse_2_seg = 0
#             else:
#                 nrmse_2_seg =  ( (( (test_seg_slice * (gt - recon )) ** 2).sum(dim=-1)/ test_seg_slice_pixels_number) ** (1/2))  / ( gt.max()-gt.min() ) * 100  #$#$#$#$ 231016
            


#                 # if seg_cnt_flag == True:
#                 #     rmse_2_seg_list += [rmse_2_seg] # rmse_2_seg_list += [rmse_2 * test_seg_slice] *  test_seg_slice_percentile
#                 #     nrmse_2_seg_list += [nrmse_2_seg]
#                 # else:
#                 #     pass

#             if seg_cnt_flag == True and test_seg_slice_pixels_number != 0:
#                 rmse_2_seg_list += [rmse_2_seg] # rmse_2_seg_list += [rmse_2 * test_seg_slice] *  test_seg_slice_percentile
#                 nrmse_2_seg_list += [nrmse_2_seg]


#     ### Ve
#             recon = test_forward_mean_3 # b c h w
#             gt = test_gt_3         # b c h w
            
#             recon = recon.contiguous().view(1,-1) # .reshape(...)#.view(batch_size, -1)
#             gt = gt.contiguous().view(1,-1)#reshape(1,...)# .view(batch_size, -1)

#             # print(f'recon - min,max: [{recon.min()}|{recon.max()}]')
#             # print(f'gt - min,max: [{gt.min()}|{gt.max()}]')
#             # mse = ((gt * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
#             rmse_3 = (((gt - recon ) ** 2).mean(dim=-1))**(1/2)
#             rmse_3_list += [rmse_3]

#             nrmse_3 =   ( (((gt - recon ) ** 2).mean(dim=-1))**(1/2)) / ( gt.max()-gt.min() ) * 100

#             if torch.isnan(nrmse_3) == True:
#                 print(f'Vp-(((gt - recon ) ** 2).mean(dim=-1)) ** (1/2)):{(((gt - recon ) ** 2).mean(dim=-1)) ** (1/2)}')  
#                 print(f'( gt.max()-gt.min() ):{( gt.max()-gt.min() )}')
#             elif nrmse_3 > 1000:
#                 cnt_3 += 1
#                 print(f'1ch-{object_num}_{slice_num}')
#             elif nrmse_3 > 0:
#                 nrmse_3_list += [nrmse_3]

#             rmse_3_seg = (rmse_3 * test_seg_slice) *  test_seg_slice_percentile #$#$#$#$

#             if test_seg_slice_pixels_number == 0:
#                 nrmse_3_seg = 0
#             else:
#                 nrmse_3_seg =  ( (( (test_seg_slice * (gt - recon )) ** 2).sum(dim=-1)/ test_seg_slice_pixels_number) ** (1/2))  / ( gt.max()-gt.min() ) * 100  #$#$#$#$ 231016


#             # print((( (test_seg_slice * (gt - recon )) ** 2).sum(dim=-1)/ test_seg_slice_pixels_number).shape)
#             # import pdb
#             # pdb.set_trace()


#                 # if seg_cnt_flag == True:
#                 #     rmse_3_seg_list += [rmse_3_seg] # rmse_3_seg_list += [rmse_3 * test_seg_slice] *  test_seg_slice_percentile
#                 #     nrmse_3_seg_list += [nrmse_3_seg]
#                 # else:
#                 #     pass
#             if seg_cnt_flag == True:
#                 rmse_3_seg_list += [rmse_3_seg] # rmse_3_seg_list += [rmse_3 * test_seg_slice] *  test_seg_slice_percentile
#                 nrmse_3_seg_list += [nrmse_3_seg]
#             else:
#                 pass

#             # nrmse_3_seg =  gt_seg * ( (((gt - recon ) ** 2).mean(dim=-1))**(1/2)) / ( gt.max()-gt.min() ) * 100

#             # if nrmse_1 > 1000:
#             #     cnt_1 += 1
#             #     print(f'1ch-{object_num}_{slice_num}')
#             # else:
#             #     nrmse_1_list += [nrmse_1]

#             #     if seg_count_flag = 1:
#             #         nrmse_1_seg_list += [nrmse_1_seg]
#             #         ssim_1_seg_list += [ssim_1_seg]

#             #         cnt
#             #     else:
#             #         pass
#             # if nrmse_2 > 1000:
#             #     cnt_2 += 1
#             #     print(f'2ch-{object_num}_{slice_num}')
#             # else:
#             #     nrmse_2_list += [nrmse_2]

#             #     if seg_count_flag = 1:
#             #         nrmse_1_seg_list += [nrmse_1_seg]
#             #         ssim_1_seg_list += [ssim_1_seg]

#             #     else:

#             #     rmse_2_seg_list += [rmse_2 * test_seg_slice] *  test_seg_slice_percentile
#             #     rmse_3_seg_list += [rmse_3 * test_seg_slice] *  test_seg_slice_percentile
                

#             # if nrmse_3 > 1000:
#             #     cnt_3 += 1
#             #     print(f'3ch-{object_num}_{slice_num}')
#             # else:
#             #     nrmse_3_list += [nrmse_3]

#             ssim_seg_percentile_total_list += ssim_seg_percentile_list


#             ####### 20-1 MEAN, STD map 저장하기

#         ####### 20-2 MEAN, STD slice 단위로   map 저장하기

#     ######### 1. mean ##########

#             # save_path_1 = f'{save_folder_path}/ktrans_mean.nii.gz' # ktrans
#             # save_path_2 = f'{save_folder_path}/vp_mean.nii.gz' # vp
#             # save_path_3 = f'{save_folder_path}/ve_mean.nii.gz' # ve

#             # # mean_nft_1 = nib.Nifti1Image(np.transpose(mean_stack[0]) ,refvol.affine,refvol.header)
#             # # mean_nft_2 = nib.Nifti1Image(np.transpose(mean_stack[1]) ,refvol.affine,refvol.header)
#             # # mean_nft_3 = nib.Nifti1Image(np.transpose(mean_stack[2]) ,refvol.affine,refvol.header)
#             # # print(f'mean_stack[0]:{mean_stack[0].shape}')
#             # # print(f'mean_stack[0]:{np.transpose(mean_stack[0], ).shape}')
            
#             # mean_nft_1 = nib.Nifti1Image(mean_stack[0].transpose((1,2,0)) ,refvol.affine,refvol.header ,refvol.affine,refvol.header)
#             # mean_nft_2 = nib.Nifti1Image(mean_stack[1].transpose((1,2,0)) ,refvol.affine,refvol.header)
#             # mean_nft_3 = nib.Nifti1Image(mean_stack[2].transpose((1,2,0)) ,refvol.affine,refvol.header)

#             # nib.save(mean_nft_1, save_path_1)
#             # nib.save(mean_nft_2, save_path_2)
#             # nib.save(mean_nft_3, save_path_3)

#     # ######### 2. mean per std ##########
#     #         save_path_1_spm = f'{save_folder_path}/ktrans_spm.nii.gz' # ktrans
#     #         save_path_2_spm = f'{save_folder_path}/vp_spm.nii.gz' # vp
#     #         save_path_3_spm = f'{save_folder_path}/ve_spm.nii.gz' # ve

#     #         spm_nft_1 = nib.Nifti1Image(std_per_mean_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
#     #         spm_nft_2 = nib.Nifti1Image(std_per_mean_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
#     #         spm_nft_3 = nib.Nifti1Image(std_per_mean_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

#     #         nib.save(spm_nft_1, save_path_1_spm)
#     #         nib.save(spm_nft_2, save_path_2_spm)
#     #         nib.save(spm_nft_3, save_path_3_spm)

#     # ######### 3. mean per std_seg! ##########

#     #         save_path_1_spm_seg = f'{save_folder_path}/ktrans_spm_seg.nii.gz' # ktrans
#     #         save_path_2_spm_seg = f'{save_folder_path}/vp_spm_seg.nii.gz' # vp
#     #         save_path_3_spm_seg = f'{save_folder_path}/ve_spm_seg.nii.gz' # ve

#     #         spm_nft_1 = nib.Nifti1Image(std_per_mean_seg_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
#     #         spm_nft_2 = nib.Nifti1Image(std_per_mean_seg_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
#     #         spm_nft_3 = nib.Nifti1Image(std_per_mean_seg_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

#     #         nib.save(spm_nft_1, save_path_1_spm_seg)
#     #         nib.save(spm_nft_2, save_path_2_spm_seg)
#     #         nib.save(spm_nft_3, save_path_3_spm_seg)


#     ssim_1_avg_list += ssim_1_list
#     ssim_2_avg_list += ssim_2_list
#     ssim_3_avg_list += ssim_3_list

#     rmse_1_avg_list += rmse_1_list
#     rmse_2_avg_list += rmse_2_list
#     rmse_3_avg_list += rmse_3_list

#     nrmse_1_avg_list += nrmse_1_list
#     nrmse_2_avg_list += nrmse_2_list
#     nrmse_3_avg_list += nrmse_3_list

#     ssim_1_seg_avg_list += ssim_1_seg_list
#     ssim_2_seg_avg_list += ssim_2_seg_list
#     ssim_3_seg_avg_list += ssim_3_seg_list


#     rmse_1_seg_avg_list += rmse_1_seg_list
#     rmse_2_seg_avg_list += rmse_2_seg_list
#     rmse_3_seg_avg_list += rmse_3_seg_list

#     nrmse_1_seg_avg_list += nrmse_1_seg_list
#     nrmse_2_seg_avg_list += nrmse_2_seg_list
#     nrmse_3_seg_avg_list += nrmse_3_seg_list
# ###########################################################################################
# ####################################   EVAL   #############################################
# ###########################################################################################


#     # std_per_mean_seg_stack = []
#     # std_per_mean_stack = []
#     # mean_stack = []
#     # with torch.no_grad():
#     #     std_per_mean_seg_stack = torch.cat(std_per_mean_seg_stack, dim=0)
#     #     std_per_mean_stack = torch.cat(std_per_mean_stack, dim=0)
#     #     mean_stack = torch.cat(mean_stack, dim=0)
#     # torch.stack(std_per_mean_seg_stack,dim=0)

#     # test_forward_mean_1 = torch.mean( torch.stack([test_forward_1]+[test_forward2_1], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
#     # test_forward_mean_2 = torch.mean( torch.stack([test_forward_2]+[test_forward2_2], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
#     # test_forward_mean_3 = torch.mean( torch.stack([test_forward_3]+[test_forward2_3], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
#     # test_forward_mean = torch.cat([test_forward_mean_1,test_forward_mean_2,test_forward_mean_3],dim=1)
#     # mean_stack += [test_forward_mean]

#     # test_forward_std_1 = torch.std( torch.stack([test_forward_1]+[test_forward2_1], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
#     # test_forward_std_2 = torch.std( torch.stack([test_forward_2]+[test_forward2_2], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
#     # test_forward_std_3 =  torch.std( torch.stack([test_forward_3]+[test_forward2_3], dim = 0), dim=0 ) # torch.Size([1, 1, 256, 256])
#     # # test_forward_std = torch.cat([test_forward_std_1,test_forward_std_2,test_forward_std_3],dim=1)

#     # test_forward_std_per_mean_1 = test_forward_std_1 / test_forward_mean_1 # torch.Size([1, 1, 256, 256])
#     # test_forward_std_per_mean_2 = test_forward_std_2 / test_forward_mean_2  # torch.Size([1, 1, 256, 256])
#     # test_forward_std_per_mean_3 = test_forward_std_3 / test_forward_mean_3  # torch.Size([1, 1, 256, 256])
#     # # test_forward_std_per_mean = torch.cat([test_forward_std_per_mean_1,test_forward_std_per_mean_2,test_forward_std_per_mean_3],dim=1)
#     # # std_per_mean_stack += [test_forward_std_per_mean]
    
#     # test_forward_std_per_mean_thres_1 = test_forward_std_1.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_1.shape)) / (test_forward_mean_1) # torch.Size([1, 1, 256, 256])
#     # test_forward_std_per_mean_thres_2 = test_forward_std_2.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_2.shape)) / (test_forward_mean_2) # torch.Size([1, 1, 256, 256])
#     # test_forward_std_per_mean_thres_3 = test_forward_std_3.where(test_forward_mean_1>0.01, torch.zeros(test_forward_mean_3.shape)) / (test_forward_mean_3) # torch.Size([1, 1, 256, 256])
#     # test_forward_std_per_mean_thres = torch.cat([test_forward_std_per_mean_thres_1,test_forward_std_per_mean_thres_2,test_forward_std_per_mean_thres_3],dim=1)
#     # std_per_mean_stack += [test_forward_std_per_mean_thres]

#     # test_forward_std_per_mean_thres_seg_1 = test_forward_std_per_mean_thres_1 * test_seg_slice # torch.Size([1, 1, 256, 256])   # test_seg_slice  # ... == torch.tile(test_seg_slice[0], (3,1,1))
#     # test_forward_std_per_mean_thres_seg_2 = test_forward_std_per_mean_thres_2 * test_seg_slice # torch.Size([1, 1, 256, 256])
#     # test_forward_std_per_mean_thres_seg_3 = test_forward_std_per_mean_thres_3 * test_seg_slice # torch.Size([1, 1, 256, 256])
#     # test_forward_std_per_mean_thres_seg = torch.cat([test_forward_std_per_mean_thres_seg_1,test_forward_std_per_mean_thres_seg_2,test_forward_std_per_mean_thres_seg_3],dim=1)
#     # std_per_mean_seg_stack += [test_forward_std_per_mean_thres_seg]        


#     # mean_nft_1 = nib.Nifti1Image(mean_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
#     # mean_nft_2 = nib.Nifti1Image(mean_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
#     # mean_nft_3 = nib.Nifti1Image(mean_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

#     # spm_nft_1 = nib.Nifti1Image(std_per_mean_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
#     # spm_nft_2 = nib.Nifti1Image(std_per_mean_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
#     # spm_nft_3 = nib.Nifti1Image(std_per_mean_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

#     # spm_nft_seg_1 = nib.Nifti1Image(std_per_mean_seg_stack[0].transpose((1,2,0)),refvol.affine,refvol.header)
#     # spm_nft_seg_2 = nib.Nifti1Image(std_per_mean_seg_stack[1].transpose((1,2,0)),refvol.affine,refvol.header)
#     # spm_nft_seg_3 = nib.Nifti1Image(std_per_mean_seg_stack[2].transpose((1,2,0)),refvol.affine,refvol.header)

#     # nib.save(spm_nft_1, save_path_1_spm_seg)
#     # nib.save(spm_nft_2, save_path_2_spm_seg)
#     # nib.save(spm_nft_3, save_path_3_spm_seg)

#     # save_path_1 = f'{save_folder_path}/ktrans_mean.nii.gz' # ktrans
#     # save_path_2 = f'{save_folder_path}/vp_mean.nii.gz' # vp
#     # save_path_3 = f'{save_folder_path}/ve_mean.nii.gz' # ve

# #############

#     # mean_nft_1 = nib.Nifti1Image(np.transpose(mean_stack[0]) ,refvol.affine,refvol.header)
#     # mean_nft_2 = nib.Nifti1Image(np.transpose(mean_stack[1]) ,refvol.affine,refvol.header)
#     # mean_nft_3 = nib.Nifti1Image(np.transpose(mean_stack[2]) ,refvol.affine,refvol.header)
#     # print(f'mean_stack[0]:{mean_stack[0].shape}')
#     # print(f'mean_stack[0]:{np.transpose(mean_stack[0], ).shape}')
    
#     save_path_std = f'{save_folder_path}/forwarded_forwarded_tcn_0_1_punet_230925_77777_STD_{N_idx}.nii.gz'
    
#     save_path_mean = f'{save_folder_path}/forwarded_forwarded_tcn_0_1_punet_230925_77777_MEAN_{N_idx}.nii.gz'
#     # save_path_1_mean = f'{save_folder_subfolder_path}/ktrans_mean_{N_idx}.nii.gz' # ktrans
#     # save_path_2_mean = f'{save_folder_subfolder_path}/vp_mean_{N_idx}.nii.gz' # vp
#     # save_path_3_mean = f'{save_folder_subfolder_path}/ve_mean_{N_idx}.nii.gz' # ve
#     save_path_spm = f'{save_folder_path}/forwarded_forwarded_tcn_0_1_punet_230925_77777_SPM_{N_idx}.nii.gz'
#     # save_path_1_spm = f'{save_folder_subfolder_path}/ktrans_spm_{N_idx}.nii.gz' # ktrans
#     # save_path_2_spm = f'{save_folder_subfolder_path}/vp_spm_{N_idx}.nii.gz' # vp
#     # save_path_3_spm = f'{save_folder_subfolder_path}/ve_spm_{N_idx}.nii.gz' # ve
#     save_path_spm_seg = f'{save_folder_path}/forwarded_forwarded_tcn_0_1_punet_230925_77777_SPM_SEG_{N_idx}.nii.gz'

    # save_path_mean = f'{save_folder_path}/forwarded_forwarded_tcn_0_1_punet_230925_77777_MEAN_{N_idx}.nii.gz'
    # save_path_1_mean = f'{save_folder_subfolder_path}/ktrans_mean_{N_idx}.nii.gz' # ktrans
    # save_path_2_mean = f'{save_folder_subfolder_path}/vp_mean_{N_idx}.nii.gz' # vp
    # save_path_3_mean = f'{save_folder_subfolder_path}/ve_mean_{N_idx}.nii.gz' # ve


    idx = 'g_fixed_clipped_0_1'
    g_save_path_mean = f'{save_folder_path}/g_kvpve_fixed_clipped_0_1.nii.gz'
    g_save_path_1_mean = f'{save_folder_subfolder_path}/ktrans_mean_{idx}.nii.gz' # ktrans
    g_save_path_2_mean = f'{save_folder_subfolder_path}/vp_mean_{idx}.nii.gz' # vp
    g_save_path_3_mean = f'{save_folder_subfolder_path}/ve_mean_{idx}.nii.gz' # ve


    # with torch.no_grad():
        # std_per_mean_seg_stack = torch.cat(std_per_mean_seg_stack, dim=0)
        # std_per_mean_stack = torch.cat(std_per_mean_stack, dim=0)
        # mean_stack = torch.cat(mean_stack, dim=0)
        # std_stack = torch.cat(std_stack, dim=0)

    g_mean_stack = torch.cat(g_mean_stack, dim = 0)
    g_mean_stack_1 = torch.cat(g_mean_stack_1, dim = 0)
    g_mean_stack_2 = torch.cat(g_mean_stack_2, dim = 0)
    g_mean_stack_3 = torch.cat(g_mean_stack_3, dim = 0)

    # g_mean_stack_1 += [test_gt_1]
    # g_mean_stack_2 += [test_gt_2]
    # g_mean_stack_3 += [test_gt_3]



    g_mean_stack = rearrange(g_mean_stack, 's c h w -> h w s c')
    
    # std_per_mean_seg_stack = rearrange(std_per_mean_seg_stack, 's c h w -> h w s c')
    # std_per_mean_stack = rearrange(std_per_mean_stack, 's c h w -> h w s c')
    # mean_stack = rearrange(mean_stack, 's c h w -> h w s c')
    # std_stack = rearrange(std_stack, 's c h w -> h w s c')

    # mean_nft_1 = nib.Nifti1Image(mean_stack[0].transpose((1,2,0)) ,refvol.affine,refvol.header ,refvol.affine,refvol.header)
    # mean_nft_2 = nib.Nifti1Image(mean_stack[1].transpose((1,2,0)) ,refvol.affine,refvol.header)
    # mean_nft_3 = nib.Nifti1Image(mean_stack[2].transpose((1,2,0)) ,refvol.affine,refvol.header)
    
    # std_seg_nft = nib.Nifti1Image(std_seg_stack,refvol.affine,refvol.header)
    # std_per_mean_seg_nft = nib.Nifti1Image(std_per_mean_seg_stack,refvol.affine,refvol.header)
    # std_per_mean_nft = nib.Nifti1Image(std_per_mean_stack,refvol.affine,refvol.header)
    # mean_nft = nib.Nifti1Image(mean_stack,refvol.affine,refvol.header)
    # std_nft = nib.Nifti1Image(std_stack,refvol.affine,refvol.header)
    
    # g_mean_nft = nib.Nifti1Image(g_mean_stack,refvol.affine,refvol.header)
    # nib.save(g_mean_nft, g_save_path_mean)



    # nib.save(std_nft, save_path_std)
    # nib.save(mean_nft, save_path_mean)
    # nib.save(std_per_mean_nft, save_path_spm)
    # nib.save(std_per_mean_seg_nft, save_path_spm_seg)

    # save_path_1_mean = f'{save_folder_subfolder_path}/ktrans_mean_{N_idx}.nii.gz' # ktrans
    # save_path_2_mean = f'{save_folder_subfolder_path}/vp_mean_{N_idx}.nii.gz' # vp
    # save_path_3_mean = f'{save_folder_subfolder_path}/ve_mean_{N_idx}.nii.gz' # ve

    # import pdb
    # pdb.set_trace()
    
    # mean_stack_PK = mean_stack[:,:,:,0] # 256 256 40 3
    # mean_nft = nib.Nifti1Image(mean_stack_PK,refvol.affine,refvol.header)
    # save_path_mean = f'{save_folder_path}/ktrans_mean_{N_idx}.nii.gz'
    # nib.save(mean_nft, save_path_mean)

    # mean_stack_PK = mean_stack[:,:,:,1] # 256 256 40 3
    # mean_nft = nib.Nifti1Image(mean_stack_PK,refvol.affine,refvol.header)
    # save_path_mean = f'{save_folder_path}/vp_mean_{N_idx}.nii.gz'
    # nib.save(mean_nft, save_path_mean)

    # mean_stack_PK = mean_stack[:,:,:,2] # 256 256 40 3
    # mean_nft = nib.Nifti1Image(mean_stack_PK,refvol.affine,refvol.header)
    # save_path_mean = f'{save_folder_path}/ve_mean_{N_idx}.nii.gz'
    # nib.save(mean_nft, save_path_mean)

    g_mean_stack_PK = g_mean_stack[:,:,:,0] # 256 256 40 3
    g_mean_nft = nib.Nifti1Image(g_mean_stack_PK,refvol.affine,refvol.header)
    save_path_mean = f'{save_folder_path}/ktrans_mean_{idx}.nii.gz'
    nib.save(g_mean_nft, save_path_mean)

    g_mean_stack_PK = g_mean_stack[:,:,:,1] # 256 256 40 3
    g_mean_nft = nib.Nifti1Image(g_mean_stack_PK,refvol.affine,refvol.header)
    save_path_mean = f'{save_folder_path}/vp_mean_{idx}.nii.gz'
    nib.save(g_mean_nft, save_path_mean)

    g_mean_stack_PK = g_mean_stack[:,:,:,2] # 256 256 40 3
    g_mean_nft = nib.Nifti1Image(g_mean_stack_PK,refvol.affine,refvol.header)
    save_path_mean = f'{save_folder_path}/ve_mean_{idx}.nii.gz'
    nib.save(g_mean_nft, save_path_mean)



# nib.Nifti1Image(mean_stack[2].transpose((1,2,0)) ,refvol.affine,refvol.header)


        # del ssim_1_list, ssim_2_list, ssim_3_list, rmse_1_list, rmse_2_list, rmse_3_list,  nrmse_1_list, nrmse_2_list, nrmse_3_list, ssim_1_seg_list

        # if ii =='4': 
        #     import pdb
        #     pdb.set_trace() 

        # nrmse_1_avg_list.shape

        ############################

    #     x = x.view(batch_size, -1)                             
    #     recon = recon.view(batch_size, -1)

    #     mse = ((x * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
    #     psnr = (-10 * torch.log10(mse)).mean()

    #     losses['psnr'].update(psnr.item(), batch_size)

    
    ###
    # torch.save()
    # KVpVe
    # test_forward_std_1.save('ktrans_mean.nii.gz')
    # test_forward_std_2.save('vp_mean.nii.gz')
    # test_forward_std_3.save('ve_mean.nii.gz')

    # '/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_84_inference/31781365/slice_00'
    # '/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_84_inference/31781365'

    # import pdb
    # pdb.set_trace()

    # print(f'\n#object_num: {object_num}')

    # # print(f'\n#slice_{ii}')
    # print( f' 1ktrans - SSIM: {sum(ssim_1_list)/len(ssim_1_list)} / NRMSE: {sum(nrmse_1_list)/len(nrmse_1_list)} / Counts: non_zeros:{256*256-counts_zeors_1} |\n Vp: {sum(ssim_2_list)/len(ssim_2_list)} / NRMSE: {sum(nrmse_2_list)/len(nrmse_2_list)}  / Counts: non_zeros:{256*256-counts_zeors_2}  | Ve: {sum(ssim_3_list)/len(ssim_3_list)} / NRMSE: {sum(nrmse_3_list)/len(nrmse_3_list)}  / Counts: non_zeros:{256*256-counts_zeors_3} ')
    # print(f'{sum(nrmse_1_list)/len(nrmse_1_list)}  / {sum(nrmse_2_list)/len(nrmse_2_list)}  /  {sum(nrmse_3_list)/len(nrmse_3_list)}'
    # if ii =='39':
    #     pass
    # else:

    # try:
    #     print( f' SEG_CNT: {seg_cnt} ... - \
    #         1ktrans - SSIM_tumor_seg: {sum(ssim_1_seg_list)/len(ssim_1_seg_list)} | NRMSE_tumor_seg: {sum(nrmse_1_seg_list)/len(sum(nrmse_1_seg_list))} | Counts_seg > 0.001 (len(ssim_1_seg_list) : {len(ssim_1_seg_list)}|\n\
    #         2Vp_tumor_seg: {sum(ssim_2_seg_list)/len(ssim_2_seg_list)} | NRMSE_tumor_seg: {sum(nrmse_2_seg_list)/len(nrmse_2_seg_list)} | Counts_seg > 0.001 (len(ssim_2_seg_list): {len(nrmse_2_seg_list)}|\n\
    #         \
    #         3Ve_tumor_seg: {sum(ssim_3_seg_list)/len(ssim_3_seg_list)} | NRMSE:_tumor_seg {sum(nrmse_3_seg_list)/len(nrmse_3_seg_list)}  | len(ssim_3_seg_list):{len(ssim_3_seg_list)} ')
    # except:
    #     print(f'#{object_num}: no seg_list !!!')
    # print(f'ssim_seg_percentile_list: {ssim_seg_percentile_list}')

    # nrmse_3_seg_list    # torch.Size([1, 1, 256, 256]) # 0~8 9개
# sum(nrmse_2_seg_list)/len(nrmse_2_seg_list)

# print( f' SEG_CNT: {seg_cnt} ... - 1ktrans - SSIM_tumor_seg: {sum(ssim_1_seg_list)/len(ssim_1_seg_list)} | NRMSE_tumor_seg: {sum(nrmse_1_seg_list)/len(sum(nrmse_1_seg_list))} | Counts_seg > 0.001 (len(ssim_1_seg_list) : {len(ssim_1_seg_list)}|\n 2Vp_tumor_seg: {sum(ssim_2_seg_list)/len(ssim_2_seg_list)} | NRMSE_tumor_seg: {sum(nrmse_2_seg_list)/len(nrmse_2_seg_list)} | Counts_seg > 0.001 (len(ssim_2_seg_list): {len(nrmse_2_seg_list)}|\n 3Ve_tumor_seg: {sum(ssim_3_seg_list)/len(ssim_3_seg_list)} | NRMSE:_tumor_seg {sum(nrmse_3_seg_list)/len(nrmse_3_seg_list)}  | len(ssim_3_seg_list):{len(ssim_3_seg_list)} ')

import pdb
pdb.set_trace()

print( sum(rmse_1_list)/len(rmse_1_list) ) # tensor([0.0011])
print( sum(rmse_2_list)/len(rmse_2_list) ) # tensor([0.0010])
print( sum(rmse_3_list)/len(rmse_3_list) ) # tensor([0.0113])


print( sum(nrmse_1_list)/len(nrmse_1_list) )
print( sum(nrmse_2_list)/len(nrmse_2_list) )
print( sum(nrmse_3_list)/len(nrmse_3_list) )



print( sum(ssim_1_avg_list)/len(ssim_1_avg_list) )
print( sum(ssim_2_avg_list)/len(ssim_2_avg_list) )
print( sum(ssim_3_avg_list)/len(ssim_3_avg_list) )

print( sum(rmse_1_avg_list)/len(rmse_1_avg_list) )
print( sum(rmse_2_avg_list)/len(rmse_2_avg_list) )
print( sum(rmse_3_avg_list)/len(rmse_3_avg_list) )
rmse_1_avg_list

print( sum(nrmse_1_avg_list)/len(nrmse_1_avg_list) )
print( sum(nrmse_2_avg_list)/len(nrmse_2_avg_list) )
print( sum(nrmse_3_avg_list)/len(nrmse_3_avg_list) )


# nrmse_3_seg_avg_list= nrmse_3_seg_avg_list - [torch.Tensor(float('inf'))]

print( sum(nrmse_1_seg_avg_list)/len(nrmse_1_seg_avg_list) )
print( sum(nrmse_2_seg_avg_list)/len(nrmse_2_seg_avg_list) )
print( sum(nrmse_3_seg_avg_list)/len(nrmse_3_seg_avg_list) )


print( sum(ssim_1_seg_avg_list)/len(ssim_1_seg_avg_list) )
print( sum(ssim_2_seg_avg_list)/len(ssim_2_seg_avg_list) )
print( sum(ssim_3_seg_avg_list)/len(ssim_3_seg_avg_list) )
# ssim_1_avg_list

    # 256*256-counts_zeors_1
    # 256*256-counts_zeors_3
    # 256*256-counts_zeors_3
    # counts_zeors_1 = test_gt_1_counts[1][0]
    # counts_zeors_2 = test_gt_2_counts[1][0]
    # counts_zeors_3 = test_gt_3_counts[1][0]


    # print(sum(ssim_2_list)/len(ssim_2_list))
    # print(sum(ssim_3_list)/len(ssim_3_list))

## slice_00

# ssim_1_list += [ssim(test_forward_1, test_gt_1)]
# ssim_2_list += [ssim(test_forward_2, test_gt_2)]
# ssim_3_list += [ssim(test_forward_3, test_gt_3)]
# /mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_73_inference/47013656/slice_00/slice_forwarded_forwarded_tcn_0_1_punet_230919.nii.gz

ssim_list


# test_brain_mask[0,...].unsqueeze(0)