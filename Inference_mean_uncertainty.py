# Import necessary libraries
from functools import total_ordering
from operator import index
import os
import numpy as np
from datetime import datetime
from glob import glob
import random
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import copy
import pandas as pd
from einops import rearrange
from einops.layers.torch import Rearrange
from pytorch_ssim import SSIM
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from utils_ylyoo import (nii_to_numpy, visualize_ktrans_patch, visualize_ktrans_final, 
                         visualize_ktrans_final_save, visualize_ktrans_final_save_extend, 
                         Weighted_L1_patch, FixedRandCropByPosNegLabel_custom_coordinates, 
                         RandCropByPosNegLabel_custom_coordinates, apply_patch_coordinates_4d, 
                         apply_patch_coordinates, apply_patch_coordinates_ch1, TemporalConvNet_custom, 
                         Wrapper_Net_fixed, Wrapper_integrate_Net, format_4f, nii_to_numpy, 
                         path_cutter, mriview, visualize_ktrans_patch_30, visualize_ktrans_patch_30_uncertainty, 
                         visualize_ktrans_compare, nonvisualize_ktrans_compare, TemporalBlock, tile_3d_to_60_torch)

# Set path
base_path = "/mnt/hdd/unist/helee/dsc-to-ktrans/preprocessing/230726"
data_path = "/mnt/ssd/ylyoo/intermediate_filtered_split"


# Set environment variables to limit CPU usage
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch.cuda.is_available:', torch.cuda.is_available())
print('torch.cuda.device_count():', torch.cuda.device_count())
print('torch.cuda.current_device():', torch.cuda.current_device())
print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0))

# Function to display and save image slices
def show_and_save_img_3output_slice(subject_number='1', objects_num=1, slice_num=1, ch=1, label=1, seg=1, mean=1, std=1, std_per_mean=1, std_per_mean_seg=1, show_flag=False, label_name='KtransVpVe'):
    global folder_path
    title_list = [f'{label_name}_label', f'{label_name}_seg', f'{label_name}_std/mean_seg' , f'{label_name}_mean', f'{label_name}_std', f'{label_name}_std/mean']
    w = 256
    h = 256
    columns = 6
    img_list = [label, seg, std_per_mean_seg, mean, std, std_per_mean]
    rows = 1
    fig = plt.figure(figsize=(6*columns, 6*rows))
    for i in range(1, columns*rows +1):
        img = img_list[(i-1)%6]
        fig.add_subplot(rows, columns, i)
        if (i-1)//6 == 0:
            plt.title(title_list[i-1], fontsize=20)
        if i in [3, 6]:
            plt.imshow(img, cmap='jet', vmin=0., vmax=1.)
        else:
            plt.imshow(img, cmap='jet', vmin=0., vmax=0.1)
        plt.axis('off')
    visuaialization_slice_file_path = f'{base_path}/231101_figs_show/visualization/{subject_number}_{str(objects_num)}_{str(slice_num)}_{str(ch)}_{label_name}_TCN-PUnet.png'
    if not os.path.exists('/'.join(visuaialization_slice_file_path.split('/')[:-1])):
        os.makedirs('/'.join(visuaialization_slice_file_path.split('/')[:-1]))
    plt.savefig(visuaialization_slice_file_path, bbox_inches='tight', pad_inches=0)
    if show_flag:
        plt.show()
    plt.close()

# Sample image for demonstration
img = np.random.randint(2, size=(256, 256))
show_and_save_img_3output_slice(objects_num=1, slice_num=1, ch=1, label=copy.deepcopy(img), seg=copy.deepcopy(img), mean=copy.deepcopy(img), std=copy.deepcopy(img), std_per_mean=copy.deepcopy(img), std_per_mean_seg=copy.deepcopy(img), show_flag=True, label_name='ktrans')

# Function to create dataset paths for NIfTI files
def make_nii_path_dataset(data_dir, input_filename, label_filename, seg_filename, mask_filename, ratio_select=1):
    path_segs = sorted(glob.glob(data_dir + "/*/" + seg_filename))
    path_images = [x.replace(seg_filename, input_filename) for x in path_segs]
    path_labels = [x.replace(seg_filename, label_filename) for x in path_segs]
    path_masks = [x.replace(seg_filename, mask_filename) for x in path_segs]
    path_dataset = [{"image": image_name, "label": label_name, "seg": seg_name, "mask": mask_name, "path": image_name[:-len(input_filename)-1]} for image_name, label_name, seg_name, mask_name in zip(path_images, path_labels, path_segs, path_masks)]
    if ratio_select != 1:
        num_select = int(ratio_select * len(path_dataset))
        path_dataset = path_dataset[:num_select]
    print(f"Created path dataset of size: {len(path_dataset)} from {data_dir}")
    print(f"Containing input: {input_filename}, label: {label_filename}, seg: {seg_filename}, mask: {mask_filename}")
    return path_dataset

# Initialize SSIM
ssim = SSIM()

# Initialize various lists to hold evaluation metrics
ssim_avg_list, ssim_1_avg_list, ssim_2_avg_list, ssim_3_avg_list = [], [], [], []
mse_list, rmse_1_avg_list, rmse_2_avg_list, rmse_3_avg_list = [], [], [], []
nrmse_1_avg_list, nrmse_2_avg_list, nrmse_3_avg_list = [], [], []
total_rmse_1_list, total_rmse_2_list, total_rmse_3_list = [], [], []
total_nrmse_1_list, total_nrmse_2_list, total_nrmse_3_list = [], [], []

# Initialize lists for saving evaluation metrics
ssim_list, ssim_1_avg_list, ssim_2_avg_list, ssim_3_avg_list = [], [], [], []
rmse_1_avg_list, rmse_2_avg_list, rmse_3_avg_list = [], [], []
nrmse_1_avg_list, nrmse_2_avg_list, nrmse_3_avg_list = [], [], []
ssim_1_seg_avg_list, ssim_2_seg_avg_list, ssim_3_seg_avg_list = [], [], []
rmse_1_seg_avg_list, rmse_2_seg_avg_list, rmse_3_seg_avg_list = [], [], []
nrmse_1_seg_avg_list, nrmse_2_seg_avg_list, nrmse_3_seg_avg_list = [], [], []
ssim_seg_percentile_total_list = []

# Define pass list
pass_tuple_list = [('50148231', 'slice_00'), ('48262855', 'slice_00'), ('20252076', 'slice_00'), ('40892562', 'slice_00'), ('50935567', 'slice_00'), ('40892562', 'slice_01'), ('40892562', 'slice_02'), ('40892562', 'slice_03'), ('40892562', 'slice_03'), ('40892562', 'slice_04'), ('48505615', 'slice_36'), ('48505615', 'slice_37'), ('29899179', 'slice_38'), ('29899179', 'slice_39'), ('50225501', 'slice_39'), ('50275847', 'slice_39'), ('47983337', 'slice_39'), ('51390840', 'slice_39'), ('48284196', 'slice_39'), ('48505615', '_slice_39')]
pass_list = []

# Load object paths
object_num_path_list = glob(f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/*')
object_num_list = [path.split('/')[-1] for path in object_num_path_list if os.path.isdir(path)]
object_num_list.sort()

# Process each object
for object_num in object_num_list:
    g_mean_stack_1 = []
    g_mean_stack_2 = [] #[test_gt_2]
    g_mean_stack_3 = [] #[test_gt_3]
    g_mean_stack = []

    test_files_full = glob(f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat0.nii.gz')
    test_files_full2 = glob(f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat1.nii.gz')
    test_files_full21 = glob(f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat6.nii.gz')
    test_files_full22 = glob(f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_83_inference/{object_num}/*/slice_forwarded_forwarded_tcn_0_1_punet_230925_77777_repeat7.nii.gz')
    
    # Ensure lists are sorted
    test_files_full.sort()
    test_files_full2.sort()
    test_files_full21.sort()
    test_files_full22.sort()

    # Initialize lists for evaluation metrics for this object
    ssim_list, ssim_1_list, ssim_2_list, ssim_3_list = [], [], [], []
    rmse_1_list, rmse_2_list, rmse_3_list = [], [], []
    nrmse_1_list, nrmse_2_list, nrmse_3_list = [], [], []
    ssim_seg_percentile_list = []

    ssim_1_seg_list, ssim_2_seg_list, ssim_3_seg_list = [], [], []
    rmse_1_seg_list, rmse_2_seg_list, rmse_3_seg_list = [], [], []
    nrmse_1_seg_list, nrmse_2_seg_list, nrmse_3_seg_list = [], [], []

    # slice 마다 결정하는 것임*
    seg_cnt_flag = False # seg_percentil이 일정 이상 넘겼을 때 발동하게 하기 ?  seg_percentil > 0.01~0.03 
    seg_cnt = 0 # 유효한 seg 갯수를 내서 나중에 평균 내줄 것임

    # Process each slice for this object
    for i, test_file_path in enumerate(test_files_full):
        with torch.no_grad():
            slice_num = test_file_path.split('/')[-2]
            slice_num_int = int(slice_num.split('_')[-1])
            test_gt_path = f'{data_path}/test/{object_num}/{slice_num}/g_kvpve_fixed_clipped_0_1.nii.gz'
            test_brain_mask_path = f'{data_path}/test/{object_num}/g_dce_mask.nii.gz'
            test_seg_path = f'{data_path}/test/{object_num}/g_seg.nii.gz'
            refvol = nib.load(f'{data_path}/test/{object_num}/g_ktrans.nii.gz')

            # Skip if in pass list
            if (object_num, slice_num) in pass_tuple_list:
                pass_list.append((object_num, slice_num))
                continue

            # Load ground truth and segmentation data
            test_gt = torch.Tensor(nii_to_numpy(test_gt_path))   
            test_gt = rearrange(test_gt, 'h w b c -> b c h w') # torch.Size([1, 3, 256, 256])
            
            test_seg = torch.Tensor(nii_to_numpy(test_seg_path))         # torch.Size([256, 256, 40])
            test_seg = rearrange(test_seg, 'h w (s c) -> s c h w', s=40) # torch.Size([40, 3, 256, 256])
            
            test_seg_slice = test_seg[slice_num_int,...].unsqueeze(0)    # torch.Size([1, 1, 256, 256])
            test_seg_slice = torch.where( test_seg_slice > 0., test_seg_slice, torch.zeros(test_seg_slice.shape) ) #  test_seg_slice > 0.03
            
            test_seg_slice_pixels_number = (256**2) - torch.unique(test_seg_slice, return_counts=True)[1][0]
            test_seg_slice_percentile = (test_seg_slice_pixels_number) / (256**2) # nonzeors number / total number


            # test_gt = torch.Tensor(nii_to_numpy(test_gt_path)).permute(2, 3, 0, 1)

            # test_seg = torch.Tensor(nii_to_numpy(test_seg_path))         # torch.Size([256, 256, 40])
            # test_seg_slice = test_seg[slice_num_int, ...].unsqueeze(0)

            test_brain_mask = torch.Tensor(nii_to_numpy(test_brain_mask_path))
            test_brain_mask = rearrange(test_brain_mask, 'h w (s c) -> s c h w', s= 40)
            test_brain_mask_slice = test_brain_mask[slice_num_int, ...].unsqueeze(0)

            # Update segmentation flag
            if test_seg_slice_percentile > 0.00005:
                seg_cnt_flag = True
                seg_cnt += 1
            else:
                seg_cnt_flag = False


            test_gt_1, test_gt_2, test_gt_3 = test_gt[:, 0, :, :].unsqueeze(1), test_gt[:, 1, :, :].unsqueeze(1), test_gt[:, 2, :, :].unsqueeze(1)
            counts_zeors_1, counts_zeors_2, counts_zeors_3 = torch.unique(test_gt_1, return_counts=True)[1][0], torch.unique(test_gt_2, return_counts=True)[1][0], torch.unique(test_gt_3, return_counts=True)[1][0]

            # Append ground truth data to respective lists
            g_mean_stack_1.append(test_gt_1)
            g_mean_stack_2.append(test_gt_2)
            g_mean_stack_3.append(test_gt_3)

            # Skip processing if in pass list
            if (object_num, slice_num) in pass_tuple_list:
                pass_list.append((object_num, slice_num))
                continue

            # Calculate SSIM and RMSE metrics
            # Code for calculating SSIM and RMSE metrics goes here (omitted for brevity)

    # Save ground truth data
    g_mean_stack = torch.cat(g_mean_stack, dim=0).permute(2, 3, 0, 1)
    g_mean_stack_1 = torch.cat(g_mean_stack_1, dim=0)
    g_mean_stack_2 = torch.cat(g_mean_stack_2, dim=0)
    g_mean_stack_3 = torch.cat(g_mean_stack_3, dim=0)

    save_folder_path = f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_100_inference/{object_num}'
    save_folder_subfolder_path = f'{base_path}/tb_logs_dce_punet_2309/full3d_lightning_1tcn_2punet/version_100_inference/{object_num}/{slice_num}'

    # Save NIfTI files
    g_save_path_mean = f'{save_folder_path}/g_kvpve_fixed_clipped_0_1.nii.gz'
    g_save_path_1_mean = f'{save_folder_subfolder_path}/ktrans_mean_g_fixed_clipped_0_1.nii.gz'
    g_save_path_2_mean = f'{save_folder_subfolder_path}/vp_mean_g_fixed_clipped_0_1.nii.gz'
    g_save_path_3_mean = f'{save_folder_subfolder_path}/ve_mean_g_fixed_clipped_0_1.nii.gz'

    g_mean_nft = nib.Nifti1Image(g_mean_stack, refvol.affine, refvol.header)
    nib.save(g_mean_nft, g_save_path_mean)
    g_mean_nft_1 = nib.Nifti1Image(g_mean_stack[:, :, :, 0], refvol.affine, refvol.header)
    nib.save(g_mean_nft_1, g_save_path_1_mean)
    g_mean_nft_2 = nib.Nifti1Image(g_mean_stack[:, :, :, 1], refvol.affine, refvol.header)
    nib.save(g_mean_nft_2, g_save_path_2_mean)
    g_mean_nft_3 = nib.Nifti1Image(g_mean_stack[:, :, :, 2], refvol.affine, refvol.header)
    nib.save(g_mean_nft_3, g_save_path_3_mean)

# Print summary of results
import pdb
pdb.set_trace()
print(sum(rmse_1_list)/len(rmse_1_list))
print(sum(rmse_2_list)/len(rmse_2_list))
print(sum(rmse_3_list)/len(rmse_3_list))
print(sum(nrmse_1_list)/len(nrmse_1_list))
print(sum(nrmse_2_list)/len(nrmse_2_list))
print(sum(nrmse_3_list)/len(nrmse_3_list))
print(sum(ssim_1_avg_list)/len(ssim_1_avg_list))
print(sum(ssim_2_avg_list)/len(ssim_2_avg_list))
print(sum(ssim_3_avg_list)/len(ssim_3_avg_list))
print(sum(rmse_1_avg_list)/len(rmse_1_avg_list))
print(sum(rmse_2_avg_list)/len(rmse_2_avg_list))
print(sum(rmse_3_avg_list)/len(rmse_3_avg_list))
print(sum(nrmse_1_avg_list)/len(nrmse_1_avg_list))
print(sum(nrmse_2_avg_list)/len(nrmse_2_avg_list))
print(sum(nrmse_3_avg_list)/len(nrmse_3_avg_list))
print(sum(nrmse_1_seg_avg_list)/len(nrmse_1_seg_avg_list))
print(sum(nrmse_2_seg_avg_list)/len(nrmse_2_seg_avg_list))
print(sum(nrmse_3_seg_avg_list)/len(nrmse_3_seg_avg_list))
print(sum(ssim_1_seg_avg_list)/len(ssim_1_seg_avg_list))
print(sum(ssim_2_seg_avg_list)/len(ssim_2_seg_avg_list))
print(sum(ssim_3_seg_avg_list)/len(ssim_3_seg_avg_list))
