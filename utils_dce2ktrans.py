############## ver.2023009:
# def nonvisualize_ktrans_compare_new(y,y_hat,vmax=0.1): # y,y_hat이 들어오면.. 
# def visualize_ktrans_patch_30_uncertainty(x,y,y_hat, seg, uncertainty_map, vmax=0.1):
#     fig, axes = plt.subplots(3, 2, figsize=(15,10))
##############

import os
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=2
from functools import total_ordering
import numpy as np
import glob
import random
from posixpath import split
import nibabel as nib
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(8)
import torch.nn as nn
from PIL import Image
from torch.nn.utils import weight_norm

def format_4f(floatnum):
    return f'{floatnum:.4f}' # 소수점 뒤의 4자리에서 자름.

def normalize_to_1_1(numpy_array):
    # normalize to [-1,1] for unet tanh output layer
    if numpy_array.max()!= numpy_array.min():
        numpy_array_normalized = (numpy_array -numpy_array.min())*(2/(numpy_array.max()- numpy_array.min())) -1
        return numpy_array_normalized
    else:
        raise ValueError("normalize to -1~1 failed because data has no variance")

def nii_to_numpy(input_path):
    input_vol_raw = nib.load(input_path)
    input_vol = np.array(input_vol_raw.dataobj)
    return input_vol

def view_array(numpy_array,cmap='jet',vmin=0,vmax=0):
    if vmin!=0 or vmax!=0:
        plt.imshow(numpy_array, cmap=cmap,vmin=vmin,vmax=vmax)
    else:
        plt.imshow(numpy_array, cmap=cmap)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def mriview(numpy_array,cmap='jet',vmin=0,vmax=0):
    import math
    print("dimensions:",numpy_array.shape)
    print("type:", type(numpy_array))
    if numpy_array.ndim==2:
        selected_array = numpy_array
    if numpy_array.ndim==3:          # 3d volume의 중간 slice선택
        mid = math.floor(numpy_array.shape[2]/2)
        selected_array = numpy_array[:,:,mid]
        print("selected mid slice :", mid)
    if numpy_array.ndim==4:
        mid1 = math.floor(numpy_array.shape[2]/2)
        mid2 = math.floor(numpy_array.shape[3]/2)
        selected_array = numpy_array[:,:,mid1,mid2]
        print("selected mid slice, mid slice :", mid1,mid2)
    view_array(selected_array,cmap=cmap,vmin=vmin,vmax=vmax)

def mriview_save(numpy_array,cmap='jet',vmin=0,vmax=0):
    import math
    print("dimensions:",numpy_array.shape)
    print("type:", type(numpy_array))
    if numpy_array.ndim==2:
        selected_array = numpy_array
    if numpy_array.ndim==3:          # 3d volume의 중간 slice선택
        mid = math.floor(numpy_array.shape[2]/2)
        selected_array = numpy_array[:,:,mid]
        print("selected mid slice :", mid)
    if numpy_array.ndim==4:
        mid1 = math.floor(numpy_array.shape[2]/2)
        mid2 = math.floor(numpy_array.shape[3]/2)
        selected_array = numpy_array[:,:,mid1,mid2]
        print("selected mid slice, mid slice :", mid1,mid2)
    if vmin!=0 or vmax!=0:
        plt.imshow(selected_array, cmap=cmap,vmin=vmin,vmax=vmax)
    else:
        plt.imshow(selected_array, cmap=cmap)
    plt.axis('off')
    random_num = str(np.random.randint(99999999))
    plt.savefig(random_num+'temp.png',bbox_inches='tight')
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    os.remove(random_num+'temp.png')
    plt.show()
    return figure_np_3channel


def visualize_ktrans_patch(x,y,y_hat,vmax=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    im1 = axes[0,0].imshow(y,cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im1,ax=axes[0,0])
    axes[0,0].set_title(" Label")
    im2 = axes[0,1].imshow(y_hat,cmap = 'jet',vmin=0,vmax=vmax)
    axes[0,1].set_title("Output_DL_w_same_colormap")
    plt.colorbar(im2,ax=axes[0,1])
    im3 = axes[1,0].imshow(x,cmap = 'jet')
    axes[1,0].set_title("Input")
    plt.colorbar(im3,ax=axes[1,0])
    im4 = axes[1,1].imshow(y_hat,cmap = 'jet')
    plt.colorbar(im4,ax=axes[1,1])
    criterion = Custom_L1()
    axes[1,1].set_title("OUTPUT_DL")   
    random_num = str(np.random.randint(99999999))
    plt.savefig(random_num+'temp.png',bbox_inches='tight')
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    os.remove(random_num+'temp.png')
    plt.show()
    return figure_np_3channel
    

def visualize_ktrans_patch_30(x,y,y_hat,vmax=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    im1 = axes[0,0].imshow(y,cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im1,ax=axes[0,0])
    axes[0,0].set_title(" Label")
    im2 = axes[0,1].imshow(y_hat,cmap = 'jet',vmin=0,vmax=vmax)
    axes[0,1].set_title("Output_DL_w_same_colormap")
    plt.colorbar(im2,ax=axes[0,1])
    im3 = axes[1,0].imshow(x,cmap = 'bwr',vmin=-0.5,vmax=0.5)
    axes[1,0].set_title("Input")
    plt.colorbar(im3,ax=axes[1,0])
    im4 = axes[1,1].imshow(y_hat,cmap = 'jet')
    plt.colorbar(im4,ax=axes[1,1])
    criterion = Custom_L1()
    axes[1,1].set_title("OUTPUT_DL")   
    random_num = str(np.random.randint(99999999))
    plt.savefig(random_num+'temp.png',bbox_inches='tight')
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    os.remove(random_num+'temp.png')
    plt.show()
    return figure_np_3channel
    



def visualize_ktrans_patch_30_uncertainty(x,y,y_hat, seg, uncertainty_map, vmax=0.1):
    fig, axes = plt.subplots(3, 2, figsize=(15,10))
    im1 = axes[0,0].imshow(y,cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im1,ax=axes[0,0])
    axes[0,0].set_title(" Label")
    im2 = axes[0,1].imshow(y_hat,cmap = 'jet',vmin=0,vmax=vmax)
    axes[0,1].set_title("Output_DL_w_same_colormap")
    plt.colorbar(im2,ax=axes[0,1])
    im3 = axes[1,0].imshow(x,cmap = 'jet')
    axes[1,0].set_title("Input") # 지금은 relaxivity map이 아니라 DCE input 이므로 cmap bwr이 아니라 jet로 똑같이 맞춰주기! 
    plt.colorbar(im3,ax=axes[1,0])
    im4 = axes[1,1].imshow(y_hat,cmap = 'jet')
    plt.colorbar(im4,ax=axes[1,1])
    axes[1,1].set_title("OUTPUT_DL")   

    im5 = axes[2,0].imshow(seg,cmap = 'jet', vmin=0, vmax=vmax)
    axes[2,0].set_title("Tumor_segment")
    plt.colorbar(im5,ax=axes[2,0])
    im6 = axes[2,1].imshow(uncertainty_map,cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im6,ax=axes[2,1])
    axes[2,1].set_title("Uncertainty_map")   
    criterion = Custom_L1()
    random_num = str(np.random.randint(99999999))
    plt.savefig(random_num+'temp.png',bbox_inches='tight')
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    os.remove(random_num+'temp.png')
    plt.show()
    return figure_np_3channel


def visualize_ktrans_compare(y,y_hat,vmax=0.1):
    plt.figure()
    concat_image = np.concatenate((y,y_hat),axis=1)
    plt.axis('off')
    random_num = str(np.random.randint(99999999))
    plt.imsave(random_num+'temp.png',concat_image,cmap='jet',vmin=0,vmax=vmax,format="png")
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용  (좌우flip안되게) 
    os.remove(random_num+'temp.png')
    plt.imshow(figure_np[:,:,:3])
    return figure_np_3channel



def nonvisualize_ktrans_compare(y,y_hat,vmax=0.1): # y,y_hat이 들어오면.. axis=1방향으로.. concat*
    plt.figure()
    concat_image = np.concatenate((y,y_hat),axis=1)
    plt.axis('off')
    random_num = str(np.random.randint(99999999))
    plt.imsave(random_num+'temp.png',concat_image,cmap='jet',vmin=0,vmax=vmax,format="png")
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용  (좌우flip안되게) -> 언제 flip 되는 편인가?
    os.remove(random_num+'temp.png')
    plt.cla()
    return figure_np_3channel


def nonvisualize_ktrans_compare_save(y,y_hat,vmax=0.1,epoch=0,id='230912'): # y,y_hat이 들어오면.. axis=1방향으로.. concat*
    plt.figure()
    concat_image = np.concatenate((y,y_hat),axis=1)
    plt.axis('off')
    random_num = str(np.random.randint(99999999))
    plt.imsave(f'{id}_{epoch}.png',concat_image,cmap='jet',vmin=0,vmax=vmax,format="png")
    figure_np = np.array(Image.open(f'{id}_{epoch}.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용  (좌우flip안되게) -> 언제 flip 되는 편인가?
    # os.remove(f'{id}_{epoch}.png')
    plt.cla()
    return figure_np_3channel



def nonvisualize_ktrans_compare_new(y,y_hat,vmax=0.1): # y,y_hat이 들어오면.. axis=1방향으로.. concat*
    plt.figure()
    concat_image = np.concatenate((y,y_hat),axis=0) # 0으로 붙이는 게 맞을 듯!
    plt.axis('off')
    random_num = str(np.random.randint(99999999))
    plt.imsave(random_num+'temp.png',concat_image,cmap='jet',vmin=0,vmax=vmax,format="png")
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용  (좌우flip안되게) -> 언제 flip 되는 편인가?
    os.remove(random_num+'temp.png')
    plt.cla()
    return figure_np_3channel


def visualize_ktrans_final(x,y,y_hat,text,vmax=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    im1 = axes[0,0].imshow(y,cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im1,ax=axes[0,0])
    axes[0,0].set_title(" Label:" + text)
    im2 = axes[0,1].imshow(y_hat,cmap = 'jet',vmin=0,vmax=vmax)
    axes[0,1].set_title("Output_DL_w_same_colormap")
    plt.colorbar(im2,ax=axes[0,1])
    im3 = axes[1,0].imshow(x,cmap = 'jet')
    axes[1,0].set_title("Input")
    plt.colorbar(im3,ax=axes[1,0])
    im4 = axes[1,1].imshow(y_hat,cmap = 'jet')
    plt.colorbar(im4,ax=axes[1,1])
    axes[1,1].set_title("OUTPUT_DL")   
    random_num = str(np.random.randint(99999999))
    plt.savefig(random_num+'temp.png',bbox_inches='tight')
    figure_np = np.array(Image.open(random_num+'temp.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    os.remove(random_num+'temp.png')
    plt.show()
    return figure_np_3channel


def visualize_ktrans_final_save(x,y,y_hat,text,vmax=0.1, id=''):
    fig, axes = plt.subplots(2, 2, figsize=(10,10))
    im1 = axes[0,0].imshow(y,cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im1,ax=axes[0,0])
    axes[0,0].set_title(" Label:" + text)
    im2 = axes[0,1].imshow(y_hat,cmap = 'jet',vmin=0,vmax=vmax)
    axes[0,1].set_title("Output_DL_w_same_colormap")
    plt.colorbar(im2,ax=axes[0,1])
    im3 = axes[1,0].imshow(x,cmap = 'jet')
    axes[1,0].set_title("Input")
    plt.colorbar(im3,ax=axes[1,0])
    im4 = axes[1,1].imshow(y_hat,cmap = 'jet')
    plt.colorbar(im4,ax=axes[1,1])
    axes[1,1].set_title("OUTPUT_DL")   
    random_num = str(np.random.randint(99999999))
    plt.savefig(f'{id}.png',bbox_inches='tight')
    figure_np = np.array(Image.open(f'{id}.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    # os.remove(random_num+'temp.png')
    plt.show()
    return figure_np_3channel


# def visualize_ktrans_patch_30_uncertainty_3output(x,y,y_hat, seg, uncertainty_map, y_mask, vmax=0.1):
def visualize_ktrans_final_save_extend(x ,y, y_hat, seg, uncertainty_map,y_mask,  text,vmax=0.1, id=''):
    fig, axes = plt.subplots(7, 4, figsize=(20,35))
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


    ''' 230831
    x,y,y_hat, seg, uncertainty_map=seg, vmax=0.1'
    x  torch.Size([3, 256, 256])
    y  torch.Size([3, 256, 256])
    y_hat  torch.Size([3, 256, 256])
    seg torch.Size([1, 256, 256])
    uncertainty_map=seg  torch.Size([1, 256, 256])
    '''
    X,Y = 0,1
    im1_1 = axes[X,Y].imshow(x[0],cmap = 'jet')
    axes[X,Y].set_title("Input 1(K)") # 지금은 relaxivity map이 아니라 DCE input 이므로 cmap bwr이 아니라 jet로 똑같이 맞춰주기! 
    plt.colorbar(im1_1,ax=axes[X,Y])

    X,Y = 0,2
    im1_2 = axes[X,Y].imshow(x[1],cmap = 'jet')
    axes[X,Y].set_title("Input 2(Vp)") # 지금은 relaxivity map이 아니라 DCE input 이므로 cmap bwr이 아니라 jet로 똑같이 맞춰주기! 
    plt.colorbar(im1_2,ax=axes[X,Y])

    X,Y = 0,3
    im1_3 = axes[X,Y].imshow(x[2],cmap = 'jet')
    axes[X,Y].set_title("Input 3(Ve)") # 지금은 relaxivity map이 아니라 DCE input 이므로 cmap bwr이 아니라 jet로 똑같이 맞춰주기! 
    plt.colorbar(im1_3,ax=axes[0,3])

    # 2. label 1
    X,Y = 1,1
    im2 = axes[X,Y].imshow(y[0],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im2,ax=axes[X,Y])
    axes[X,Y].set_title(" Label 1(K)")
    # 3. label 2
    X,Y = 1,2
    im3 = axes[X,Y].imshow(y[1],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im3,ax=axes[X,Y])
    axes[X,Y].set_title(" Label 2(Vp)")
    # 4. label 3
    X,Y = 1,3
    im4 = axes[X,Y].imshow(y[2],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im4,ax=axes[X,Y])
    axes[X,Y].set_title(" Label 3(Ve)")

    # 5. seg
    X,Y = 2,0

    # import pdb
    # pdb.set_trace()

    seg = seg[0] # 1번 slice만 계속 출력? slice 번호는?
    im5 = axes[X,Y].imshow(seg,cmap = 'jet', vmin=0, vmax=vmax)
    axes[X,Y].set_title("Tumor_segment")
    plt.colorbar(im5,ax=axes[X,Y])


    # 6_0. mask
    X,Y = 1,0
    y_mask = y_mask[0]
    im6_0 = axes[X,Y].imshow(y_mask,cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im6_0,ax=axes[ X,Y])
    axes[X,Y].set_title(" Mask")

    # 6. output 1
    X,Y = 2,1
    im6 = axes[X,Y].imshow(y_hat[0],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im6,ax=axes[ X,Y])
    axes[X,Y].set_title(" Output 1")
    # 7. output 2
    X,Y = 2,2
    im7 = axes[ X,Y].imshow(y_hat[1],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im7,ax=axes[ X,Y])
    axes[X,Y].set_title(" Output 2")
    # 8. output 3
    X,Y = 2,3
    im8 = axes[ X,Y].imshow(y_hat[2],cmap = 'jet',vmin=0,vmax=vmax)
    plt.colorbar(im8,ax=axes[X,Y])
    axes[X,Y].set_title(" Output 3")

    # 9. Uncertainty Map
    X,Y = 3,0
    im9 = axes[ X,Y].imshow(uncertainty_map[0],cmap = 'jet', vmin=0)
    axes[X,Y].set_title("UncertaintyMap(..Ktrans_max)")
    plt.colorbar(im9,ax=axes[ X,Y])

    # 10. output 1
    X,Y = 3,1
    im10 = axes[X,Y ].imshow(y_hat[0],cmap = 'jet',vmin=0)
    plt.colorbar(im10,ax=axes[X,Y])
    axes[X,Y].set_title("Output 1 max")
    # 11. output 2
    X,Y = 3,2
    im11 = axes[X,Y].imshow(y_hat[1],cmap = 'jet',vmin=0)
    plt.colorbar(im11,ax=axes[X,Y])
    axes[X,Y].set_title("Output 2 max")
    # 12. output 3
    X,Y = 3,3
    im12 = axes[X,Y].imshow(y_hat[2],cmap = 'jet',vmin=0, vmax=1.0)
    plt.colorbar(im12,ax=axes[X,Y])
    axes[X,Y].set_title("Output 3 max")

    # 2. label 1
    X,Y = 4,1
    im2 = axes[X,Y].imshow(y[0],cmap = 'jet',vmin=0)
    plt.colorbar(im2,ax=axes[X,Y])
    axes[X,Y].set_title(" Label 1(K) max")
    # 3. label 2
    X,Y = 4,2
    im3 = axes[X,Y].imshow(y[1],cmap = 'jet',vmin=0)
    plt.colorbar(im3,ax=axes[X,Y])
    axes[X,Y].set_title(" Label 2(Vp) max")
    # 4. label 3
    X,Y = 4,3
    im4 = axes[X,Y].imshow(y[2],cmap = 'jet',vmin=0)
    plt.colorbar(im4,ax=axes[X,Y])
    axes[X,Y].set_title(" Label 3(Ve) max")

    # 2. label 1
    X,Y = 5,1
    im2 = axes[X,Y].imshow(uncertainty_map[0],cmap = 'jet',vmin=0) # , vmax=vmax
    plt.colorbar(im2,ax=axes[X,Y])
    axes[X,Y].set_title(" UncertaintyMap(..Ktrans_max) 1.0")
    # 3. label 2
    X,Y = 5,2
    im3 = axes[X,Y].imshow(uncertainty_map[1],cmap = 'jet',vmin=0)# , vmax=vmax
    plt.colorbar(im3,ax=axes[X,Y])
    axes[X,Y].set_title(" UncertaintyMap(..Vp) 1.0")
    # 4. label 3
    X,Y = 5,3
    im4 = axes[X,Y].imshow(uncertainty_map[2],cmap = 'jet',vmin=0)# , vmax=vmax
    plt.colorbar(im4,ax=axes[X,Y])
    axes[X,Y].set_title(" UncertaintyMap(..Ve_max) 1.0")

    # 2. label 1
    X,Y = 6,1
    im2 = axes[X,Y].imshow(uncertainty_map[0],cmap = 'jet',vmin=0, vmax=vmax) # , vmax=vmax
    plt.colorbar(im2,ax=axes[X,Y])
    axes[X,Y].set_title(" UncertaintyMap(..Ktrans_max) 1.0")
    # 3. label 2
    X,Y = 6,2
    im3 = axes[X,Y].imshow(uncertainty_map[1],cmap = 'jet',vmin=0, vmax=vmax)# , vmax=vmax
    plt.colorbar(im3,ax=axes[X,Y])
    axes[X,Y].set_title(" UncertaintyMap(..Vp) 1.0")
    # 4. label 3
    X,Y = 6,3
    im4 = axes[X,Y].imshow(uncertainty_map[2],cmap = 'jet',vmin=0, vmax=vmax)# , vmax=vmax
    plt.colorbar(im4,ax=axes[X,Y])
    axes[X,Y].set_title(" UncertaintyMap(..Ve_max) 1.0")

    # # criterion = Custom_L1()
    # random_num = str(np.random.randint(99999999))
    # plt.savefig(random_num+'temp.png',bbox_inches='tight')
    # figure_np = np.array(Image.open(random_num+'temp.png'))
    # figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    # figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    # os.remove(random_num+'temp.png')
    # plt.show()
    # return figure_np_3channel

    # fig, axes = plt.subplots(2, 2, figsize=(10,10))
    # im1 = axes[0,0].imshow(y,cmap = 'jet',vmin=0,vmax=vmax)
    # plt.colorbar(im1,ax=axes[0,0])
    # axes[0,0].set_title(" Label:" + text)
    # im2 = axes[0,1].imshow(y_hat,cmap = 'jet',vmin=0,vmax=vmax)
    # axes[0,1].set_title("Output_DL_w_same_colormap")
    # plt.colorbar(im2,ax=axes[0,1])
    # im3 = axes[1,0].imshow(x,cmap = 'jet')
    # axes[1,0].set_title("Input")
    # plt.colorbar(im3,ax=axes[1,0])
    # im4 = axes[1,1].imshow(y_hat,cmap = 'jet')
    # plt.colorbar(im4,ax=axes[1,1])
    # axes[1,1].set_title("OUTPUT_DL")   
    # random_num = str(np.random.randint(99999999))
    plt.savefig(f'{id}.png',bbox_inches='tight')
    figure_np = np.array(Image.open(f'{id}.png'))
    figure_np_3channel_temp = np.swapaxes(figure_np[:,:,:3],0,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    figure_np_3channel = np.swapaxes(figure_np_3channel_temp,1,2)   # [3 X Y] 로 바꿈. Tensorboard용   
    # os.remove(random_num+'temp.png')
    plt.show()

    del fig, axes, figure_np_3channel_temp, figure_np

    return figure_np_3channel




def tile_2d_to_3d(numpy_array,repeat_num):
    # 256 256 을 256 256 60으로 확장 : tile_2d_to_3d(array,60)
    # 256 256 을 256 256 1으로 확장 : tile_2d_to_3d(array,1)
    tiled_array = np.transpose(np.tile(numpy_array,(repeat_num,1,1)),(1,2,0))
    #print(numpy_array.shape,tiled_array.shape)
    return tiled_array

def path_cutter(PATH,num):
    path_cut = '/'.join(PATH.split('/')[:-num])
    filename = PATH.split('/')[-1]
    return path_cut,filename

def train_validate_split(n,train_percent=.80, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(n)
    train_end = int(train_percent * n)
    train = perm[:train_end]
    validate = perm[train_end:]
    return train, validate
# fix seed as 0


def split_and_save_volume(file_name):
    SOURCE_DIR = '/mnt/hdd/kschoi/DCE_DSC_CKS/data/intermediate/'
    IDLIST_RAW  = glob.glob(SOURCE_DIR+'/*/*/Ktrans2DSC_masked_resampled_linear.nii.gz')
    IDLIST = [X.split('/')[-1] for X in IDLIST_RAW]
    print("total # of subjects : ", len(IDLIST))

    for ID_DIR in tqdm(IDLIST_RAW):
        BASE_DIR, _ = path_cutter(ID_DIR,1)
        vol_file = nii_to_numpy(BASE_DIR+"/"+file_name)
        z_dim = vol_file.shape[-1] # 18일것
        if len(vol_file.shape)==4:   # 256 256 18 60  for DSC
            num_repeat = vol_file.shape[2]  # 18
        else:
            num_repeat = vol_file.shape[-1] #18
        for slice_num in range(num_repeat):
            save_path = BASE_DIR+"/"+'slice'+str(slice_num).zfill(2)+"/"+'sliced_'+file_name
            
            if len(vol_file.shape)==4:   # 256 256 18 60  for DSC
                slice_file = vol_file[:,:,slice_num,:]
            else:   # 256 256 18   for leakage, ktrans
                slice_file = vol_file[:,:,slice_num]
            nft_img = nib.Nifti1Image(slice_file, np.identity(4)) #엄밀히는 affine제대로해줘야.
            nib.save(nft_img, save_path)

class SSIM_Loss(nn.Module):    #ms-ssim아님 => ms-ssim은 음수처리에 이상함
    def __init__(self):
        super(SSIM_Loss, self).__init__();

    def forward(self, predictions, target):       
        loss_ssim = 1-ssim(predictions[:,:,0,:,:],target[:,:,0,:,:]) #ms-ssim아님
        #loss = 1- msssim(1-predictions[:,:,0,:,:],1-target[:,:,0,:,:],normalize="relu")      
        return loss_ssim

class Custom_L1(nn.Module):  # N 1 60 32 32 꼴 input, output에서 t=0가지고만 비교 (ktrans prediction이 60이니)
    def __init__(self):
        super(Custom_L1, self).__init__();

    def forward(self, predictions, target):
        L1_loss = torch.nn.L1Loss()
        loss_t0 = L1_loss(predictions[:,:,0,:,:],target[:,:,0,:,:] )         
        #print("L1loss_pred.shape", predictions.shape)
        return loss_t0

class Weighted_L1(nn.Module):  # N 1 60 32 32 꼴 input, output에서 t=0가지고만 비교 (ktrans prediction이 60이니)
    def __init__(self,epsilon):
        super(Weighted_L1, self).__init__();
        self.epsilon = epsilon
    def forward(self, predictions, target):
        L1_loss = torch.nn.L1Loss()
        weights = self.epsilon + target[:,:,0,:,:]  # ktrans가 -1이면 epsilon, 그 이상이면 +값.
        loss_t0 = L1_loss(predictions[:,:,0,:,:]*weights,target[:,:,0,:,:]*weights)         
        #print("L1loss_pred.shape", predictions.shape)
        return loss_t0


class Weighted_L1_patch(nn.Module):  # N 60 16 16 꼴 input, output에서 t=0가지고만 비교 (ktrans prediction이 60이니)
    def __init__(self,epsilon):
        super(Weighted_L1_patch, self).__init__();
        self.epsilon = epsilon
    def forward(self, predictions, target):
        L1_loss = torch.nn.L1Loss()
        weights = self.epsilon + target[:,0,:,:]  # ktrans가 -1이면 epsilon, 그 이상이면 +값.
        loss_t0 = L1_loss(predictions[:,0,:,:]*weights,target[:,0,:,:]*weights)         
        #print("L1loss_pred.shape", predictions.shape)
        return loss_t0


class SquareWeighted_L1(nn.Module):  # N 1 60 32 32 꼴 input, output에서 t=0가지고만 비교 (ktrans prediction이 60이니)
    def __init__(self,epsilon):
        super(SquareWeighted_L1, self).__init__();
        self.epsilon = epsilon
    def forward(self, predictions, target):
        L1_loss = torch.nn.L1Loss()
        weights = 1+target[:,:,0,:,:]  # ktrans가 -1이면 epsilon, 그 이상이면 +값.
        weights_squared = weights*weights + self.epsilon
        loss_t0 = L1_loss(predictions[:,:,0,:,:]*weights_squared,target[:,:,0,:,:]*weights_squared)         
        #print("L1loss_pred.shape", predictions.shape)
        return loss_t0

class CubeWeighted_L1(nn.Module):  # N 1 60 32 32 꼴 input, output에서 t=0가지고만 비교 (ktrans prediction이 60이니)
    def __init__(self,epsilon):
        super(CubeWeighted_L1, self).__init__();
        self.epsilon = epsilon
    def forward(self, predictions, target):
        L1_loss = torch.nn.L1Loss()       
        weights = 1+target[:,:,0,:,:]  # ktrans가 -1이면 epsilon, 그 이상이면 +값.
        weights_cubed = weights*weights*weights + self.epsilon
        loss_t0 = L1_loss(predictions[:,:,0,:,:]*weights_cubed,target[:,:,0,:,:]*weights_cubed)         
        #print("L1loss_pred.shape", predictions.shape)
        return loss_t0

class SqrtWeighted_L1(nn.Module):  # N 1 60 32 32 꼴 input, output에서 t=0가지고만 비교 (ktrans prediction이 60이니)
    def __init__(self,epsilon):
        super(SqrtWeighted_L1, self).__init__();
        self.epsilon = epsilon
    def forward(self, predictions, target):
        L1_loss = torch.nn.L1Loss()       
        weights = 1+target[:,:,0,:,:]  # ktrans가 -1이면 epsilon, 그 이상이면 +값.
        weights_sqrt = torch.sqrt(weights) + self.epsilon
        loss_t0 = L1_loss(predictions[:,:,0,:,:]*weights_sqrt,target[:,:,0,:,:]*weights_sqrt)         
        #print("L1loss_pred.shape", predictions.shape)
        return loss_t0



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


def tile_3d_to_60_torch(torch_array):
    # 1000 16 16 을 1000 60 16 16 으로 확장
    tiled_array =torch.tile(torch.unsqueeze(torch_array,1),(1,60,1,1))
    return tiled_array

def tile_3d_to_1_torch(torch_array):
    # 1000 16 16 을 1000 60 16 16 으로 확장
    tiled_array =torch.tile(torch.unsqueeze(torch_array,1),(1,1,1,1))
    return tiled_array



def patch_border_correction(dim_size, patch_dim_size,center):
    lower_raw = center - int(patch_dim_size/2)
    upper_raw = lower_raw + patch_dim_size
    if lower_raw <0 : 
        lower_raw,upper_raw = 0,patch_dim_size
    if upper_raw > (dim_size - 1):
        lower_raw,upper_raw = dim_size -1- patch_dim_size,dim_size - 1
    return lower_raw, upper_raw
# patch_border_correction(23,8,7)
# 전체 dim 크기 23 patch size 8일때 center 7인 patch의 
# 시작과 끝을 return  (border넘어가면 땡겨서)

def patch_border_correction_3d(array_shape,patch_size,center_coordinate):
    lower_coordinate = torch.tensor([0,0,0])
    upper_coordinate = torch.tensor([0,0,0])
    for i in range(len(array_shape)):
        lower_coordinate[i],upper_coordinate[i] = patch_border_correction(array_shape[i],patch_size[i],center_coordinate[i])
    return {'lower':lower_coordinate, 'upper':upper_coordinate}

#array_shape = torch.Size([256, 256, 18])
#patch_size = torch.tensor([16,16,1])
#center_coordinate = torch.tensor([55,108,9])
#x = patch_border_correction_3d(array_shape,patch_size,center_coordinate)
#print(x['lower'],x['upper'])

def RandCropByPosNegLabel_custom_coordinates(seg_file,patch_size,num_patch_samples,pos_neg_ratio):
    array_shape = seg_file.shape  # torch.Size([256, 256, 18])
    positive_label_coord =torch.nonzero(seg_file)
    n  = len(positive_label_coord)
    if n==0:
        #print("this has no tumor")
        pos_neg_ratio = 0
    n_pos = int(num_patch_samples*pos_neg_ratio)
    n_neg = num_patch_samples - n_pos
    index_pos = np.random.choice(n,n_pos,replace="False")
    index_neg_1 = np.random.choice(256,n_neg,replace="True") 
    index_neg_2 = np.random.choice(256,n_neg,replace="True") 
    index_neg_3 = np.random.choice(18,n_neg,replace="True") 
    index_neg = [torch.tensor([a,b,c]) for a,b,c in zip(index_neg_1,index_neg_2,index_neg_3)]
    #coord_pos = [positive_label_coord[i] for i in index_pos]
    coord_pos = [patch_border_correction_3d(array_shape,patch_size,positive_label_coord[i]) for i in index_pos]
    coord_neg = [patch_border_correction_3d(array_shape,patch_size,i) for i in index_neg]
    coord_final = coord_neg+coord_pos
    random.shuffle(coord_final) 
    return coord_final

def FixedRandCropByPosNegLabel_custom_coordinates(x, seg_file,patch_size,num_patch_samples,pos_neg_ratio):
    array_shape = seg_file.shape  # torch.Size([256, 256, 18])    
    positive_label_coord =torch.nonzero(seg_file)
    n  = len(positive_label_coord)
    if n==0:
        #print("this has no tumor")
        pos_neg_ratio = 0
    n_pos = int(num_patch_samples*pos_neg_ratio)
    n_neg = num_patch_samples - n_pos
    np.random.seed(x)
    index_pos = np.random.choice(n,n_pos,replace="False")
    index_neg_1 = np.random.choice(256,n_neg,replace="True") 
    index_neg_2 = np.random.choice(256,n_neg,replace="True") 
    index_neg_3 = np.random.choice(18,n_neg,replace="True") 
    np.random.seed()
    index_neg = [torch.tensor([a,b,c]) for a,b,c in zip(index_neg_1,index_neg_2,index_neg_3)]
    #coord_pos = [positive_label_coord[i] for i in index_pos]
    coord_pos = [patch_border_correction_3d(array_shape,patch_size,positive_label_coord[i]) for i in index_pos]
    coord_neg = [patch_border_correction_3d(array_shape,patch_size,i) for i in index_neg]
    coord_final = coord_neg+coord_pos
    random.shuffle(coord_final)
    return coord_final



def apply_patch_coordinates_single(array,coord):
    return array[coord['lower'][0]:coord['upper'][0],coord['lower'][1]:coord['upper'][1],coord['lower'][2]:coord['upper'][2]]

def apply_patch_coordinates_ch1(torch_array, coords):
    patches = [apply_patch_coordinates_single(torch_array,coord) for coord in coords]
    patches_stacked = torch.stack(patches)[:,:,:,0] # 1000 16 16 
    patches_stacked_1 = tile_3d_to_1_torch(patches_stacked) # 1000 60 16 16   
    return patches_stacked_1 

def apply_patch_coordinates(torch_array, coords):
    # print(f'def apply_patch_coordinates(torch_array, coords): ... makes channel to tiled 60 channel! utils_dce2ktrans.apply_patch_coordinates')
    patches = [apply_patch_coordinates_single(torch_array,coord) for coord in coords]
    patches_stacked = torch.stack(patches)[:,:,:,0] # 1000 16 16 
    patches_stacked_60 = tile_3d_to_60_torch(patches_stacked) # 1000 60 16 16   
    return patches_stacked_60 


def apply_patch_coordinates_4d_single(array,coord):
    return array[coord['lower'][0]:coord['upper'][0],coord['lower'][1]:coord['upper'][1],coord['lower'][2]:coord['upper'][2],:]

def apply_patch_coordinates_4d(torch_array_4d, coords):
    patches = [apply_patch_coordinates_4d_single(torch_array_4d,coord) for coord in coords]
    patches_stacked = torch.stack(patches) # 1000 16 16 1 60 
    patches_stacked_final = patches_stacked[:,:,:,0,:].permute((0,3,1,2)) # 1000 60 16 16
    return patches_stacked_final



class TemporalConvNet_fixed(nn.Module):  # N 60 1 > N 60 1 
    def __init__(self, num_timeframes, input_channels, num_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet_fixed, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
    def forward(self, input):  # N 1 60 32 32
        #print("input shape",input.shape)
        #print("input_shape2",input.shape[2])
        output = self.model(input) # 
        return output
    def forward_test(self):
        test_input = torch.rand(100,1,60)
        test_output = self.forward(test_input)
        print("test_input: 100 samples of 1 channel - sequence length 60",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet_fixed(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()



class Wrapper_Net_fixed(nn.Module):  # N 60 1 > N 60 1  네트워크로
       # Wrap해서  N 60 16 16 넣으면 N 60 16 16 오는 거  (patch size 16 16 1이면)
       # 이걸 이후에 patch sliding forward해서 
       # N 256 256 18 60  넣으면 N 256 256 18 60 나오는게 최종적으로 나올 것임. 

     # relu랑 linear를 사용한다. 

    def __init__(self, input_network):
        super(Wrapper_Net_fixed, self).__init__()
        self.input_network = input_network
        hidden_layer_size = 16
        self.linear = nn.Linear(60,hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size,hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size,1)
        self.relu = nn.ReLU()

    def forward(self, input): #input patch : N 60 16 16     
         # input patch : N 16 16 1 60
        #print('input', input.shape)
        input_permuted = input.permute((0,2,3,1))  # N 16 16 60                                       
        reshaped_input = input_permuted.reshape(input_permuted.shape[0]*input_permuted.shape[1]*input_permuted.shape[2],1, input_permuted.shape[3]) # N*16*16, 1, 60
        output = self.input_network(reshaped_input) #N*16*16 1 60
        output = output[:,0,]  #N*16*16 60   (Linear layer넣기 위해서 1 제거)
        output_fcn = self.linear3(self.relu(self.linear2(self.relu(self.linear(self.relu(output))))))  # N*16*16 1 1
        reshaped_output = output_fcn.reshape(input_permuted.shape[0],input_permuted.shape[1],input_permuted.shape[2],1)  # N 16 16 1 
        reshaped_output_60 = torch.tile(reshaped_output,[1,1,1,60]) # N 16 16 60
        reshaped_output_60_permute = reshaped_output_60.permute((0,3,1,2)) # N 60 16 16
        return reshaped_output_60_permute
    def forward_test(self):
        test_input = torch.rand(3,60,16,16)
        test_output = self.forward(test_input)
        print("test_input for Wrapper_Net: ",test_input.shape)
        print("test_output for Wrapper_Net: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()
        # m = Wrapper_Net(n)
        # m.forward_test()



class TemporalConvNet(nn.Module):  # N 60 1 > N 60 1 
    def __init__(self, num_timeframes, num_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        output_size=1
        hidden_layer_size = 16 
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_timeframes if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)

    def forward(self, input):  # N 1 60 32 32
        #print("input shape",input.shape)
        #print("input_shape2",input.shape[2])
        output = self.model(input) # 
        return output
    def forward_test(self):
        test_input = torch.rand(100,60,1)
        test_output = self.forward(test_input)
        print("test_input: ",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()

class TemporalConvDualNet(nn.Module):  # N 60 2 > N 60 1 
    def __init__(self, num_timeframes, num_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvDualNet, self).__init__()
        output_size=2
        hidden_layer_size = 16 
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_timeframes if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)

    def forward(self, input):  # N 1 60 32 32
        #print("input shape",input.shape)
        #print("input_shape2",input.shape[2])
        output = self.model(input) # 
        return output
    def forward_test(self):
        test_input = torch.rand(100,60,1)
        test_output = self.forward(test_input)
        print("test_input: ",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()






class TemporalConvNet_AIF(nn.Module):
    def __init__(self, num_timeframes, num_channels, kernel_size=5, dropout=0.2,output_scale=1):
        super(TemporalConvNet_AIF, self).__init__()
        output_size=1
        hidden_layer_size = 16  # 실험 - 원래 5였음. 
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_timeframes if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.model = nn.Sequential(*layers)
        self.linear0 = nn.Linear(2,1)
        self.linear = nn.Linear(num_channels[-1],hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size,hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size,output_size)
        self.relu = nn.ReLU()
        self.output_scale = output_scale

    def forward(self, input, AIF):  # N 1 60 32 32
        #print("input shape",input.shape)
        #print("input_shape2",input.shape[2])

        input_fixed = input  # -1 ~1 > 0~1
        input_permuted = input_fixed.permute(0,3,4,2,1) # N 32 32 60 1
        reshaped_input = input_permuted.reshape(input_permuted.shape[0]*input_permuted.shape[1]*input_permuted.shape[2],input_permuted.shape[3],input_permuted.shape[4]) # N*32*32, 60, 1

        reshaped_AIF_1 = torch.unsqueeze(torch.unsqueeze(AIF,dim=0),dim=2) # 1 60 1
        reshaped_AIF_2 =  torch.tile(reshaped_AIF_1 ,[input_permuted.shape[0]*input_permuted.shape[1]*input_permuted.shape[2],1,1])  # N*32*32 60 1

        reshaped_input_and_AIF = torch.cat([reshaped_input,reshaped_AIF_2],dim=2)  #N*32*32 60 2
        output = self.model(reshaped_input_and_AIF) #N*32*32 60 2
        output_1channel = self.linear0(output) #N*32*32 60 1
        output_final = self.linear(output_1channel[:,:,0]) # N*32*32,1,1
        #output_final_relu = self.relu(output_final)
        output_final_fcn = self.linear3(self.relu(self.linear2(self.relu(output_final))))
        output_reshaped = output_final_fcn.reshape(input_permuted.shape[0],input_permuted.shape[1],input_permuted.shape[2],1,1)  # N 32 32 1 1
        output_reshaped_permuted = self.output_scale * output_reshaped.permute(0,3,4,1,2)  # N 1 1 32 32
        return output_reshaped_permuted

class Wrapper_Net(nn.Module):  # N 60 1 > N 60 1  네트워크로
       # Wrap해서  N 60 16 16 넣으면 N 60 16 16 오는 거  (patch size 16 16 1이면)
       # 이걸 이후에 patch sliding forward해서 
       # N 256 256 18 60  넣으면 N 256 256 18 60 나오는게 최종적으로 나올 것임. 

     # relu랑 linear를 사용한다. 

    def __init__(self, input_network):
        super(Wrapper_Net, self).__init__()
        self.input_network = input_network
        hidden_layer_size = 16
        self.linear = nn.Linear(60,hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size,hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size,1)
        self.relu = nn.ReLU()

    def forward(self, input): #input patch : N 60 16 16 
     
         # input patch : N 16 16 1 60
        #print('input', input.shape)
        input_permuted = input.permute((0,2,3,1))  # N 16 16 60                                       
        reshaped_input = input_permuted.reshape(input_permuted.shape[0]*input_permuted.shape[1]*input_permuted.shape[2],input_permuted.shape[3],1) # N*16*16, 60, 1
        output = self.input_network(reshaped_input) #N*16*16 60 1
        output = output[:,:,0]  #N*16*16 60   (Linear layer넣기 위해서 1 제거)
        output_fcn = self.linear3(self.relu(self.linear2(self.relu(self.linear(self.relu(output))))))  # N*16*16 1 1
        reshaped_output = output_fcn.reshape(input_permuted.shape[0],input_permuted.shape[1],input_permuted.shape[2],1)  # N 16 16 1 
        reshaped_output_60 = torch.tile(reshaped_output,[1,1,1,60]) # N 16 16 60
        reshaped_output_60_permute = reshaped_output_60.permute((0,3,1,2)) # N 60 16 16
        return reshaped_output_60_permute
    def forward_test(self):
        test_input = torch.rand(3,60,16,16)
        test_output = self.forward(test_input)
        print("test_input for Wrapper_Net: ",test_input.shape)
        print("test_output for Wrapper_Net: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()
        # m = Wrapper_Net(n)
        # m.forward_test()


        


class Wrapper_integrate_Net(nn.Module):  # N 60 1 > N 60 1  네트워크로
       # Wrap해서  N 60 16 16 넣으면 N 60 16 16 오는 거  (patch size 16 16 1이면)
       # 이걸 이후에 patch sliding forward해서 
       # N 256 256 18 60  넣으면 N 256 256 18 60 나오는게 최종적으로 나올 것임. 

     # relu랑 linear를 사용한다. 

    def __init__(self, input_network):
        super(Wrapper_integrate_Net, self).__init__()
        self.input_network = input_network
        self.relu = nn.ReLU()
        self.linear_simple = nn.Linear(1,1)
    def forward(self, input): #input patch : N 60 16 16 
         # input patch : N 16 16 1 60
        #print('input', input.shape)
        input_permuted = input.permute((0,2,3,1))  # N 16 16 60                                       
        reshaped_input = input_permuted.reshape(input_permuted.shape[0]*input_permuted.shape[1]*input_permuted.shape[2],input_permuted.shape[3],1) # N*16*16, 60, 1
        output = self.input_network(reshaped_input) #N*16*16 60 1
        output = output[:,:,0]  #N*16*16 60   (Linear layer넣기 위해서 1 제거)
        output_integrated = torch.sum(output,axis=-1,keepdim=True)  # N*16*16 1
        output_integrated_linear = self.linear_simple(output_integrated) #N*16*16 1 
        reshaped_output = output_integrated_linear.reshape(input_permuted.shape[0],input_permuted.shape[1],input_permuted.shape[2],1)  # N 16 16 1 
        reshaped_output_60 = torch.tile(reshaped_output,[1,1,1,60]) # N 16 16 60
        reshaped_output_60_permute = reshaped_output_60.permute((0,3,1,2)) # N 60 16 16
        return reshaped_output_60_permute
    def forward_test(self):
        test_input = torch.rand(3,60,16,16)
        test_output = self.forward(test_input)
        print("test_input for Wrapper_Net: ",test_input.shape)
        print("test_output for Wrapper_Net: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()
        # m = Wrapper_Net(n)
        # m.forward_test()


        
class Chomp1d_custom(nn.Module):
# Chomp = Padding trick for causal convolutions
# https://discuss.pytorch.org/t/causal-convolution/3456/14
# ABCD > PABCDP > PA, AB, BC, CD로 chomp
    def __init__(self, chomp_size):
        super(Chomp1d_custom, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[...,:-self.chomp_size].contiguous()

class TemporalBlock_custom(nn.Module): # N_batch n_input 64 64 60  > N_batch n_output 64 64 60
    def __init__(self, n_inputs, n_outputs, kernel, stride, dilation, dropout=0): # padding지움
        super(TemporalBlock_custom, self).__init__()
        kernel_size = (1,1,kernel)
        dilation_size = (1,1,dilation)
        padding_size = (0,0,(kernel-1)*dilation)
        self.conv1 = weight_norm(nn.Conv3d(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride,padding=padding_size,dilation=dilation_size))
        self.chomp1 = Chomp1d_custom(padding_size[-1])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv3d(n_outputs, n_outputs, kernel_size=kernel_size, stride=stride,padding=padding_size,dilation=dilation_size))
        self.chomp2 = Chomp1d_custom(padding_size[-1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv3d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out+res)



class TemporalConvNet_custom(nn.Module):  # 16 60 64 64 > 16 60 64 64  
                                   # n_batch  256(batch_size) 60 ()  16x16 : patch size
    # list_channels : 시작이 1  (1x60 하나 넣는 경우)
    #  끝도 1 (통상적으로 하는 경우)
    # 사이엔 channel 다양하게

    def __init__(self, num_timeframes, list_channels, kernel, dropout=0):
        super(TemporalConvNet_custom, self).__init__()
        layers = []
        num_levels = len(list_channels)-1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            layers += [TemporalBlock_custom(in_channels, out_channels, kernel, stride=1, dilation=dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
    def forward(self, input):  
        input_reshaped = torch.transpose(torch.unsqueeze(input,-1),1,-1) #  16 1 64 64 60
        #print("input_reshaped shape", input_reshaped.shape)
        output = self.model(input_reshaped) #  16 list_channel[-1] 64 64 60
        #print("output shape",output.shape)
        output_reshaped_1 = torch.transpose(output,1,-1) # 16 60 64 64 list_channel[-1]  
        #print("outputres1",output_reshaped_1.shape)
        output_reshaped_2 = output_reshaped_1[:,-1:,:,:,0] # 16 1 64 64 
        #print("outputres2",output_reshaped_2.shape)
        output_tiled = torch.tile(output_reshaped_2,[1,60,1,1]) # 마지막 값 복제
        #print("output_tiled_shape",output_tiled.shape)
        return output_tiled
    def forward_test(self):
        test_input = torch.rand(16,60,64,64)
        test_output = self.forward(test_input)
        print("test_input: 256 samples of 60*16*16",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet_fixed(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()
# x = torch.rand(1,3,256,256,60)
# n_inputs = 3
# n_outputs = 3
# kernel = 3
# kernel_size = (1,1,kernel)
# stride = 1
# dilation = 2
# dilation_size = (1,1,dilation)
# padding_size = (0,0,(kernel-1)*dilation)
# dropout = 0.2
# a = TemporalBlock_custom(n_inputs, n_outputs, kernel, stride, dilation)
# print(a(x).shape) # 1 3 256 256 60 

# num_timeframes = 60
# list_channels = [1,10,10,10,1]
# kernel = 3
# b = TemporalConvNet_fixed(num_timeframes,list_channels,kernel)

# c = torch.rand(1,60,256,256)
# b(c).shape



class TemporalBlock_custom_norelu(nn.Module): # 1 n_input 64 64 60  > 1 n_output 64 64 60
    def __init__(self, n_inputs, n_outputs, kernel, stride, dilation, dropout=0): # padding지움
        super(TemporalBlock_custom_norelu, self).__init__()
        kernel_size = (1,1,kernel)
        dilation_size = (1,1,dilation)
        padding_size = (0,0,(kernel-1)*dilation)
        self.conv1 = weight_norm(nn.Conv3d(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride,padding=padding_size,dilation=dilation_size))
        self.chomp1 = Chomp1d_custom(padding_size[-1])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv3d(n_outputs, n_outputs, kernel_size=kernel_size, stride=stride,padding=padding_size,dilation=dilation_size))
        self.chomp2 = Chomp1d_custom(padding_size[-1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv3d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out+res


class TemporalConvNet_custom_norelu(nn.Module):  # 16 60 64 64 > 16 60 64 64  
                                   # n_batch  256(batch_size) 60 ()  16x16 : patch size
    # list_channels : 시작이 1  (1x60 하나 넣는 경우)
    #  끝도 1 (통상적으로 하는 경우)
    # 사이엔 channel 다양하게

    def __init__(self, num_timeframes, list_channels, kernel, dropout=0):
        super(TemporalConvNet_custom_norelu, self).__init__()
        layers = []
        num_levels = len(list_channels)-1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            layers += [TemporalBlock_custom_norelu(in_channels, out_channels, kernel, stride=1, dilation=dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
    def forward(self, input):  
        input_reshaped = torch.transpose(torch.unsqueeze(input,-1),1,-1) #  16 1 64 64 60
        #print("input_reshaped shape", input_reshaped.shape)
        output = self.model(input_reshaped) #  16 list_channel[-1] 64 64 60
        #print("output shape",output.shape)
        output_reshaped_1 = torch.transpose(output,1,-1) # 16 60 64 64 list_channel[-1]  
        #print("outputres1",output_reshaped_1.shape)
        output_reshaped_2 = output_reshaped_1[:,-1:,:,:,0] # 16 1 64 64 
        #print("outputres2",output_reshaped_2.shape)
        output_tiled = torch.tile(output_reshaped_2,[1,60,1,1]) # 마지막 값 복제
        #print("output_tiled_shape",output_tiled.shape)
        return output_tiled
    def forward_test(self):
        test_input = torch.rand(16,60,64,64)
        test_output = self.forward(test_input)
        print("test_input: 256 samples of 60*16*16",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet_fixed(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()



class TemporalConvNet_custom_norelu_integrate(nn.Module):  # 16 60 64 64 > 16 60 64 64  
                                   # n_batch  256(batch_size) 60 ()  16x16 : patch size
    # list_channels : 시작이 1  (1x60 하나 넣는 경우)
    #  끝도 1 (통상적으로 하는 경우)
    # 사이엔 channel 다양하게

    def __init__(self, num_timeframes, list_channels, kernel, dropout=0):
        super(TemporalConvNet_custom_norelu_integrate, self).__init__()
        layers = []
        num_levels = len(list_channels)-1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            layers += [TemporalBlock_custom_norelu(in_channels, out_channels, kernel, stride=1, dilation=dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
    def forward(self, input):  
        input_reshaped = torch.transpose(torch.unsqueeze(input,-1),1,-1) #  16 1 64 64 60
        #print("input_reshaped shape", input_reshaped.shape)
        output = self.model(input_reshaped) #  16 list_channel[-1] 64 64 60
        #print("output shape",output.shape)
        output_reshaped_1 = torch.transpose(output,1,-1) # 16 60 64 64 list_channel[-1]  
        #print("outputres1",output_reshaped_1.shape)
        output_reshaped_2 = torch.sum(output_reshaped_1[...,0],1,keepdim=True) # 16 1 64 64
        output_tiled = torch.tile(output_reshaped_2,[1,60,1,1]) # 마지막 값 복제
        #print("output_tiled_shape",output_tiled.shape)
        return output_tiled
    def forward_test(self):
        test_input = torch.rand(16,60,64,64)
        test_output = self.forward(test_input)
        print("test_input: 256 samples of 60*16*16",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet_fixed(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()




class TemporalConvNet_custom_norelu_multi_input(nn.Module): 
     # n_channel_in 16 60 64 64 > n_channel_output 16 60 64 64 
     # 즉 이건 multichannel로 받아서 5d > 5d로 하는 네트워크 
     # n_batch  256(batch_size) 60 ()  16x16 : patch size
    # list_channels : 시작이 1  (1x60 하나 넣는 경우)
    #  끝도 1 (통상적으로 하는 경우)
    # 사이엔 channel 다양하게

    def __init__(self, num_timeframes, list_channels, kernel, dropout=0):
        super(TemporalConvNet_custom_norelu_multi_input, self).__init__()
        layers = []
        num_levels = len(list_channels)-1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            layers += [TemporalBlock_custom_norelu(in_channels, out_channels, kernel, stride=1, dilation=dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
        self.list_channels = list_channels
    def forward(self, input):  
        # input : n_c n_b 60 64 64
        input_reshaped = torch.transpose(torch.unsqueeze(torch.transpose(input,0,1),-1),2,-1)[:,:,0,:,:,:] # n_b n_c 64 64 60
        output = self.model(input_reshaped)
        # 마지막 성분 가져와서 그걸로 60 timecurve 복제
        output_last = torch.tile(output[:,:,:,:,-1:],[1,1,1,1,60]) # n_b n_c 64 64 60
        output_reshaped = torch.transpose(torch.unsqueeze(torch.transpose(output_last,0,1),2),2,-1)[...,0] # n_c n_b 60 64 64 
        return output_reshaped
    def forward_test(self):
        test_input = torch.rand(self.list_channels[0],16,60,64,64)
        test_output = self.forward(test_input)
        print("test_input:",test_input.shape)
        print("test_output:",test_output.shape)



class TemporalConvNet_custom_norelu_multi_input_linear(nn.Module): 
     # n_channel_in 16 60 64 64 > n_channel_output 16 60 64 64 
     # 즉 이건 multichannel로 받아서 5d > 5d로 하는 네트워크 
     # n_batch  256(batch_size) 60 ()  16x16 : patch size
    # list_channels : 시작이 1  (1x60 하나 넣는 경우)
    #  끝도 1 (통상적으로 하는 경우)
    # 사이엔 channel 다양하게

    def __init__(self, num_timeframes, list_channels, kernel, dropout=0):
        super(TemporalConvNet_custom_norelu_multi_input_linear, self).__init__()
        self.linear = nn.Linear(2,2)
        layers = []
        num_levels = len(list_channels)-1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            layers += [TemporalBlock_custom_norelu(in_channels, out_channels, kernel, stride=1, dilation=dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
        self.list_channels = list_channels
    def forward(self, input):  
        # input : n_c n_b 60 64 64
        input_linear = self.linear(input.permute(4,1,2,3,0)).permute(4,1,2,3,0) # 2 16 60 64 64
        input_reshaped = torch.transpose(torch.unsqueeze(torch.transpose(input_linear,0,1),-1),2,-1)[:,:,0,:,:,:] # n_b n_c 64 64 60
        output = self.model(input_reshaped)
        # 마지막 성분 가져와서 그걸로 60 timecurve 복제
        output_last = torch.tile(output[:,:,:,:,-1:],[1,1,1,1,60]) # n_b n_c 64 64 60
        output_reshaped = torch.transpose(torch.unsqueeze(torch.transpose(output_last,0,1),2),2,-1)[...,0] # n_c n_b 60 64 64 
        return output_reshaped
    def forward_test(self):
        test_input = torch.rand(self.list_channels[0],16,60,64,64)
        test_output = self.forward(test_input)
        print("test_input:",test_input.shape)
        print("test_output:",test_output.shape)

class TemporalConvNet_custom_norelu_3output(nn.Module):  # 16 60 64 64 > 16 3 64 64  
                                   # n_batch  256(batch_size) 60 ()  16x16 : patch size
    # list_channels : 시작이 1  (1x60 하나 넣는 경우)
    #  끝도 1 (통상적으로 하는 경우)
    # 사이엔 channel 다양하게

    def __init__(self, num_timeframes, list_channels, kernel, dropout=0):
        super(TemporalConvNet_custom_norelu_3output, self).__init__()
        layers = []
        num_levels = len(list_channels)-1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            layers += [TemporalBlock_custom_norelu(in_channels, out_channels, kernel, stride=1, dilation=dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
        self.linear = nn.Linear(60,3)

    def forward(self, input):  
        input_reshaped = torch.transpose(torch.unsqueeze(input,-1),1,-1) #  16 1 64 64 60
        #print("input_reshaped shape", input_reshaped.shape)
        output = self.model(input_reshaped) #  16 list_channel[-1] 64 64 60
        output_linear = self.linear(output) # 16 list_channel[-1] 64 64 3
        #print("output shape",output.shape)
        output_reshaped_1 = torch.transpose(output_linear,1,-1) # 16 3 64 64 list_channel[-1]   ### 이 부분 체크!
        output_reshaped_2 = output_reshaped_1[:,:,:,:,0] # 16 3 64 64 
        return output_reshaped_2
    def forward_test(self):
        test_input = torch.rand(16,60,64,64)
        test_output = self.forward(test_input)
        print("test_input: 256 samples of 60*16*16",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet_fixed(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()


# input: (B 60 256 256)

# ouput: '':(), '':(), '':()

class TemporalConvNet_PUnet_custom_norelu_3output(nn.Module):  # 16 60 64 64 > 16 3 64 64  
                                   # n_batch  256(batch_size) 60 ()  16x16 : patch size
    # list_channels : 시작이 1  (1x60 하나 넣는 경우)
    #  끝도 1 (통상적으로 하는 경우)
    # 사이엔 channel 다양하게

    def __init__(self, num_timeframes, list_channels, kernel, dropout=0):
        super(TemporalConvNet_custom_norelu_3output, self).__init__()
        ############# 모델 설계해서 붙이는 부분*
        layers = []
        num_levels = len(list_channels)-1
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = list_channels[i]
            out_channels = list_channels[i+1]
            layers += [TemporalBlock_custom_norelu(in_channels, out_channels, kernel, stride=1, dilation=dilation_size, dropout=dropout)]
        self.model = nn.Sequential(*layers)
        self.linear = nn.Linear(60,3)

    def forward(self, input):  
        input_reshaped = torch.transpose(torch.unsqueeze(input,-1),1,-1) #  16 1 64 64 60
        #print("input_reshaped shape", input_reshaped.shape)
    ### TCN model 통과하는 부분
        output = self.model(input_reshaped) #  16 list_channel[-1] 64 64 60
        output_linear = self.linear(output) # 16 list_channel[-1] 64 64 3
        #print("output shape",output.shape)
        output_reshaped_1 = torch.transpose(output_linear,1,-1) # 16 3 64 64 list_channel[-1]   ### 이 부분 체크!
        output_reshaped_2 = output_reshaped_1[:,:,:,:,0] # 16 3 64 64 
        return output_reshaped_2
    def forward_test(self):
        test_input = torch.rand(16,60,64,64)
        test_output = self.forward(test_input)
        print("test_input: 256 samples of 60*16*16",test_input.shape)
        print("test_output: ",test_output.shape)
        # 사용법
        # n =  TemporalConvNet_fixed(num_timeframes=60, num_channels = [60]*5, kernel_size=5)
        # n.forward_test()


