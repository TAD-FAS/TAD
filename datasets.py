# -*- coding: utf-8 -*-

"""
Created on 2021/10/7 22:42
@author: Acer
@description: 
"""

import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# import torchvision.transforms as transforms
from mytransform import *
import random

from pytorchtools import main


'''
the dir exsample of dataset oulu-npu
|--oulu-npu
|  |--Protocol_1
|  |  |--testA      # live face in testset
|  |  |--testAD     # depth map of  live face in testset
|  |  |--testB      # spoof face in testset
|  |  |--trainA     # live face in trainset
|  |  |--trainAD    # depth map of live face in trainset
|  |  |--trainB     # spoof face in trainset
|  |  |--trainBD    # depth map of spoof face in trainset
|  |--Protocol_2
|  |  |--...
|  |--Protocol_3
|  |  |--test1A
|  |  |--test1B
|  |  |--...
|  |  |--test6A
|  |  |--train1A
|  |  |--train1B
|  |  |--train1D
|  |  |--...
|  |  |--train6A
|  |  |--train6B
|  |  |--train6D
|  |--Protocol_4
|  |  |--...
'''

class UnpairDataset(Dataset):
    def __init__(self, params, phase='train'):
        super(UnpairDataset, self).__init__()
        self.data_root = params['data_root']
        self.phase = phase
        self.params = params

        # live images
        l_img_names = os.listdir(os.path.join(self.data_root, phase + 'A'))  # 活体人脸放在trainA目录下
        self.l_img_dirs = [os.path.join(self.data_root, phase + 'A', img_name) for img_name in l_img_names]
        # spoof images
        s_img_names = os.listdir(os.path.join(self.data_root, phase + 'B'))  # 欺骗人脸放在trainB目录下
        self.s_img_dirs = [os.path.join(self.data_root, phase + 'B', img_name) for img_name in s_img_names]
        self.image_dirs = self.l_img_dirs + self.s_img_dirs
        # depth images
        d_limg_names = os.listdir(os.path.join(self.data_root, phase + 'AD'))
        self.d_limg_dirs = [os.path.join(self.data_root, phase + 'AD', img_name) for img_name in d_limg_names]
        if phase == 'train':
            d_simg_names = os.listdir(os.path.join(self.data_root, phase + 'BD'))
            self.d_simg_dirs = [os.path.join(self.data_root, phase + 'BD', img_name) for img_name in d_simg_names]
            self.depth_dirs = self.d_limg_dirs + self.d_simg_dirs
            
        self.len_l_imgs = len(self.l_img_dirs)
        self.len_s_imgs = len(self.s_img_dirs)
                
        self.dataset_size = self.len_l_imgs + self.len_s_imgs       
        # transform
        self.transform_train = Compose([
            RandomHorizontalFlip(0.5),
            # Pad(2, 0, 0, 'edge'),
            RandomRotation(2),  # 中心旋转
            ColorJitter(0.2, 0.2, 0.2, 0),
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),  # 维度是[256, 256]的depth map会变成[1, 256, 256]
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_test = Compose([           
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),            
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        mix_image = Image.open(self.image_dirs[idx]).convert('RGB')
        rdm_num = random.randint(0, self.len_l_imgs-1)
        live_image = Image.open(self.l_img_dirs[rdm_num]).convert('RGB')  # 随机选一个
        label = 1
        if idx > self.len_l_imgs-1:
            label = 0  # 欺骗人脸标签为0

        if self.phase == 'train':
            depth_map = Image.open(self.depth_dirs[idx]).convert('L') 
            live_image, _ = self.transform_train(live_image)
            mix_image, depth_map = self.transform_train(mix_image, depth_map)
            return live_image, mix_image, depth_map, label
        else:
            if label == 1:
                depth_map = Image.open(self.d_limg_dirs[idx]).convert('L')  
            else:
                depth_map = Image.new('L', (self.params['image_size'], self.params['image_size']), color=0).convert('L')
            live_image, _ = self.transform_test(live_image)  # 此时depth_image为None
            mix_image, depth_map = self.transform_test(mix_image, depth_map)
            return live_image, mix_image, depth_map, label
        

class MixDataset(Dataset):
    def __init__(self, params, phase='train'):
        super(MixDataset, self).__init__()
        self.data_root = params['data_root']
        self.phase = phase
        self.params = params

        # live images
        l_img_names = os.listdir(os.path.join(self.data_root, phase + 'A'))  # 活体人脸放在trainA目录下
        self.l_img_dirs = [os.path.join(self.data_root, phase + 'A', img_name) for img_name in l_img_names]
        # spoof images
        s_img_names = os.listdir(os.path.join(self.data_root, phase + 'B'))  # 欺骗人脸放在trainB目录下
        self.s_img_dirs = [os.path.join(self.data_root, phase + 'B', img_name) for img_name in s_img_names]
        self.image_dirs = self.l_img_dirs + self.s_img_dirs
        # depth images
        d_limg_names = os.listdir(os.path.join(self.data_root, phase + 'AD'))
        self.d_limg_dirs = [os.path.join(self.data_root, phase + 'AD', img_name) for img_name in d_limg_names]
        d_simg_names = os.listdir(os.path.join(self.data_root, phase + 'BD'))
        self.d_simg_dirs = [os.path.join(self.data_root, phase + 'BD', img_name) for img_name in d_simg_names]
        self.depth_dirs = self.d_limg_dirs + self.d_simg_dirs

        self.len_l_imgs = len(self.l_img_dirs)
        self.len_s_imgs = len(self.s_img_dirs)
                
        self.dataset_size = self.len_l_imgs + self.len_s_imgs       
        # transform
        self.transform_train = Compose([
            RandomHorizontalFlip(0.5),
            # Pad(2, 0, 0, 'edge'),
            # RandomRotation(0),  # 中心旋转
            ColorJitter(0.2, 0.2, 0.2,0),
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),  # 维度是[256, 256]的depth map会变成[1, 256, 256]
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_test = Compose([           
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),            
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        mix_image = Image.open(self.image_dirs[idx]).convert('RGB')
        if idx > self.len_l_imgs-1:
            depth_map = Image.new('L', (self.params['image_size'], self.params['image_size']), color=0).convert('L')
            label = 0  # 欺骗人脸标签为0
        else:
            depth_map = Image.open(self.depth_dirs[idx]).convert('L') 
            label = 1

        if self.phase == 'train':
            mix_image, depth_map = self.transform_train(mix_image, depth_map)
            return mix_image, depth_map, label
        else:
            mix_image, depth_map = self.transform_test(mix_image, depth_map)
            return mix_image, depth_map, label



class TestDataset(Dataset):
    '''返回mix_image和label, 用于计算acer'''
    def __init__(self, params, phase='test'):
        super(TestDataset, self).__init__()
        self.data_root = params['data_root']
        self.phase = phase
        self.params = params

        # live images
        l_img_names = os.listdir(os.path.join(self.data_root, phase + 'A'))  # 活体人脸放在trainA目录下
        self.l_img_dirs = [os.path.join(self.data_root, phase + 'A', img_name) for img_name in l_img_names]
        # spoof images
        s_img_names = os.listdir(os.path.join(self.data_root, phase + 'B'))  # 欺骗人脸放在trainB目录下
        self.s_img_dirs = [os.path.join(self.data_root, phase + 'B', img_name) for img_name in s_img_names]
        self.image_dirs = self.l_img_dirs + self.s_img_dirs

        self.len_l_imgs = len(self.l_img_dirs)
        self.len_s_imgs = len(self.s_img_dirs)
                
        self.dataset_size = self.len_l_imgs + self.len_s_imgs       
        # transform
        self.transform_train = Compose([
            RandomHorizontalFlip(0.5),
            # Pad(2, 0, 0, 'edge'),
            # RandomRotation(0),  # 中心旋转
            ColorJitter(0.2, 0.2, 0.2,0),
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),  # 维度是[256, 256]的depth map会变成[1, 256, 256]
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_test = Compose([           
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),            
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        mix_image = Image.open(self.image_dirs[idx]).convert('RGB')
        if idx > self.len_l_imgs-1:
            label = 0  # 欺骗人脸标签为0
        else:
            label = 1

        if self.phase == 'train':
            mix_image, _ = self.transform_train(mix_image)
            return mix_image, label
        else:
            mix_image, _ = self.transform_test(mix_image)
            return mix_image, label

class DisplayDataset(Dataset):
    '''用来展示不同类别的欺骗攻击在潜在特征空间中的分布'''
    def __init__(self, params,phase='test'):
        
        super(DisplayDataset, self).__init__()
        self.data_root = params['data_root']
        self.PAI_dict = {'print1':1, 'print2':2, 'replay1':3, 'replay2':4}

        self.transform_test = Compose([           
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),            
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # live images
        self.l_img_names = os.listdir(os.path.join(self.data_root, phase + 'A'))  # 活体人脸放在testA目录下
        self.l_img_dirs = [os.path.join(self.data_root, phase + 'A', img_name) for img_name in self.l_img_names]
        # spoof images
        self.s_img_names = os.listdir(os.path.join(self.data_root, phase + 'B'))  # 欺骗人脸放在testB目录下
        self.s_img_dirs = [os.path.join(self.data_root, phase + 'B', img_name) for img_name in self.s_img_names]
        self.names = self.l_img_names + self.s_img_names
        self.image_dirs = self.l_img_dirs + self.s_img_dirs
        self.len_imgs = len(self.image_dirs)
    
    def __len__(self):
        return self.len_imgs

    def __getitem__(self, idx):
        image = Image.open(self.image_dirs[idx]).convert('RGB')
        image, _ = self.transform_test(image)
        if idx > len(self.l_img_dirs) -1:
            name = self.names[idx].split('_')[1]
            label = self.PAI_dict.get(name)
        else:
            label = 0
        return image, label

               

class DepthDataset(Dataset):
    '''训练深度预测模型需要的dataset'''
    def __init__(self, params, phase='train'):
        super(DepthDataset, self).__init__()
        self.data_root = params['data_root']
        self.phase = phase
        self.params = params
        
        # live images
        l_img_names = os.listdir(os.path.join(self.data_root, phase + 'A'))  # 活体人脸放在trainA目录下
        self.l_img_dirs = [os.path.join(self.data_root, phase + 'A', img_name) for img_name in l_img_names]
        # spoof images
        s_img_names = os.listdir(os.path.join(self.data_root, phase + 'B'))  # 欺骗人脸放在trainB目录下
        self.s_img_dirs = [os.path.join(self.data_root, phase + 'B', img_name) for img_name in s_img_names]
        # depth images
        d_limg_names = os.listdir(os.path.join(self.data_root, phase + 'AD'))
        self.d_limg_dirs = [os.path.join(self.data_root, phase + 'AD', img_name) for img_name in d_limg_names]
        # d_simg_names = os.listdir(os.path.join(self.data_root, phase + 'BD'))
        # self.d_simg_dirs = [os.path.join(self.data_root, phase + 'BD', img_name) for img_name in d_simg_names]

        self.image_dirs = self.l_img_dirs + self.s_img_dirs
        # self.depth_dirs = self.d_limg_dirs + self.d_simg_dirs

        self.len_l_imgs = len(self.l_img_dirs)
        self.len_s_imgs = len(self.s_img_dirs)
        self.dataset_size = self.len_l_imgs + self.len_s_imgs

        # transform
        self.transform_train = Compose([
            Resize((params['image_size'], params['image_size'])),
            RandomHorizontalFlip(0.5),
            RandomRotation(2),  # 中心旋转
            ColorJitter(0.2, 0.2, 0.2,0),
            ToTensor(),  # 维度是[256, 256]的depth map会变成[1, 256, 256]
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_test = Compose([           
            Resize((params['image_size'], params['image_size'])),
            ToTensor(),            
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        mix_image = Image.open(self.image_dirs[idx]).convert('RGB')
        # depth_map = Image.open(self.depth_dirs[idx]).convert('L')   
        if idx > self.len_l_imgs-1:
            depth_map = Image.new('L', (self.params['image_size'], self.params['image_size']), color=0).convert('L')
        else:
            depth_map = Image.open(self.d_limg_dirs[idx]).convert('L')

        if self.phase == 'train':
            mix_image, depth_map = self.transform_train(mix_image,depth_map)
            return mix_image, depth_map
        else:
            mix_image, depth_map = self.transform_test(mix_image,depth_map)
            return mix_image, depth_map



