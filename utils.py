# -*- coding: utf-8 -*-

"""
Created on 2021/10/5 10:00
@author: Acer
@description: 
"""

import torch
import yaml
import json
from torch.optim import lr_scheduler
import torch.nn.init as init
import math
import os
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
# from tsne_torch import TorchTSNE as TSNE
import torch.nn.functional as F
import numpy as np
import time
from torchvision import transforms as T
import random
import numpy.ma as ma


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def get_json(path):
    with open(path, 'r') as stream:
        return json.load(stream)


def get_scheduler(optim, params, last_epoch=-1):
    if 'lr_policy' not in params or params['lr_policy'] == 'constant':
        scheduler = None
    elif params['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optim, step_size=params['step_size'], gamma=params['gamma'],last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', params['lr_policy'])
    return scheduler


def weight_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

# Get model list for resume
def get_model_list(dirname):
    '''
    在checkpoint目录下保存的所有模型中，找到最后保存的那个模型的名字，便于恢复训练时候使用
    :param dirname: 模型在目录
    :return: 最后保存的模型所在路径（包括模型名称在内）
    '''
    if os.path.exists(dirname) is False:
        return None
    # gen_models是模型所在目录+名称的列表
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if  # os.listdir(dirname)表示这个目录下所有文件/文件夹的名字
                  os.path.isfile(os.path.join(dirname, f)) and ".pt" in f]  # 本文模型保存的名字虽然是pth（见train.调用的save函数），但是不影响
    if gen_models is None:
        return None
    gen_models.sort()  # 模型保存的时候是按照epoch保存的，所以对模型名称进行排序
    last_model_name = gen_models[-1]  # 对排序后的模型取最后一个，就是最新的模型
    return last_model_name


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    early_stop_model_directory = os.path.join(output_directory, 'early_stop_model')
    if not os.path.exists(early_stop_model_directory):
        print("Creating directory: {}".format(early_stop_model_directory))
        os.makedirs(early_stop_model_directory)
    return checkpoint_directory, image_directory, early_stop_model_directory

def prepare_score_folder(output_directory):
    score_directory = os.path.join(output_directory, 'test_score')
    if not os.path.exists(score_directory):
        print("Creating directory: {}".format(score_directory))
        os.makedirs(score_directory)
    return score_directory

def write_lr(epoch, trainer, train_writer):
    train_writer.add_scalar('lr_of_gen', trainer.gen_opt.state_dict()['param_groups'][0]['lr'], epoch)

def write_loss(iterations, trainer, train_writer, phase='train'):
    members = [attr for attr in dir(trainer) if not callable(getattr(trainer, attr)) and not attr.startswith("__") and 'loss' in attr]
    # print(members)
    # print(getattr(trainer, 'cacl_livecode_dis_loss'))
    for m in members:
        train_writer.add_scalar(phase + '/' + m, getattr(trainer, m), iterations + 1)
        # print(m)

# def display_embedding(embedding, labels, writer, iter, phase='train'):
#     '''
#         将潜在特征空间的向量可视化出来
#         embedding维度：[N, X]
#     '''
#     X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(embedding)  # returns shape (n_samples, 2)
#     fig = plt.figure(dpi=150)
#     plt.scatter(X_emb[:, 0], X_emb[:, 1], 20, c=labels)
#     writer.add_figure(phase + '/enc_embedding', fig, iter)


def display_images(image_outputs, display_image_num, writer, iter, phase='train'):
    '''
        image_outputs:[display_image_num张input_image,display_image_num张gen_image,display_image_num张depth_map....],
        具体见trainer.sample返回的image
    '''
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    scale_each=False
    # traces = get_trace(image_outputs[4], image_outputs[1])
    # image_outputs[2] = traces  # 将原来trainer.sample返回的trace用处理后的trace替换下来
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)  # 就是把上面的list转成tensor        
    # normalize=True,表示shift the image to the range (0, 1).  scale_each=True表示对batchsize的每一张图片分别计算改图的最大值最小值进行归一化
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True, scale_each=scale_each)
    writer.add_image(phase, image_grid, iter)

def __write_images(image_outputs, display_image_num, file_name):
    '''
    保存images
    :param image_outputs: 要么是网络返回的有关limage的图片，要么是有关mimage的图片
    :param display_image_num: 网络返回的图片数量
    :param file_name:
    :return:
    '''
    # print(image_outputs[0].size())  # torch.Size([16, 3, 256, 256])
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    # traces = get_trace(image_outputs[4], image_outputs[1])
    # image_outputs[2] = traces  # 将原来trainer.sample返回的trace用处理后的trace替换下来
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)  # 就是把上面的list转成tensor
    # print(image_tensor.size())  # [网络返回的图片种类数,display_image_num,256,256]
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True, scale_each=False)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    '''
    保存图片输入到网络中后，生成的图片的结果
    :param image_outputs: 网络生成的images,包括limage相关的图片和mimage相关的图片，各占image_outputs列表的一半
    :param display_image_num: 每种图片要保存数量，比如输入到网络中16张limage和16张mlimage，那网络生成的每种图片都有16张
    :param image_directory:
    :param postfix: 文件名。 'train_%08d' % (total_iter + 1)或者是'test_%08d' % (total_iter + 1)
    :return:
    '''
    n = len(image_outputs)
    __write_images(image_outputs, display_image_num, '%s/gen_%s.jpg' % (image_directory, postfix))
    # __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_m2l_%s.jpg' % (image_directory, postfix))


def plot_embedding_2D(data, label, title, save_path):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    colors = ['#ff0000', '#00ff00']  
    for i in range(data.shape[0]):
        if (int(label[i])) == 0:
            spoof = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
        else:
            live = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
        
    plt.legend([live, spoof],['live face', 'spoof face'], loc = 'upper right')  # 图注
    # plt.axes().get_xaxis().set_visible(False) # 隐藏x坐标轴   
    # plt.axes().get_yaxis().set_visible(False) # 隐藏y坐标轴
    plt.title(title)
    # 保存必须在show之前，否则保存的图片是空白
    plt.savefig(os.path.join(save_path, 'test_score/' + title + '.jpg'), dpi = 720)  
    # plt.show()
    
    return fig

def plot_embedding_3D(data,label,title, save_path): 
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0) 
    data = (data- x_min) / (x_max - x_min) 
    fig = plt.figure()
    colors = ['#ff0000', '#00ff00']  
    ax = fig.add_subplot(111,projection='3d') 
    for i in range(data.shape[0]): 
        if (int(label[i])) == 0:
            spoof = plt.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[int(label[i])])
        else:
            live = plt.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[int(label[i])])  
    plt.legend([live, spoof], ['live face', 'spoof face'], loc = 'best')  # 图注
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'test_score/' + title + '_3d_.jpg'), dpi = 720)
    # plt.show()


def plot_PAI_2D(data, label, title, save_path, protocol="protocol_1"):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    colors = ['#00ff00', '#ff00ff', '#990000', '#999900', '#009999'] 
    # print(protocol)
    if protocol == 'protocol_1': 
        for i in range(data.shape[0]):
            if (int(label[i])) == 0:
                live = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 1:
                print1 = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 2:
                print2 = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 3:
                replay1 = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 4:
                replay2 = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
        plt.legend([live, print1, print2, replay1, replay2],['live face', 'print1','print2','replay1','replay2'], loc = 'upper right')  # 图注
    elif protocol == 'protocol_2':  # 协议2测试集合中没有print1和replay1
        for i in range(data.shape[0]):
            if (int(label[i])) == 0:
                live = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 2: # print2
                print2 = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 4: # replay2
                replay2 = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
        plt.legend([live, print2, replay2],['live face', 'print2','replay2'], loc = 'upper right') 

    # plt.axes().get_xaxis().set_visible(False) # 隐藏x坐标轴   
    # plt.axes().get_yaxis().set_visible(False) # 隐藏y坐标轴
    plt.title(title)
    # 保存必须在show之前，否则保存的图片是空白
    plt.savefig(os.path.join(save_path, 'test_score/' + title + '.jpg'), dpi = 720)  
    # plt.show()
    
    return fig

def plot_PAI_2D_onTrainTest(data, label, title, save_path, protocol='protocol_1'):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    if protocol == 'protocol_1':
        colors = [
            '#00ff00', '#ff00ff', '#990000', '#999900', '#009999', # 测试集标签对应颜色
            '#00aa00', '#aa00aa', '#770000', '#777700', '#007777'  #训练集标签对应颜色
            ]  
        for i in range(data.shape[0]):
            if (int(label[i])) == 0:
                live_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 1:
                print1_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 2:
                print2_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 3:
                replay1_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 4:
                replay2_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            
            elif (int(label[i])) == 5:
                live_train = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 6:
                print1_train = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 7:
                print2_train = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 8:
                replay1_train = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 9:
                replay2_train = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            else:
                print('utilis.plot_PAI_2D_onTrainTest: 检查标签类别是否过多')           
        plt.legend([live_test, print1_test, print2_test, replay1_test, replay2_test, live_train, print1_train, print2_train, replay1_train, replay2_train],
            ['live_test', 'print1_test','print2_test','replay1_test','replay2_test',
            'live_train', 'print1_train','print2_train','replay1_train','replay2_train'], 
            loc = 'best')  # 图注
    elif protocol == 'protocol_2':
        colors = [
        '#00ff00', '#ff00ff', '#990000', '#999900', '#009999', '#00aa00', # 测试集标签对应颜色
        # '#00aa00', '#aa00aa', '#770000', '#777700', '#007777'  #训练集标签对应颜色
        ]  
        for i in range(data.shape[0]):
            if (int(label[i])) == 0:
                live_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 1:
                print1_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 2:
                print2_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 3:
                replay1_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 4:
                replay2_test = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            elif (int(label[i])) == 5: # 训练集的真实人脸
                live_train = plt.scatter(data[i, 0], data[i, 1], color=colors[int(label[i])])
            else:
                print('utilis.plot_PAI_2D_onTrainTest: 检查标签类别是否过多')           
        plt.legend([live_test, print1_test, print2_test, replay1_test, replay2_test, live_train],
            ['live_test', 'print1_train','print2_test','replay1_train','replay2_test', 'live_train',
            # 'live_train', 'print1_train','print2_train','replay1_train','replay2_train'
            ], 
            loc = 'best') 
    # plt.axes().get_xaxis().set_visible(False) # 隐藏x坐标轴   
    # plt.axes().get_yaxis().set_visible(False) # 隐藏y坐标轴
    plt.title(title)
    # 保存必须在show之前，否则保存的图片是空白
    plt.savefig(os.path.join(save_path, 'test_score/' + title + '.jpg'), dpi = 720)  
    # plt.show()
    
    return fig

def plot_PAI_3D(data,label,title, save_path): 
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0) 
    data = (data- x_min) / (x_max - x_min) 
    fig = plt.figure()
    colors = ['#00ff00', '#ff00ff', '#990000', '#999900', '#009999']   
    ax = fig.add_subplot(111,projection='3d') 
    for i in range(data.shape[0]): 
        if (int(label[i])) == 0:
            live = plt.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[int(label[i])])
        elif (int(label[i])) == 1:
            print1 = plt.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[int(label[i])])
        elif (int(label[i])) == 2:
            print2 = plt.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[int(label[i])])
        elif (int(label[i])) == 3:
            replay1 = plt.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[int(label[i])])
        else :
            replay2 = plt.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[int(label[i])])
            
    plt.legend([live, print1, print2, replay1, replay2],['live face', 'print1','print2','replay1','replay2'], loc = 'upper right')  # 图注
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'test_score/' + title + '_3d_.jpg'), dpi = 720)
    # plt.show()


def display_evaluation(cls_out, lb, writer, epoch):
    test_cont = 0
    num_real = 0
    err_cls_real = 0
    num_fake = 0
    err_cls_fake = 0
    
    test_cls_out = torch.softmax(cls_out, dim=1)
    pre_cls_lb = torch.max(test_cls_out, dim=1)[1]
    test_cont += (pre_cls_lb != lb).sum().item()
    for i in range(len(lb)):
        if lb[i] == 1:
            num_real += 1
            if pre_cls_lb[i] != lb[i]:
                err_cls_real += 1
        else:
            num_fake += 1
            if pre_cls_lb[i] != lb[i]:
                err_cls_fake += 1
    apcer = np.around(100*err_cls_fake/num_fake, 4)
    bpcer = np.around(100*err_cls_real/num_real, 4)
    acer = np.around((apcer+bpcer)/2, 4)
    writer.add_scalar('test/apcer' , apcer, epoch)
    writer.add_scalar('test/bpcer' , bpcer, epoch)
    writer.add_scalar('test/acer' , acer, epoch)

###########################################################################
#------------------------------evaluation----------------------------------
###########################################################################
#-------------------------find proper threshold---------------------------#
def calcu_scores(inputs, p=1):
    '''
        inputs 类型：tensor
        ||x||/(N*N)
    '''
    inputs = torch.sigmoid(inputs)  # 将输入归一化到0-1
    scores = []
    for i in range(inputs.size(0)):
        image = inputs[i]
        score = torch.norm(image, p=p)/(image.size(0)*image.size(1)*image.size(2))
        score = score.cpu().detach().numpy()  # tensor转np，
        scores.append(score)
    
    return scores

def calcu_spoof_trace_score(spoof_traces, p=1):
    inputs = torch.sigmoid(spoof_traces)  # 将输入归一化到0-1
    scores = []
    for i in range(inputs.size(0)):
        trace = inputs[i]
        score = torch.norm(trace, p=p)/(trace.size(0)*trace.size(1)*trace.size(2))
        # score = -torch.log(score)
        score = score.cpu().detach().numpy()  # tensor转np
        scores.append(score)
    
    return scores

def calcu_final_score(depth_scores, patch_scores=None, trace_scores=None):
    scores = []
    if patch_scores is not None and trace_scores is not None:
        for i in range(len(depth_scores)):
            score = depth_scores[i] + patch_scores[i] - 0.5*trace_scores[i]
            # score = np.around(score, 2)  # tensor转np，同时保留两位有效数字（四舍五入）
            scores.append(score)
    else:
        # 只用深度图计算score
        for i in range(len(depth_scores)):
            score = depth_scores[i]
            # score = np.around(score, 2)  # tensor转np，同时保留两位有效数字（四舍五入）
            scores.append(score)
    return scores

def write_scores(file, scores, img_nums, PAIs):
    '''
        PAI: presentation attack instrument,e.g. display,print, if the input is live face, PAI=None
        file item format: spoof face:imgnum_PAI_score, live face: imgnum_score
    '''
    with open(file, encoding="utf-8",mode="a") as f:  
        for i in range(len(scores)):
            content = str(img_nums[i]+'_'+ PAIs[i]+'_'+str(scores[i]))
            f.write(content + '\n')  

def read_score(dir):
    '''拼接真假人脸的分数，并排序返回'''
    scores = []
    with open(os.path.join(dir, 'live_scores.txt'), encoding="utf-8") as f:  
        for line in f:
            score = line.split('_')[-1].rstrip("\n") 
            scores.append(score)
    with open(os.path.join(dir, 'spoof_scores.txt'), encoding="utf-8") as f:  
        for line in f:
            score = line.split('_')[-1] 
            scores.append(score)
    scores.sort(reverse=False)  # 升序
    return scores
            

def calcu_APCER(attack_file, threshold, use_PAI=False):
    '''use_PAI：是否根据PAI的种类计算出不同的分数'''
    if use_PAI == True:
        PAIs_dict = {}  # {PAI: [记录该PAI的总数, 记录分类错误的数量]}
        with open(attack_file, encoding="utf-8") as f:  
            for line in f:
                img_num, PAI, score = line.split('_') 
                if PAI not in PAIs_dict.keys():
                    # 假设score大于threshold，则判别为真
                    # 如果score>=threshhold, 说明该spoof face被分类错误，所以分类错误数量要置为1
                    PAIs_dict[PAI] = [1, 1 if float(score) >= float(threshold) else 0] 
                else:  # 总数在原来基础上+1，分类错误数如果分类错误就+1，否则不变
                    PAIs_dict[PAI] = [PAIs_dict[PAI][0]+1, PAIs_dict[PAI][1]+1 if float(score) >= float(threshold) else PAIs_dict[PAI][1]]
        APCERs = {}
        for key, value in PAIs_dict.items():
            APCERs[key] = round(100*value[1]/value[0], 2)  # 保留两位有效数字
        # print(APCERs)
        APCER = max(APCERs.values())
        return APCER, APCERs
    else:
        total_num, err_cls_num = 0, 0
        with open(attack_file, encoding="utf-8") as f:  
            for line in f:
                total_num += 1
                img_num, PAI, score = line.split('_') 
                if float(score) >= float(threshold):
                    err_cls_num += 1
        APCER = round(100*err_cls_num/total_num, 2)
        return APCER

def calcu_BPCER(live_file, threshold):
    '''threshold:str类型'''
    total, err_cls = 0, 0  # 样本总数、错误分类数
    with open(live_file, encoding="utf-8") as f:  
        for line in f:
            total += 1
            img_num, _, score = line.split('_') 
            if float(score) < float(threshold):  # 如果live face被分类为spoof
                err_cls += 1

    BPCER = round(100*err_cls/total, 2)
    return BPCER

def calcu_ACER(BPCER, APCER):
    return round((BPCER+APCER)/2, 2)


#-------------------------------no threshold-------------------------------#
# 如果网络直接给出分类结果而不需要阈值时,就可以使用这个评估方式
def calcu_BPCER_NO_THRESH(trainer, test_loader, device, path=None):
    start_time = time.perf_counter()
    err_cls = 0  # 分类错误的个数
    total = len(test_loader.dataset)
    for iter, (images, img_num, PAI)  in enumerate(test_loader):
        with torch.no_grad():
            images = images.to(device)
            _, predicts = trainer.forward(images)
            #########计算分类分数##########
            cls_score = torch.softmax(predicts, dim=1)
            pre_cls = torch.argmax(cls_score, dim=1)  # 可以改为# pre_cls = torch.max(predicts, dim=1)[1]  # batch size中每个样本分类的分类结果
            labels = torch.ones_like(pre_cls).to(device)
            err_cls += (pre_cls != labels).sum().item()
            #########保存分类错误的样本序号，便于查看哪些图片分类错误##########
            if path is not None:
                store_err_cls_imgnum(pre_cls, labels, img_num, path)
            test_time = time.perf_counter() - start_time
            rate = (iter + 1) / len(test_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\r{:^3.0f}%[{}->{}]{:.2f}s,cur_iter:{}".format(int(rate * 100), a, b, test_time, iter), end="")
    print()
    BPCER = round(100*err_cls/total, 2)
    return BPCER

def calcu_APCER_NO_THRESH(trainer, test_loader, device, path=None):
    '''show_emb是否可视化样本潜在特征空间, emb_num潜在特征空间样本数量'''
    start_time = time.perf_counter()
    err_cls = 0  # 分类错误的个数
    total = len(test_loader.dataset)
    for iter, (images, img_num, PAI)  in enumerate(test_loader):
        with torch.no_grad():
            images = images.to(device)
            _, predicts = trainer.forward(images)  
            #########计算分类分数##########
            cls_score = torch.softmax(predicts, dim=1)
            pre_cls = torch.argmax(cls_score, dim=1)  # 可以改为# pre_cls = torch.max(predicts, dim=1)[1]  # batch size中每个样本分类的分类结果
            print(pre_cls)
            labels = torch.zeros_like(pre_cls).to(device)
            err_cls += (pre_cls != labels).sum().item()
            #########保存分类错误的样本序号，便于查看哪些图片分类错误##########
            if path is not None:
                store_err_cls_imgnum(pre_cls, labels, img_num, path)
            test_time = time.perf_counter() - start_time
            rate = (iter + 1) / len(test_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\r{:^3.0f}%[{}->{}]{:.2f}s,cur_iter:{}".format(int(rate * 100), a, b, test_time, iter), end="")
    print()
    APCER = round(100*err_cls/total, 2)
    return APCER


def store_err_cls_imgnum(predict, labels, img_num, save_path):
    mask = torch.eq(predict, labels).cpu().numpy()  # mask表示要隐藏（要去掉的索引，所以要去掉分类正确的项目）
    err_cls_imgnum = ma.masked_array(img_num, mask)  
    err_cls_imgnum=err_cls_imgnum.tolist()
    img_num = []
    for i in range(len(err_cls_imgnum)):
        if err_cls_imgnum[i] is not None:
            img_num.append(err_cls_imgnum[i])
    str = '\n'
    f=open(save_path,encoding="utf-8",mode="a")
    f.write(str.join(img_num) + '\n')

###########################################################################
#------------------------------show embeding-------------------------------
###########################################################################
def get_live_embedding(trainer, live_loader, device, sample_num=100):
    '''iter_num: 迭代次数.即返回的emb的个数是iter_num*batch_size'''
    live_num = 0  # 记录已存数据数量
    pai_dict = {}   # {pai: [[sample_num, emb_dim], labels}
    labels = []
    for iter, (images, img_num, PAIs)  in enumerate(live_loader):
        with torch.no_grad():
            images = images.to(device)
            predicts = trainer.forward(images)  # forward返回的最后一个是分类分数,倒数第二个是emb_out
            emb_outs = predicts[-2]  # [bs, 512]
            for i in range(emb_outs.size(0)):  # bs个emb一个一个的取出来
                emb_out = emb_outs[i]
                emb_out = emb_out.unsqueeze(0)
                if live_num == 0:  # 如果还没有数据记录到pai_diict中
                    labels.append(0)
                    # print(emb_out.shape)
                    pai_dict[PAIs[i]] = [emb_out, labels]
                    
                    live_num += 1
                elif live_num < sample_num:
                    labels.append(0)
                    pai_dict[PAIs[i]] = [torch.cat((pai_dict[PAIs[i]][0], emb_out), dim=0), labels]             
                    live_num += 1
                else:
                    return pai_dict

def get_spoof_embedding(trainer, spoof_loader, device, iter_num=10):
    '''iter_num: 迭代次数.即返回的emb的个数是iter_num*batch_size'''
    # live_num = 0  # 记录已存数据数量
    pai_dict = {}   # {pai: [embs, labels], pai2:[embs, labels]}
    labels = []
    pai_label_dict = {}  # {pai: [label,...]}
    label = 1  # 不同PAI的label不同,发现一种PAI,label就+1,相当于为每一种pai设置一个编号
    for iter, (images, img_num, PAIs)  in enumerate(spoof_loader):
        if iter < iter_num:
            with torch.no_grad():
                images = images.to(device)
                predicts = trainer.forward(images)  # forward返回的最后一个是分类分数,倒数第二个是emb_out
                emb_outs = predicts[-2]  # [bs, 512]
                for i in range(emb_outs.size(0)):  # bs个emb一个一个的取出来
                    emb_out = emb_outs[i]  # [512]
                    emb_out = emb_out.unsqueeze(0)
                    pai = PAIs[i]
                    if pai not in pai_dict.keys():  # 如果还没有数据记录到pai_diict中
                        pai_label_dict[pai] = []  # 设置字典的value是列表
                        pai_label_dict[pai].append(label)
                        pai_dict[pai] = [emb_out, pai_label_dict[pai]]  # 第一次数显pai,为其分配一个编号最为emb_out的label
                        label += 1  # 下一个新的pai到来时,使用+1后的label作为该新的pai的编号
                    else:
                        pai_label_dict[pai].append(pai_label_dict[pai][0])
                        pai_dict[pai] = [torch.cat((pai_dict[pai][0], emb_out), dim=0), pai_label_dict[pai]]
        else:
            return pai_dict            




if __name__ == '__main__':
    config = get_config('./configs/oulu_npu.yaml')
    # print(json.dumps(config, sort_keys=False, indent=4))
    # with open('./configs/options.json', 'w') as f:
    #     f.write(json.dumps(config, sort_keys=False, indent=4))
    # from trainer import *
    # trainer = DLSR(config)
    # write_loss(1, trainer, 1)