# -*- coding: utf-8 -*-

"""
Created on 2021/10/7 15:33
@author: Acer
@description: 
"""

from torch.utils.data.sampler import WeightedRandomSampler
from depth_trainer import Depth_trainer
from utils import *
from depth_trainer import Depth_trainer
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from datasets import *
import time
from pytorchtools import EarlyStopping
import numpy as np
import torch.nn.functional as fun


parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
parser.add_argument('--output_path', type=str, default='../autodl-tmp', help="outputs path")  # ../autodl-tmp
parser.add_argument('--model_save_path', type=str, default='../autodl-tmp', help="pre-trained depth model save path")  # ../autodl-tmp
opts = parser.parse_args()
# Load experiment setting
config = get_config('./configs/oulu_npu.yaml')
display_size = config['display_size']
# Setup logger and output folders
# 哪个数据集训练模型就以哪个数据集的yaml文件名为模型名称，模型名称用于保存数据的时候便于对不同数据集的输出结果保存到不同目录
model_name = os.path.splitext(os.path.basename('./configs/oulu_npu.yaml'))[0]
print('############################################################')
print('current training dataset:\t%s' % model_name)
train_writer = SummaryWriter(os.path.join("logs", "TAD_" + model_name))  # ../tf-logs
output_directory = os.path.join(opts.output_path, model_name + "_TAD/" + config['protocol'])
checkpoint_directory, image_directory, depth_model_directory = prepare_sub_folder(output_directory)
# shutil.copy('configs/oulu_npu.yaml', os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:\t\t\t\t%s' % device)

# set data loader
print('load dataset\t\t\t%s' % model_name)
# sample weight
trainA_len = len(os.listdir(os.path.join(config['data_root'], 'trainA')))
trainB_len = len(os.listdir(os.path.join(config['data_root'], 'trainB')))
train_len = trainA_len + trainB_len
print('lenght of train dataset\t\t{}'.format(train_len))
limg_sample_w = train_len / trainA_len
simg_sample_w = train_len / trainB_len
sample_l_w = [limg_sample_w] * trainA_len
sample_s_w = [simg_sample_w] * trainB_len
train_sample_w = sample_l_w + sample_s_w
train_sampler = WeightedRandomSampler(train_sample_w, min(trainA_len, trainB_len), replacement=False)

testA_len = len(os.listdir(os.path.join(config['data_root'], 'testA')))
testB_len = len(os.listdir(os.path.join(config['data_root'], 'testB')))
test_len = testA_len + testB_len
print('lenght of test dataset\t\t{}'.format(test_len))
limg_sample_w = test_len / testA_len
simg_sample_w = test_len / testB_len
sample_l_w = [limg_sample_w] * testA_len
sample_s_w = [simg_sample_w] * testB_len
test_sample_w = sample_l_w + sample_s_w
test_sampler = WeightedRandomSampler(test_sample_w, min(testA_len, testB_len), replacement=False)

train_dataset = DepthDataset(config, phase='train')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, 
                num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler)
test_dataset = DepthDataset(config, phase='test')
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, 
                num_workers=config['num_workers'], pin_memory=True, sampler=test_sampler)

print('protocol\t\t\t', config['protocol'])
print('load model to device')
trainer = Depth_trainer(config).to(device)
if opts.resume == False:
    trainer.initnetworks()
    start_epoch = -1
    total_iter = 0
else:
    start_epoch, total_iter = trainer.resume('depth_model/protocal_1', config)
start_epoch += 1
print('start the training at epoch\t %d' % (start_epoch))
print('depthnet learning rate:\t\t{}'.format(trainer.depth_opt.state_dict()['param_groups'][0]['lr']))
print('############################################################')

patience = 100
early_stopping = EarlyStopping(patience, verbose=True)

display_mimg = None
display_dimg = None
display_pdimg = None

for epoch in range(start_epoch, config['total_epoch']):
    start_time = time.perf_counter()
    for iter, (mix_images, depth_map) in enumerate(train_loader):
        # input data
        mix_images = mix_images.to(device).detach()
        depth_map = depth_map.to(device).detach()
        pre_depth_map = trainer.depth_updata(mix_images, depth_map)

        display_mimg = mix_images
        display_dimg = depth_map
        display_pdimg = pre_depth_map

        #######################输出日志#######################
        if (total_iter + 1) % config['log_iter'] == 0:
            # print("Iteration: %08d" % (total_iter + 1))
            write_loss(total_iter, trainer, train_writer, 'train')
        total_iter += 1
        train_time = time.perf_counter() - start_time
        rate = (iter + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\repoch:{:^5}/{:^5} {:^3.0f}%[{}->{}]{:.2f}s,cur_iter:{}".format(epoch, config['total_epoch'], int(rate * 100), a, b, train_time, total_iter), end="")
    print()
    # decay learning rate,每config['step_size']个epoch之后才会更新lr
    trainer.updata_learning_rate()
    #######################保存图片#######################
    with torch.no_grad():  # 将测试图片输入到网络中得到输出
        display_pdimg =  fun.interpolate(display_pdimg, (display_mimg.size(2), display_mimg.size(3)), mode='bilinear', align_corners=True)
        display_image = [display_mimg, display_dimg, display_pdimg]
        # write_2images保存的是最后一个iter的图像，如果数据集长度不能被config['batch_size']整除，那保存的图片和标签就会错位
        write_2images(display_image, config['batch_size'], image_directory, 'depth_train_%03d' % (epoch + 1))
    # 测试
    with torch.no_grad():
        depth_loss = []
        for iter, (mix_img, d_map) in enumerate(test_loader):
            mix_img = mix_img.to(device)
            d_map = d_map.to(device)
            valid_depth_loss, pre_map = trainer.depth_valid(mix_img, d_map)

            depth_loss.append(valid_depth_loss.item())
        avg_depth_loss = np.mean(depth_loss)
        train_writer.add_scalar('test/avg_depth_loss' , avg_depth_loss, epoch)
        early_stopping(avg_depth_loss, trainer, 'depth_model/protocal_1/final_model/'+str('%05d'%epoch)+'_final.pth')
        if early_stopping.early_stop:
            print("Early stopping")
            break
    with torch.no_grad():  # 将测试图片输入到网络中得到输出
        pre_map = fun.interpolate(pre_map, (mix_img.size(2), mix_img.size(3)), mode='bilinear', align_corners=True)
        display_image = [mix_img, d_map, pre_map]
        write_2images(display_image, 20, image_directory, 'depth_test_%03d' % (epoch + 1))
    if (epoch + 1) % config['model_save_freq'] == 0:
        print('save the model @ epoch %d' % (epoch))
        trainer.save('%s/%05d.pth' % ('depth_model/protocal_1', epoch), epoch, total_iter)

print('finished the training')








