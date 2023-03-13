# -*- coding: utf-8 -*-

"""
Created on 2021/10/7 15:33
@author: Acer
@description: 
"""

from torch.utils.data.sampler import WeightedRandomSampler
from utils import *
from trainer import TAD_trainer
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
from datasets import *
import time
from pytorchtools import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
parser.add_argument('--output_path', type=str, default='outputs', help="outputs path")  # ../autodl-tmp
opts = parser.parse_args()
# Load experiment setting
config = get_config('./configs/oulu_npu.yaml')
display_size = config['display_size']
# Setup logger and output folders
# 哪个数据集训练模型就以哪个数据集的yaml文件名为模型名称，模型名称用于保存数据的时候便于对不同数据集的输出结果保存到不同目录
model_name = os.path.splitext(os.path.basename('./configs/oulu_npu.yaml'))[0]
print('############################################################')
print('current training dataset:\t%s' % model_name)
train_writer = SummaryWriter(os.path.join("logs",'TAD_' + model_name))  # ../tf-logs
output_directory = os.path.join(opts.output_path, model_name + "_TAD/" + config['protocol'])
checkpoint_directory, image_directory, early_stop_model_directory = prepare_sub_folder(output_directory)
shutil.copy('configs/oulu_npu.yaml', os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

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
# train_sampler = WeightedRandomSampler(train_sample_w, 5, replacement=False)

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
# test_sampler = WeightedRandomSampler(test_sample_w, 5, replacement=False)

train_dataset = UnpairDataset(config, phase='train')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, 
                num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler)
display_dataset = UnpairDataset(config, phase='test')
display_loader = DataLoader(display_dataset, batch_size=config['test_bs'], shuffle=False, 
                num_workers=config['num_workers'], pin_memory=True, sampler=test_sampler)
test_dataset = TestDataset(config, phase='test')
test_loader = DataLoader(test_dataset, batch_size=config['test_bs'], shuffle=False, 
                num_workers=config['num_workers'], pin_memory=True) # , sampler=test_sampler
print('protocol\t\t\t', config['protocol'])
print('load model to device')
trainer = TAD_trainer(config).to(device)

if opts.resume == False:
    trainer.initnetworks(config['init'])
    start_epoch = -1
    total_iter = 0
else:
    start_epoch, total_iter = trainer.resume(checkpoint_directory, config)
start_epoch += 1
print('start the training at epoch\t %d' % (start_epoch))
print('gen learning rate:\t\t{}'.format(trainer.gen_opt.state_dict()['param_groups'][0]['lr']))
print('dis learning rate:\t\t{}'.format(trainer.dis_opt.state_dict()['param_groups'][0]['lr']))
print('############################################################')


patience = 100
early_stopping = EarlyStopping(patience, verbose=True, delta=0.01)

for epoch in range(start_epoch, config['total_epoch']):
    start_time = time.perf_counter()
    trainer.train()
    for iter, (live_images, mix_images, depth_map, label) in enumerate(train_loader):
        # input data
        live_images = live_images.to(device).detach()
        mix_images = mix_images.to(device).detach()
        depth_map = depth_map.to(device).detach()
        label = label.to(device).detach()
        for _ in range(config['dis_updata_freq']):
            trainer.dis_updata(live_images, mix_images, label, config)
        trainer.gen_updata(mix_images, depth_map, label, config)
        #######################输出日志#######################
        if (total_iter + 1) % config['log_iter'] == 0:
            write_loss(total_iter, trainer, train_writer, 'train')
        #######################保存图片#######################
        if (total_iter + 1) % config['image_save_iter'] == 0:  
            with torch.no_grad():  # 将测试图片输入到网络中得到输出(bug:当image_save_iter能整除iter+1时，这个epoch就会跳过)
                train_ds = train_loader.dataset  # 防止取出image和depth时各自都调用dataset从而导致image和depth水平翻转不一致
                rand_list = [random.randint(0, len(train_dataset)-1) for i in range(display_size)]
                # train_display_limages = torch.stack([train_ds[i][0] for i in rand_list]).cuda()  # 随机选择照片
                train_display_mimages = torch.stack([train_ds[i][1] for i in rand_list]).cuda()
                train_display_depth = torch.stack([train_ds[i][2] for i in rand_list]).cuda()
                train_display_mlabel = torch.stack([torch.tensor(train_ds[i][3]) for i in rand_list]).cuda()
                display_ds = display_loader.dataset
                rand_list = [random.randint(0, len(display_dataset)-1) for i in range(display_size)]
                # test_display_limages = torch.stack([display_ds[i][0] for i in rand_list]).cuda()
                test_display_mimages = torch.stack([display_ds[i][1] for i in rand_list]).cuda()
                test_display_depth = torch.stack([display_ds[i][2] for i in rand_list]).cuda()
                test_display_mlabel = torch.stack([torch.tensor(display_ds[i][3]) for i in rand_list]).cuda()
                test_image_randouts = trainer.sample(test_display_mimages, test_display_depth, test_display_mlabel)
                train_image_randouts = trainer.sample(train_display_mimages, train_display_depth, train_display_mlabel)
                write_2images(train_image_randouts, display_size, image_directory, 'train_%08d' % (total_iter + 1))
                write_2images(test_image_randouts, display_size, image_directory, 'test_%08d' % (total_iter + 1))
        
        total_iter += 1
        train_time = time.perf_counter() - start_time
        rate = (iter + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "-" * int((1 - rate) * 50)
        print("\repoch:{:^5}/{:^5} {:^3.0f}%[{}->{}]{:.2f}s,cur_iter:{}".format(epoch, config['total_epoch'], int(rate * 100), a, b, train_time, total_iter), end="")
    print()
    # decay learning rate,每config['step_size']个epoch之后才会更新lr
    trainer.updata_learning_rate()
    write_lr(epoch, trainer, train_writer)

    # save model  MUNIT保存模型是把所有子模块如生成器、鉴别器分开保存，本文和DRIT则是所有的都保存到一个文件中
    if (epoch + 1) % config['model_save_freq'] == 0:
        print('save the model @ epoch %d' % (epoch))
        trainer.save('%s/%05d.pth' % (checkpoint_directory, epoch), epoch, total_iter)

    ############测试#################
    trainer.eval()
    with torch.no_grad():
        cls_losses = []
        test_cont = 0
        test_cont = 0
        num_real = 0
        err_cls_real = 0
        num_fake = 0
        err_cls_fake = 0
        for iter, (mix_img, label) in enumerate(test_loader):
            mix_img = mix_img.to(device)
            label = label.to(device)
            cls_out = trainer.forward(mix_img)
            test_cls_out = torch.softmax(cls_out, dim=1)
            pre_cls_lb = torch.max(test_cls_out, dim=1)[1]
            test_cont += (pre_cls_lb == label).sum().item()
            for i in range(len(label)):
                if label[i] == 1:
                    num_real += 1
                    if pre_cls_lb[i] != label[i]:
                        err_cls_real += 1
                else:
                    num_fake += 1
                    if pre_cls_lb[i] != label[i]:
                        err_cls_fake += 1
            cls_loss = F.cross_entropy(cls_out, label)
            cls_losses.append(cls_loss.item())
        avg_cls_loss = np.mean(cls_losses)
        train_writer.add_scalar('test/avg_cls_loss' , avg_cls_loss, epoch)
        acc = test_cont/(testA_len+testB_len)
        train_writer.add_scalar('test/acc' , acc, epoch)
        apcer = np.around(100*err_cls_fake/num_fake, 4)
        bpcer = np.around(100*err_cls_real/num_real, 4)
        print('APCER: {}, BPCER: {}, ACER: {}'.format(apcer, bpcer, np.around((apcer+bpcer)/2, 4)))
        early_stopping(avg_cls_loss, trainer, early_stop_model_directory+'/'+str('%05d'%epoch)+'_final_'+str(acc*10000)[:4]+'.pth')
        if early_stopping.early_stop:
            print("Early stopping")
            break
print('finished the training')








