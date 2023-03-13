# -*- coding: utf-8 -*-

"""
Created on 2021/10/7 15:33
@author: Acer
@description: 
"""


import torch
import numpy as np
from utils import *
import torch
import argparse
from datasets import *
import time
from trainer import TAD_trainer
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='oulu_npu', help='test datasets')
parser.add_argument('--output_path', type=str, default='outputs', help="outputs path")  # ../autodl-tmp
opts = parser.parse_args()
# Load experiment setting
config = get_config('./configs/' + opts.dataset + '.yaml')
display_size = config['display_size']
# Setup logger and output folders
# 哪个数据集训练模型就以哪个数据集的yaml文件名为模型名称，模型名称用于保存数据的时候便于对不同数据集的输出结果保存到不同目录
model_name = os.path.splitext(os.path.basename('./configs/' + opts.dataset + '.yaml'))[0]
print('*************************************************')
print('current testing dataset:\t%s' % model_name)  # 如果和输入的不一致，那说明yaml文件名和输入的不一致
# test_writer = SummaryWriter(os.path.join("test/logs", model_name))  # ../tf-logs
output_directory = os.path.join(opts.output_path, model_name + "_TAD/" + config['protocol'])
score_directory = prepare_score_folder(output_directory)
print('test protocol\t\t\t', config['protocol'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: \t\t\t%s' % device)

trainer = TAD_trainer(config).to(device)
trainer.eval()
# 提前终止 加载模型方式
checkpoint_directory = os.path.join(output_directory, 'early_stop_model/00059_final_1.56.pth')
checkpoint = torch.load(checkpoint_directory)
trainer.load_state_dict(checkpoint)


# checkpoint 加载模型方式
# checkpoint_directory = os.path.join(output_directory, 'checkpoints')
# # print(checkpoint_directory)
# last_model_name = get_model_list(checkpoint_directory)  # 保存的模型是以epoch命名的,测试的时候选择文件名最大的模型
# print('load model to device\t\t' + last_model_name.split('\\')[-1])
# checkpoint = torch.load(last_model_name)
# trainer.gen_m2t.load_state_dict(checkpoint['gen_m2t'])
# trainer.msm.load_state_dict(checkpoint['msm'])
# trainer.cls.load_state_dict(checkpoint['cls'])
# print('batch size\t\t\t', config['test_bs'])

testA_len = len(os.listdir(os.path.join(config['data_root'], 'testA')))
testB_len = len(os.listdir(os.path.join(config['data_root'], 'testB')))
test_len = testA_len + testB_len
limg_sample_w = test_len / testA_len
simg_sample_w = test_len / testB_len
sample_l_w = [limg_sample_w] * testA_len
sample_s_w = [simg_sample_w] * testB_len
test_sample_w = sample_l_w + sample_s_w
sample_num = min(testA_len, testB_len)
print('sample num of test dataset\t{}'.format(sample_num))
test_sampler = WeightedRandomSampler(test_sample_w, sample_num, replacement=False)
test_dataset = TestDataset(config, phase='test')  # 测试准确率的
test_loader = DataLoader(test_dataset, batch_size=config['test_bs'], shuffle=True, 
                num_workers=config['num_workers'], pin_memory=True)  # , sampler=test_sampler

# 展示潜在表征的数据集
display_spoof_dataset = DisplayDataset(config, 'test')
test_spoof_loader = DataLoader(display_spoof_dataset, batch_size=config['test_bs'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
display_train_dataset = DisplayDataset(config, 'train')
train_spoof_loader = DataLoader(display_train_dataset, batch_size=config['test_bs'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
 
print('*************************************************')


def visul_feat():  # siw和oulu都可用
    h_lives, h_noises, center_feats, labels = None, None, None, None
    with torch.no_grad():
        for iter, (mix_img, lb) in enumerate(test_loader):
            mix_img = mix_img.to(device)
            h_live, h_noise, center_feat = trainer.get_feat(mix_img)
            if h_lives is None:
                h_lives, h_noises, center_feats, labels = h_live, h_noise, center_feat, lb
            else:
                h_lives = torch.cat((h_lives, h_live))
                h_noises = torch.cat((h_noises, h_noise))
                center_feats = torch.cat((center_feats, center_feat))
                labels = torch.cat((labels, lb))
            if labels.size(0) > 800:
                break
    h_lives = h_lives.view(h_lives.size(0), -1).cpu().numpy()
    h_noises = h_noises.view(h_noises.size(0), -1).cpu().numpy()
    center_feats = center_feats.view(center_feats.size(0), -1).cpu().numpy()
    labels = labels.view(labels.size(0), -1).cpu().numpy()

    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    center_feats_2D = tsne_2D.fit_transform(center_feats)
    h_noises_2D = tsne_2D.fit_transform(h_noises)
    h_lives_2D = tsne_2D.fit_transform(h_lives)

    # tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    # center_feats_3D = tsne_3D.fit_transform(center_feats)
    # h_noises_3D = tsne_3D.fit_transform(h_noises)
    # h_lives_3D = tsne_3D.fit_transform(h_lives)

    plot_embedding_2D(center_feats_2D, labels, 'center feat', output_directory)
    plot_embedding_2D(h_noises_2D, labels, 'noise feat', output_directory)
    plot_embedding_2D(h_lives_2D, labels, 'live feat', output_directory)
    # plot_embedding_3D(center_feats_3D, labels, 'center feat', output_directory)
    # plot_embedding_3D(h_noises_3D, labels, 'noise feat', output_directory)
    # plot_embedding_3D(h_lives_3D, labels, 'live feat', output_directory)

# 仅仅oulu可用
def visual_feat_PAI_onTrainTest():  # 训练集和测试集各自取600
    h_lives, h_noises, center_feats, labels = None, None, None, None
    num = 0
    if config['protocol'] == 'protocol_1':
        num=5  # 为了区别训练集和测试集，对训练集合中的标签都+num
    elif config['protocol'] == 'protocol_2':
        num=0  # 由于协议2中训练和测试使用的PAI（即labels）本来就不同，没必须要+num(除真实人脸外)
    else:
        print('本实验不支持其他协议')
    with torch.no_grad():
        for iter, (mix_img, lb) in enumerate(test_spoof_loader):
            mix_img = mix_img.to(device)
            h_live, h_noise, center_feat = trainer.get_feat(mix_img)
            if h_lives is None:
                h_lives, h_noises, center_feats, labels = h_live, h_noise, center_feat, lb
            else:
                h_lives = torch.cat((h_lives, h_live))
                h_noises = torch.cat((h_noises, h_noise))
                center_feats = torch.cat((center_feats, center_feat))
                labels = torch.cat((labels, lb))
            if labels.size(0) >= 500:
                break
    print('测试集取出样本：',h_lives.size(0))
    with torch.no_grad():
        for iter, (mix_img, lb) in enumerate(train_spoof_loader):
            # print("替换前：", lb)
            if config['protocol'] == 'protocol_2':  
                lb = torch.where(lb==0, 5, lb) # 为了区别协议2中训练集和测试集中的真实人脸，我们把训练集中的真实人脸标签设置为5
            mix_img = mix_img.to(device)
            # print("替换后：",lb)
            h_live, h_noise, center_feat = trainer.get_feat(mix_img)
            if h_lives is None:
                h_lives, h_noises, center_feats, labels = h_live, h_noise, center_feat, lb
                print('有问题，h_lives不应该为空')
            else:
                h_lives = torch.cat((h_lives, h_live))
                h_noises = torch.cat((h_noises, h_noise))
                center_feats = torch.cat((center_feats, center_feat))
                labels = torch.cat((labels, lb+num))  # +5是为了和测试集中的样本区分开来，0-4是测试集标签，5-9是训练集标签
            if labels.size(0) >= 1000:
                break
    

    # print(labels)
    print('测试集+训练集共取出样本：',h_lives.size(0))
    h_lives = h_lives.view(h_lives.size(0), -1).cpu().numpy()
    h_noises = h_noises.view(h_noises.size(0), -1).cpu().numpy()
    center_feats = center_feats.view(center_feats.size(0), -1).cpu().numpy()
    labels = labels.view(labels.size(0), -1).cpu().numpy()

    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    center_feats_2D = tsne_2D.fit_transform(center_feats)
    h_noises_2D = tsne_2D.fit_transform(h_noises)
    h_lives_2D = tsne_2D.fit_transform(h_lives)

    plot_PAI_2D_onTrainTest(center_feats_2D, labels, 'center feat_train_test', output_directory, config['protocol'])
    plot_PAI_2D_onTrainTest(h_noises_2D, labels, 'noise feat_train_test', output_directory, config['protocol'])
    plot_PAI_2D_onTrainTest(h_lives_2D, labels, 'live feat_train_test', output_directory, config['protocol'])

# 仅仅oulu可用
def visual_feat_PAI():
    h_lives, h_noises, center_feats, labels = None, None, None, None
    with torch.no_grad():
        for iter, (mix_img, lb) in enumerate(test_spoof_loader):
            mix_img = mix_img.to(device)
            h_live, h_noise, center_feat = trainer.get_feat(mix_img)
            if h_lives is None:
                h_lives, h_noises, center_feats, labels = h_live, h_noise, center_feat, lb
            else:
                h_lives = torch.cat((h_lives, h_live))
                h_noises = torch.cat((h_noises, h_noise))
                center_feats = torch.cat((center_feats, center_feat))
                labels = torch.cat((labels, lb))
            if labels.size(0) >= 800:
                break
    # print(labels)
    h_lives = h_lives.view(h_lives.size(0), -1).cpu().numpy()
    h_noises = h_noises.view(h_noises.size(0), -1).cpu().numpy()
    center_feats = center_feats.view(center_feats.size(0), -1).cpu().numpy()
    labels = labels.view(labels.size(0), -1).cpu().numpy()

    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    center_feats_2D = tsne_2D.fit_transform(center_feats)
    h_noises_2D = tsne_2D.fit_transform(h_noises)
    h_lives_2D = tsne_2D.fit_transform(h_lives)

    # tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    # center_feats_3D = tsne_3D.fit_transform(center_feats)
    # h_noises_3D = tsne_3D.fit_transform(h_noises)
    # h_lives_3D = tsne_3D.fit_transform(h_lives)

    plot_PAI_2D(center_feats_2D, labels, 'center feat', output_directory, config['protocol'])
    plot_PAI_2D(h_noises_2D, labels, 'noise feat', output_directory, config['protocol'])
    plot_PAI_2D(h_lives_2D, labels, 'live feat', output_directory, config['protocol'])
    # plot_PAI_3D(center_feats_3D, labels, 'center feat', output_directory)
    # plot_PAI_3D(h_noises_3D, labels, 'noise feat', output_directory)
    # plot_PAI_3D(h_lives_3D, labels, 'live feat', output_directory)

def calcu_result_NO_THRESH(test_num=5):
    with torch.no_grad():
        apcers = []
        bpcers = []
        acers = []
        for idx in range(test_num):
            test_cont = 0
            num_real = 0
            err_cls_real = 0
            num_fake = 0
            err_cls_fake = 0
            start_time = time.perf_counter()
            for iter, (mix_img, lb) in enumerate(test_loader):
                mix_img = mix_img.to(device)
                lb = lb.to(device)
                cls_out = trainer.forward(mix_img)
                # 计算评估标准
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
                test_time = time.perf_counter() - start_time
                rate = (iter + 1) / len(test_loader)
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\r{:^3.0f}%[{}->{}]{:.2f}s,cur_iter:{}".format(int(rate * 100), a, b, test_time, iter), end="")
            print()
            apcer = np.around(100*err_cls_fake/num_fake, 4)
            bpcer = np.around(100*err_cls_real/num_real, 4)
            apcers.append(apcer)
            bpcers.append(bpcer)
            acers.append(np.around((apcer+bpcer)/2, 4))
            # acer1 = 100*test_cont/(sample_num)
            # print(acer1)
        
        print('APCER: {}'.format(apcers))
        print('BPCER: {}'.format(bpcers))
        print('ACER: {}'.format(acers))
    print('finished the testing ......')

        

if __name__ == '__main__':
    # calcu_result_NO_THRESH(1)
    # visul_feat()  #local variable 'spoof' referenced before assignment,需要把shuffer改为True
    visual_feat_PAI() 

    










