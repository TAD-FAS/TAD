# -*- coding: utf-8 -*-

"""
Created on 2021/10/4 21:40
@author: Acer
@description: 
"""


from pytorchtools import CenterLoss
from depth_trainer import Depth_trainer
from networks import *
from utils import *
import torch
import os


class TAD_trainer(nn.Module):
    def __init__(self, hyperparms):
        super(TAD_trainer, self).__init__()
        # initial related parameters
        lr = hyperparms['lr']
        beta1 = hyperparms['beta1']
        beta2 = hyperparms['beta2']
        self.gpuid = hyperparms['gpuID']
        self.gan_type = hyperparms['dis']['gan_type'] 
        self.norm_1_2 = hyperparms['gen']['norm_1_2']
        self.batch_size = hyperparms['batch_size']
        self.display_size = hyperparms['display_size']
        
        self.gen_m2l = Generator(3, hyperparms['gen'])  # mix->live
        self.gen_m2t = Generator(3, hyperparms['gen'])  # mix->trace
        self.msm = MSM(hyperparms['gen'])
        self.cls = Classifier(2048)
        self.center = CenterLoss(2, 384)

        self.depth_model = Depth_trainer(hyperparms)
        checkpoint = torch.load('depth_model/protocal_1/oulu_npu_pt100148_final.pth')
        self.depth_model.load_state_dict(checkpoint)
        for p in self.depth_model.parameters():
            p.requires_grad = False

        self.dis_mix = MsDiscriminator(in_channels=3, params=hyperparms['dis'], sn=True)
        self.dis_cont = Dis_content(self.gen_m2l.enc.output_dim)
        # 设置优化器
        gen_params = list(self.gen_m2l.parameters()) + list(self.gen_m2t.parameters())
        self.dis_opt = torch.optim.Adam(self.dis_mix.parameters(),
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparms['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparms['weight_decay'])
        self.dis_cont_opt = torch.optim.Adam(self.dis_cont.parameters(), lr = lr/2., betas=(beta1, beta2), weight_decay=hyperparms['weight_decay'])
        self.map_opt = torch.optim.Adam(self.msm.parameters(),
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparms['weight_decay'])
        self.cls_opt = torch.optim.Adam(self.cls.parameters(),
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparms['weight_decay'])
        self.center_opt = torch.optim.Adam(self.center.parameters(),
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparms['weight_decay'])
        # lr scheduler
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparms)
        self.dis_cont_scheduler = get_scheduler(self.dis_cont_opt, hyperparms)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparms)
        self.map_scheduler = get_scheduler(self.map_opt, hyperparms)
        self.cls_scheduler = get_scheduler(self.cls_opt, hyperparms)
        self.center_scheduler = get_scheduler(self.center_opt, hyperparms)

        # initial weights of network
        self.initnetworks(hyperparms['init'])  # 只会设置生成器的的初始化方式
        self.initlossfunc()

    def initnetworks(self, init_type='gaussian'):
        self.gen_m2l.apply(weight_init(init_type))
        self.gen_m2t.apply(weight_init(init_type))

        self.dis_mix.apply(weight_init(init_type))
        self.dis_cont.apply(weight_init(init_type))

        self.msm.apply(weight_init(init_type))
        self.cls.apply(weight_init(init_type))
        self.center.apply(weight_init(init_type))
    
    def initlossfunc(self):
        self.triplet_func = nn.TripletMarginLoss(margin=0.2, p=2)
        self.kl_func = nn.KLDivLoss(reduction='batchmean')
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        

    def recon_criterion(self, input, target):
        # nn.L1Loss() # 默认会对batch取平均值
        return torch.mean(torch.abs(input - target))

    def denorm(self, img):
        min, max = torch.min(img), torch.max(img)
        img = (img - min) / (max - min + 1e-5)
        return img

    def norm(self, img):
        img = (img - 0.5)/0.5
        return img

    def forward(self, mix_image):
        h_noise = self.gen_m2t.enc(mix_image)
        pre_depth_map, pre_path_map, center_feat = self.msm(h_noise)
        cls_out = self.cls(pre_depth_map, pre_path_map, center_feat)
        return cls_out

    def get_feat(self, mix_image):
        h_noise = self.gen_m2t.enc(mix_image)
        h_live = self.gen_m2l.enc(mix_image)
        pre_depth_map, pre_path_map, center_feat = self.msm(h_noise)
        # cls_out = self.cls(pre_depth_map, pre_path_map, center_feat)
        return h_live, h_noise, center_feat
        

    def content_updata(self, mix_image, label, hyperparams):
        self.dis_cont_opt.zero_grad()
        # h_llive = self.gen_m2l.enc(live_image)
        h_mlive = self.gen_m2l.enc(mix_image)
        h_live,_ = self.index_feature(h_mlive, label, 1)  # 取出live face的特征(当bs个样本中没有live face时返回None)
        h_spoof,_ = self.index_feature(h_mlive, label, 0)  # 取出spoof face的特征
        h_live_out, h_spoof_out = None, None
        if h_live is not None:
            h_live_out = self.dis_cont(h_live)
        if h_spoof is not None:
            h_spoof_out = self.dis_cont(h_spoof)
        self.dis_content_loss = hyperparams['gan_w']*self.calcu_dis_cont_loss(h_live_out, h_spoof_out)
        self.dis_content_loss.backward()
        # nn.utils.clip_grad_norm_(self.dis_cont.parameters(), 5)
        self.dis_cont_opt.step()

    def dis_updata(self, live_image, mix_image, label, hyperparams):
        self.dis_opt.zero_grad()
        self.dis_cont_opt.zero_grad()
        # h_llive = self.gen_m2l.enc(live_image)
        gen_mlive, h_mlive = self.gen_m2l(mix_image)
        h_live,_ = self.index_feature(h_mlive, label, 1)  # 取出live face的特征(当bs个样本中没有live face时返回None)
        h_spoof,_ = self.index_feature(h_mlive, label, 0)  # 取出spoof face的特征

        ############# calculate loss for discriminator############
        h_live_out, h_spoof_out = None, None
        if h_live is not None:
            h_live_out = self.dis_cont(h_live)
        if h_spoof is not None:
            h_spoof_out = self.dis_cont(h_spoof)
        self.dis_content_loss = self.calcu_dis_cont_loss(h_live_out, h_spoof_out)
        fale_m2live_outs = self.dis_mix(gen_mlive.detach())
        real_live_out = self.dis_mix(live_image.detach())
        self.dis_m2l_adv_loss = self.calcu_dis_loss(fale_m2live_outs, real_live_out)  # 
        self.dis_total_loss = hyperparams['gan_w'] * (self.dis_content_loss + self.dis_m2l_adv_loss)
        self.dis_total_loss.backward()
        # nn.utils.clip_grad_norm_(self.dis_cont.parameters(), 5) # dis_content update
        self.dis_opt.step()
        self.dis_cont_opt.step()

    def gen_updata(self, mix_image, depth_map, label, hyperparams):
        '''
        计算生成器的总损失，并更新生成器的参数
        :param live_images:
        :param mix_images:
        :param depth_gt: real face对应的depth map ground truth
        :param hyperparams:
        :return:
        '''
        self.gen_opt.zero_grad()
        self.dis_cont_opt.zero_grad()
        self.map_opt.zero_grad()
        self.cls_opt.zero_grad()
        self.center_opt.zero_grad()
        gen_mlive, h_mlive = self.gen_m2l(mix_image)
        h_noise = self.gen_m2t.enc(mix_image)    
        gen_mmix = self.gen_m2t.dec(h_mlive.detach()+h_noise)
        pre_depth_map, pre_path_map, center_feat = self.msm(h_noise)
        cls_out = self.cls(pre_depth_map, pre_path_map, center_feat)
        # latent reconstruction
        h_m2l_recon = self.gen_m2l.enc(gen_mlive)
        h_m2ml_recon = self.gen_m2l.enc(gen_mmix)
        h_m2mt_recon = self.gen_m2t.enc(gen_mmix)
        
        '''------------------calculate loss for generator--------------------'''
        ############# calculate adversarial loss for generator############
        h_live,_ = self.index_feature(h_mlive, label, 1)  # 取出live face的特征(当bs个样本中没有live face时返回None)
        h_spoof,_ = self.index_feature(h_mlive, label, 0)  # 取出spoof face的特征
        h_live_out, h_spoof_out = None, None
        if h_live is not None:
            h_live_out = self.dis_cont(h_live)
        if h_spoof is not None:
            h_spoof_out = self.dis_cont(h_spoof)
        self.gen_content_loss = self.calcu_gen_cont_loss(h_live_out, h_spoof_out)
        # domain adv loss
        m2live_outs = self.dis_mix(gen_mlive)
        self.gen_m2l_adv_loss = self.calcu_gen_loss(m2live_outs)   
        # self.gen_m2m_adv_loss = self.calcu_gen_loss(m2mix_outs)
        ########################### pix-wise loss#########################
        self.recon_mix_loss = self.calcu_pix_wise_loss(gen_mmix, mix_image)
        self.mix2live_loss = self.calcu_m2live_loss(gen_mlive, mix_image, label)
        ######################### latent recon loss#######################
        # self.h_m2l_recon_loss = self.calcu_pix_wise_loss(h_m2l_recon, h_mlive)
        self.h_m2ml_recon_loss = self.calcu_pix_wise_loss(h_m2ml_recon, h_mlive)
        self.h_m2mt_recon_loss = self.calcu_pix_wise_loss(h_m2mt_recon, h_noise)
        #########################  KL loss  ##############################
        # self.h_trace_kl_loss = self.__compute_kl(h_noise)

        center_live, lb_live = self.index_feature(center_feat, label, 1)  # 取出live face的noise特征(当bs个样本中没有live face时返回None)
        if center_live is not None:
            self.center_loss = self.center(lb_live, center_live)
        else:
            self.center_loss = 0
        self.m2l_depth_loss = self.calcu_m2l_depth_loss(gen_mlive, depth_map)
        self.depth_loss = self.calcu_depth_loss(pre_depth_map, depth_map, label)
        self.patch_loss = self.calcu_patch_loss(pre_path_map, label)
        self.cls_loss = self.calcu_cls_loss(cls_out, label)
        # total loss
        self.gen_total_loss = hyperparams['gan_w'] * self.gen_content_loss + \
                              hyperparams['gan_w'] * self.gen_m2l_adv_loss + \
                              hyperparams['recon_img_w'] * self.recon_mix_loss + \
                              hyperparams['recon_live_w'] * self.mix2live_loss + \
                              hyperparams['recon_code_w'] * self.h_m2ml_recon_loss + \
                              hyperparams['recon_code_w'] * self.h_m2mt_recon_loss + \
                              hyperparams['center_w'] * self.center_loss + \
                              hyperparams['m2l_depth_map_w'] * self.m2l_depth_loss + \
                              hyperparams['depth_map_w'] * self.depth_loss + \
                              hyperparams['patch_map_w'] * self.patch_loss + \
                              hyperparams['cls_w'] * self.cls_loss 
                              # hyperparams['recon_kl_cyc_w'] * self.h_trace_kl_loss
                            #   hyperparams['recon_code_w'] * self.h_m2l_recon_loss + \

        assert torch.isnan(self.gen_total_loss).sum() == 0, print(self.gen_total_loss)
        self.gen_total_loss.backward()
        self.gen_opt.step()
        self.dis_cont_opt.step()
        self.map_opt.step()
        self.cls_opt.step()
        self.center_opt.step()


    def calcu_gen_loss(self, out_fakes):
        '''domain translation adv loss of gen'''
        loss = 0
        # 循环多尺度鉴别器的三个返回结果，并分别计算损失，然后相加
        for out_fake in out_fakes:  
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_fake - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out_fake.data).cuda(self.gpuid)
                loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out_fake), all1))
        return loss

    def calcu_dis_loss(self, fake_outs, real_outs):
        '''adv loss of dis'''
        loss = 0
        # 循环多尺度鉴别器的三个返回结果，并分别计算损失，然后相加
        for iter, (out_fake, out_real) in enumerate(zip(fake_outs, real_outs)):  
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_fake - 0)**2) + torch.mean((out_real - 1)**2)
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out_real.data).cuda(self.gpuid)
                all0 = torch.zeros_like(out_fake.data).cuda(self.gpuid)
                loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out_fake), all0) +
                                   F.binary_cross_entropy(torch.sigmoid(out_real), all1))
        return loss

    def calcu_gen_cont_loss(self, h_live, h_spoof):
        loss_ContentD = 0
        if h_live is None:
            if self.gan_type == 'lsgan':
                loss_ContentD += torch.mean((h_spoof - 0.5)**2)
            elif self.gan_type == 'nsgan':
                all_half = 0.5 * torch.ones_like(h_spoof.data).cuda(self.gpuid)
                loss_ContentD += torch.mean(F.binary_cross_entropy(F.sigmoid(h_spoof), all_half))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        elif h_spoof is None:
            if self.gan_type == 'lsgan':
                loss_ContentD += torch.mean((h_live - 0.5)**2)
            elif self.gan_type == 'nsgan':
                all_half = 0.5 * torch.ones_like(h_live.data).cuda(self.gpuid)
                loss_ContentD += torch.mean(F.binary_cross_entropy(F.sigmoid(h_live), all_half))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        else:
            if self.gan_type == 'lsgan':
                loss_ContentD += torch.mean((h_live - 0.5)**2) + torch.mean((h_spoof - 0.5)**2)
            elif self.gan_type == 'nsgan':
                all_half = 0.5 * torch.ones_like(h_spoof.data).cuda(self.gpuid)
                loss_ContentD += torch.mean(F.binary_cross_entropy(F.sigmoid(h_live), all_half) +
                                    F.binary_cross_entropy(F.sigmoid(h_spoof), all_half))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss_ContentD
    
    def calcu_dis_cont_loss(self, h_live, h_spoof):
        loss_D = 0
        if h_live is None:
            if self.gan_type == 'lsgan':
                loss_D += torch.mean((h_spoof - 1)**2)
            elif self.gan_type == 'nsgan':
                # all0 = torch.zeros_like(h_live.data).cuda(self.gpuid)
                all1 = torch.ones_like(h_spoof.data).cuda(self.gpuid)
                loss_D += torch.mean(F.binary_cross_entropy(F.sigmoid(h_spoof), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        elif h_spoof is None:
            if self.gan_type == 'lsgan':
                loss_D += torch.mean((h_live - 0)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(h_live.data).cuda(self.gpuid)
                # all1 = torch.ones_like(h_spoof.data).cuda(self.gpuid)
                loss_D += torch.mean(F.binary_cross_entropy(F.sigmoid(h_live), all0))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        else:
            if self.gan_type == 'lsgan':
                loss_D += torch.mean((h_live - 0)**2) + torch.mean((h_spoof - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(h_live.data).cuda(self.gpuid)
                all1 = torch.ones_like(h_spoof.data).cuda(self.gpuid)
                loss_D += torch.mean(F.binary_cross_entropy(F.sigmoid(h_live), all0) +
                                    F.binary_cross_entropy(F.sigmoid(h_spoof), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss_D
   

    def calcu_depth_loss(self, pre_depth, target_depth, label):
        target_depth = F.interpolate(target_depth, (pre_depth.size(2), pre_depth.size(3)), mode='bilinear', align_corners=True)
        for i in range(len(label)):
            if label[i] == 0:  # 如果是欺骗人脸
                target_depth[i] = -torch.ones_like(pre_depth[i]).cuda(self.gpuid)  # 全0图
        criterion_MSE = nn.MSELoss()
        absolute_loss = criterion_MSE(pre_depth, target_depth)
        contrast_out = self.contrast_depth_conv(pre_depth)
        contrast_label = self.contrast_depth_conv(target_depth)    
        contrast_loss = criterion_MSE(contrast_out, contrast_label)
        return absolute_loss + contrast_loss

    def calcu_patch_loss(self, input, label):
        criterion_MSE = nn.MSELoss()
        label_maps = []
        for lb in label:
            if lb == 0:
                label_maps.append(-torch.ones_like(input[0]).cuda(self.gpuid))
            else:
                label_maps.append(torch.ones_like(input[0]).cuda(self.gpuid))
        label_maps = torch.stack([lb_map for lb_map in label_maps])
        loss = criterion_MSE(input, label_maps)
        return loss

    def calcu_m2live_loss(self, m2limage, miximage, label, norm=1):
        loss = 0
        live_img_num = 0
        for i in range(len(label)):
            loss += label[i] * self.calcu_pix_wise_loss(m2limage, miximage)
            if label[i] != 0:
                live_img_num += 1                
        if live_img_num != 0: 
            loss = loss / live_img_num
        return loss

    def calcu_m2l_depth_loss(self, input, depth_map, norm=1):
        pre_depth_map = self.depth_model(input)
        depth_gt = F.interpolate(depth_map, (pre_depth_map.size(2), pre_depth_map.size(3)), mode='bilinear', align_corners=True)
        loss = self.calcu_pix_wise_loss(pre_depth_map, depth_gt, norm)
        return loss

    def calcu_latent_loss(self, input, target, norm=1):
        loss = self.calcu_pix_wise_loss(input, target, norm)
        return loss

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss


    def calcu_pix_wise_loss(self, input, target, norm=1):
        loss = 0
        if norm == 1:
            loss = self.l1(input, target)
        elif norm == 2:
            loss = self.l2(input, target)
        else:
            print('please select correct norm')
        return loss

    def calcu_cls_loss(self, input, target):
        input = torch.softmax(input, dim=1)
        return F.cross_entropy(input, target)

    def contrast_depth_conv(self, input):
        ''' compute contrast depth in both of (out, label) '''
        '''
            input  32x32
            output 8x32x32
        '''
        # print(input.size())  # torch.Size([bs, 1, 32, 32])
        kernel_filter_list =[
                            [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                            [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                            [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                            ]
        
        kernel_filter = np.array(kernel_filter_list, np.float32)   
        kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda(self.gpuid)
        # weights (in_channel, out_channel, kernel, kernel)
        kernel_filter = kernel_filter.unsqueeze(dim=1) 
        # print(kernel_filter.size())  # torch.Size([bs, 1, 3, 3])
        input = input.expand(input.shape[0], 8, input.shape[2],input.shape[3])
        # print(input.size()) 
        contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
        # print(contrast_depth.size())   
        return contrast_depth

    def index_feature(self, feat, label, idx):
        '''在feat中取出label=idx的feature并返回'''
        is_zero = label == torch.tensor(idx)  # 逐个元素比较并返回逐元素比较结果bool类型，并返回给c  ==和函数eq一样，但是和equal不一样
        all0_idx = torch.zeros_like(label, dtype=bool) 
        if is_zero.equal(all0_idx):
            # print('该bs个样本中没有指定标签为idx的样本')
            return None, None
        else:
            index = is_zero.nonzero().view(-1)  # 获取标签=idx的样本在bs中的索引
            idx_feature = feat.index_select(0, index)  # 根据索引选出label=idx对应的特征
            idx_lb = label.index_select(0, index)
            return idx_feature, idx_lb


    def updata_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.dis_cont_scheduler is not None:  
            self.dis_cont_scheduler.step()
        if self.map_scheduler is not None:
            self.map_scheduler.step()
        if self.cls_scheduler is not None:
            self.cls_scheduler.step()
        if self.center_scheduler is not None:
            self.center_scheduler.step()


    def sample(self, mix_image, depth_map, label):
        # self.eval()
        with torch.no_grad():
            gen_mlive, h_mlive = self.gen_m2l(mix_image)
            h_noise = self.gen_m2t.enc(mix_image)
            # gen_mmix = gen_trace + gen_mlive
            gen_mmix = self.gen_m2t.dec(h_mlive+h_noise)

            pre_depth_map, pre_path_map, _ = self.msm(h_noise)
            # cls_out = self.cls(pre_depth_map, pre_path_map, center_feat)

            mix_depth = self.depth_model(mix_image)
            m2l_depth = self.depth_model(gen_mlive)
            mix_depth = F.interpolate(mix_depth, (gen_mlive.size(2), gen_mlive.size(3)), mode='bilinear', align_corners=True)
            m2l_depth = F.interpolate(m2l_depth, (gen_mlive.size(2), gen_mlive.size(3)), mode='bilinear', align_corners=True)
            pre_depth_map = F.interpolate(pre_depth_map, (gen_mlive.size(2), gen_mlive.size(3)), mode='bilinear', align_corners=True)
            pre_path_map = F.interpolate(pre_path_map, (gen_mlive.size(2), gen_mlive.size(3)), mode='bilinear', align_corners=True)

            # get trace label
            mimg_norm = self.denorm(mix_image)  # [-1,1]->[0,1]
            mlimg_norm = self.denorm(gen_mlive)  # [-1,1]->[0,1]
            trace_gt = torch.abs(mimg_norm - mlimg_norm)  # [0,1]
            trace_gt = self.norm(trace_gt)  # [0,1]->[-1,1]
        
            for i in range(len(label)):
                if label[i] == 0:  # 如果是欺骗人脸
                    depth_map[i] = -torch.ones_like(depth_map[i]).cuda(self.gpuid)  # 全0图

            images = [mix_image, gen_mlive, gen_mmix, trace_gt, mix_depth, m2l_depth, pre_depth_map, pre_path_map, depth_map]
        # self.train()
        return images

    def resume(self, checkpoint_dir, hyperparams):
        '''恢复模型以及优化器。如果后续添加了网络记得在这里保存'''
        last_model_name = get_model_list(checkpoint_dir)  # 保存的模型是以epoch命名的，所以恢复的时候选择文件名最大的模型
        checkpoint = torch.load(last_model_name)
        # print(checkpoint.keys())
        # load weight
        self.dis_mix.load_state_dict(checkpoint['dis_mix'])
        self.dis_cont.load_state_dict(checkpoint['dis_cont'])
        self.gen_m2l.load_state_dict(checkpoint['gen_m2l'])
        self.gen_m2t.load_state_dict(checkpoint['gen_m2t'])

        self.msm.load_state_dict(checkpoint['msm'])
        self.cls.load_state_dict(checkpoint['cls'])
        self.center.load_state_dict(checkpoint['center'])

        # load optimizer
        self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        self.dis_opt.load_state_dict(checkpoint['dis_opt'])
        self.dis_cont_opt.load_state_dict(checkpoint['dis_cont_opt'])
        self.map_opt.load_state_dict(checkpoint['map_opt'])
        self.cls_opt.load_state_dict(checkpoint['cls_opt'])
        self.center_opt.load_state_dict(checkpoint['center_opt'])
        last_epoch = checkpoint['epoch']
        total_iter = checkpoint['total_it']

        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparams, last_epoch+1)
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparams, last_epoch+1)
        self.dis_cont_scheduler = get_scheduler(self.dis_cont_opt, hyperparams, last_epoch +1)
        self.map_scheduler = get_scheduler(self.map_opt, hyperparams, last_epoch + 1)
        self.cls_scheduler = get_scheduler(self.cls_opt, hyperparams, last_epoch + 1)
        self.center_scheduler = get_scheduler(self.center_opt, hyperparams, last_epoch + 1)

        return last_epoch, total_iter

    def save(self, filename, epoch, total_it):
        # Save generators, discriminators, and optimizers
        state = {
            'dis_mix': self.dis_mix.state_dict(),
            'dis_cont': self.dis_cont.state_dict(),
            'gen_m2l': self.gen_m2l.state_dict(),
            'gen_m2t': self.gen_m2t.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'msm': self.msm.state_dict(),
            'cls': self.cls.state_dict(),
            'center': self.center.state_dict(),
            'dis_opt': self.dis_opt.state_dict(),
            'dis_cont_opt': self.dis_cont_opt.state_dict(),
            'map_opt': self.map_opt.state_dict(),
            'cls_opt': self.cls_opt.state_dict(),
            'center_opt': self.center_opt.state_dict(),
            'epoch': epoch,
            'total_it': total_it
        }
        torch.save(state, filename)


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from utils import *
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # writer = SummaryWriter('logs')
    model = TAD_trainer(get_config('./configs/oulu_npu.yaml')).to(device)
    # x = torch.rand(4, 3, 256, 256).to(device)
    # writer.add_graph(model, input_to_model=x)
    # writer.close()


    


















