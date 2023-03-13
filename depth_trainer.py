from depthnet import *
from utils import *
import torch




class Depth_trainer(nn.Module):
    def __init__(self, hyperparms):
        super(Depth_trainer, self).__init__()
        # initial related parameters
        lr = hyperparms['lr']
        beta1 = hyperparms['beta1']
        beta2 = hyperparms['beta2']
        self.gpuid = hyperparms['gpuID']
        self.gan_type = hyperparms['dis']['gan_type'] 
        self.norm_1_2 = hyperparms['depth']['norm_1_2']
        self.batch_size = hyperparms['batch_size']
        self.display_size = hyperparms['display_size']
        
        self.depthnet = DepthNet(3, hyperparms['depth'])
        self.depth_opt = torch.optim.Adam(self.depthnet.parameters(),
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparms['weight_decay'])
        # lr scheduler
        self.depth_scheduler = get_scheduler(self.depth_opt, hyperparms)
        # initial weights of networks
        self.initnetworks(hyperparms['init'])  
        

    def initnetworks(self, init_type='gaussian'):
        self.depthnet.apply(weight_init(init_type))

    def calcu_depth_loss(self, input, target):
        criterion_MSE = nn.MSELoss()
        absolute_loss = criterion_MSE(input, target)

        contrast_out = self.contrast_depth_conv(input)
        contrast_label = self.contrast_depth_conv(target)    
        contrast_loss = criterion_MSE(contrast_out, contrast_label)
        return absolute_loss + contrast_loss


    def forward(self, image):
        pre_depth_map = self.depthnet(image)
        return pre_depth_map
    
    def depth_valid(self, mix_image, depth_map):
        pre_depth_map = self.depthnet(mix_image)
        depth_map = F.interpolate(depth_map, (pre_depth_map.size(2), pre_depth_map.size(3)), mode='bilinear', align_corners=True)
        depth_loss = self.calcu_depth_loss(pre_depth_map, depth_map)
        return depth_loss, pre_depth_map

    def depth_updata(self, mix_image, depth_map):
        self.depth_opt.zero_grad()
        pre_depth_map = self.depthnet(mix_image)
        depth_map = F.interpolate(depth_map, (pre_depth_map.size(2), pre_depth_map.size(3)), mode='bilinear', align_corners=True)
        '''------------------calculate loss for depthnet--------------------'''
        self.depth_loss = self.calcu_depth_loss(pre_depth_map, depth_map)
        self.depth_loss.backward()

        self.depth_opt.step()
        return pre_depth_map
    
    def contrast_depth_conv(self, input):
        '''https://github.com/ZitongYu/CDCN'''
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
        input = input.expand(input.shape[0], 8, input.shape[2],input.shape[3])
        contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
        return contrast_depth

    def updata_learning_rate(self):
        if self.depth_scheduler is not None:
            self.depth_scheduler.step()


    def resume(self, depth_model_dir, hyperparams):
        '''恢复模型以及优化器。如果后续添加了网络记得在这里保存'''
        last_model_name = get_model_list(depth_model_dir)  # 保存的模型是以epoch命名的，所以恢复的时候选择文件名最大的模型
        checkpoint = torch.load(last_model_name)
        # load weight
        self.depthnet.load_state_dict(checkpoint['depthnet'])
        # load optimizer
        self.depth_opt.load_state_dict(checkpoint['depth_opt'])
        last_epoch = checkpoint['epoch']
        total_iter = checkpoint['total_it']

        self.depth_scheduler = get_scheduler(self.depth_opt, hyperparams, last_epoch)
        return last_epoch, total_iter

    def save(self, filename, epoch, total_it, _use_new_zipfile_serialization=False):
        # Save generators, discriminators, and optimizers
        state = {
            'depthnet': self.depthnet.state_dict(),
            'depth_opt': self.depth_opt.state_dict(),
            'epoch': epoch,
            'total_it': total_it
        }
        torch.save(state, filename,  _use_new_zipfile_serialization=_use_new_zipfile_serialization)



if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    from utils import *
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # writer = SummaryWriter('logs')
    model = Depth_trainer(get_config('./configs/oulu_npu.yaml')).to(device)
    x = torch.rand(4, 3, 256, 256).to(device)
    # writer.add_graph(model, input_to_model=x)
    # writer.close()

