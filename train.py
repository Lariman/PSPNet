import random
import math
import time
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from DataLoader import VOCDataset, DataTransform
from utils.datapath import make_datapath_list
from PSPNet import PSPNet

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

# 创建文件路径列表
rootpath = 'E:/数据集/VOCdevkit/VOC2012/'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)

# 创建Dataset
# (RGB)颜色的平均值和均方差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",
                           transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", 
                         transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

# 生成DataLoader
batch_size = 2  # batch_size大小

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 集中保存到字典型变量中
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 通过微调创建PSPNet
# 使用ADE20K数据集中事先训练好的模型，ADE20K的分类数量是150
net = PSPNet(n_classes=150)

# 载入ADE20K中事先训练好的参数
state_dict = torch.load("./weights/pspnet50_ADE20K.pth")
net.load_state_dict(state_dict)

# 将分类用的卷积层替换为输出数量为21的卷积层
n_classes = 21
net.decode_feature.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
net.aux.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

# 对替换的卷积层进行初始化。由于激活函数是Sigmoid，因此使用Xavier进行初始化
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

net.decode_feature.classification.apply(weights_init)
net.aux.classification.apply(weights_init)

print("网络设置完毕：成功载入预训练权重")

# 设置损失函数
class PSPLoss(nn.Module):
    """PSPNet的损失函数类"""
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight
    
    def forward(self, outputs, targets):
        """
            损失函数的计算
            输入参数：
                outputs(PSPNet的输出,tuple类型):(output=torch.Size([num_batch, 21, 475, 475]), output_aux=torch.Size([num_batch, 21, 475, 475]))
                targets(正解的标注信息):[num_batch, 475, 475]
            输出参数：
                loss(张量):损失值
        """
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')
        return loss + self.aux_weight * loss_aux
criterion = PSPLoss(aux_weight=0.4)
print(criterion)

# 利用调度器调整每轮epoch的学习率
# 由于使用的是微调，故将靠近输入的模块的学习率调低，将包含被替换的卷积层的Decoder模块和AuxLoss模块的学习率调高。
optimizer = optim.SGD([
    {'params': net.feature_conv.parameters(), 'lr': 1e-3},
    {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
    {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
    {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
    {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
    {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
    {'params': net.decode_feature.parameters(), 'lr': 1e-2},
    {'params': net.aux.parameters(), 'lr': 1e-2},
], momentum=0.9, weight_decay=0.0001)  # momentum:动量因子（起到正向加速，反向减速的作用）; weight_decay:权重衰减(L2惩罚)，防止过拟合

# 设置调度器:对每轮epoch的学习率进行动态调整，随着epoch的运行逐渐减小学习率。
# 若要通过调度器来动态改变学习率，需要在进行网络学习时调用scheduler.step()方法。
def lambda_epoch(epoch):
    max_epoch = 30
    return math.pow((1-epoch/max_epoch), 0.9)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)  # 从lambda_epoch函数返回的值与optimizer的学习率进行乘法运算


# 创建对模型进行训练的函数
def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):
    # 确认GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用的设备：", device)

    # 将网络载入GPU中
    net.to(device)

    # 如果网络相对固定，开启高速处理选项
    torch.backends.cudnn.benchmark = True

    # 图像的张数
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    # 设置迭代计数器
    iteration = 1
    logs = []

    # multiple minibatch
    batch_multiplier = 3

    # epoch的循环
    for epoch in range(num_epochs):
        # 保存开始时间
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0  # epoch的损失和
        epoch_val_loss = 0.0  # epoch的损失和

        print('----------------')
        print("Epoch{}/{}".format(epoch+1, num_epochs))
        print('----------------')

        # 对每轮epoch进行训练和验证的循环
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # 将模式设置为训练模式
                scheduler.step()  # 更新最优化调度器
                optimizer.zero_grad()
                print('(train)')
            else:
                if((epoch+1) % 5 == 0):  # 每5轮验证一次
                    net.eval()  # 将模型设置为验证模式
                    print('-----------')
                    print('(val)')
                else:
                    continue
            
            # 从数据加载器中读取每个小批量并进行循环
            count = 0  # multiple minibatch
            for images, anno_class_images in dataloaders_dict[phase]:
                # 如果小批量的尺寸是1，批归一化处理会报错，因此需要避免
                if images.size()[0] == 1:
                    continue

                # 如果GPU可用，将数据传输到GPU中
                images = images.to(device)
                anno_class_images = anno_class_images.to(device)

                # 使用multiple minibatch对参数进行更新
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier
                
                # 正向传播计算
                with torch.set_grad_enabled(phase == 'train'):  # 根据判断条件，控制是否允许进行梯度更新
                    outputs = net(images)
                    loss = criterion(outputs, anno_class_images.long()) / batch_multiplier

                    # 训练时采用反向传播
                    if phase == 'train':
                        loss.backward()  # 梯度的计算
                        count -= 1  # multiple minibatch

                        if(iteration % 10 == 0):  # 每10次迭代显示一次loss
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('迭代 {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(iteration, loss.item()/batch_size*batch_multiplier, duration))
                            t_iter_start = time.time()
                        
                        epoch_train_loss = loss.item() * batch_multiplier
                        iteration += 1
                    
                    # 验证时
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        # 每个epoch的phase的loss和正解率       
        t_epoch_finish = time.time()
        print('----------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
        epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # 保存日志
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss / num_train_imgs, 
                     'val_loss': epoch_val_loss / num_val_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("./log/log_output.csv")

    # 保存最终的网络
    torch.save(net.state_dict(), 'weights/pspnet50_final.pth')


if __name__ == "__main__":
    # 执行学习和验证操作
    num_epochs = 1
    train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs)
