
# 首先导入数据处理类和数据增强类
from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
from utils.datapath import make_datapath_list
import os.path as osp
from PIL import Image

import torch.utils.data as data

class DataTransform():
    """
        图像和标注的预处理类。训练和验证时分别采取不同的处理方法
        将图像的尺寸调整为input_size * input_size
        训练时进行数据增强处理
        属性：
            input_size(int):指定调整图像尺寸的大小
            color_mean(R,G,B):指定每个颜色通道的平均值
            color_std(R,G,B):指定每个颜色通道的标准差
    """
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 图像的缩放
                RandomRotation(angle=[-10, 10]),  # 随机旋转10°
                RandomMirror(),  # 随机镜像
                Resize(input_size),  # 调整图像尺寸
                Normalize_Tensor(color_mean, color_std),  # 颜色信息的归一化和张量化
            ]),
            'val': Compose([
                Resize(input_size),  # 调整图像尺寸
                Normalize_Tensor(color_mean, color_std)  # 颜色信息的归一化和张量化
            ])
        }
    
    def __call__(self, phase, img, anno_class_img):
        # 指定与处理的执行模式
        return self.data_transform[phase](img, anno_class_img)

class VOCDataset(data.Dataset):
    """
        用于创建VOC2012的Dataset的类，继承自Pytorch的Dataset类
        属性：
            img_list(列表):保存了图像列表
            anno_list(列表):保存了标注路径列表
            phase('train' or 'test'):设置是学习模式还是训练模式
            transform:预处理类的实例
    """
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
    
    def __len__(self):
        """返回图像张数"""
        return len(self.img_list)
    
    def __getitem__(self, index):
        """获取经过预处理的图像的张量形式的数据和标注"""
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img
    
    def pull_item(self, index):
        """获取图像的张量形式的数据和标注"""
        # 1. 读入图像数据
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)
        # 2. 读入标注图像数据
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)
        # 3. 进行预处理操作
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)
        return img, anno_class_img


if __name__ == '__main__':
    def main():
        # ===========以下代码段为测试单张图片作预处理(不包含归一化和转tensor)后的模样========
        from PIL import Image

        # (RGB)颜色的平均值和均方差
        color_mean = (0.485, 0.456, 0.406)
        color_std = (0.229, 0.224, 0.225)

        def image_open():
            image_path = 'E:/数据集/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg'
            anno_image_path = 'E:/数据集/VOCdevkit/VOC2012/SegmentationClass/2007_000039.png'
            image = Image.open(image_path)
            anno_image_path = Image.open(anno_image_path)
            print("image_shape:", image.size)
            # image.show()
            image_datatransform = DataTransform(300, color_mean, color_std)
            image_out = image_datatransform('train', image, anno_image_path)
            image_out_0 = image_out[0]
            # image_pil = Image.fromarray(image_out_0, mode='RGB')
            # print(image_out[0].detach().numpy())
            print(type(image_out_0))
            image_out_0.show()

        # image_open()
        # ===========以上代码段为测试单张图片作预处理(不包含归一化和转tensor)后的模样========

        # ===========以下代码段为确认执行结果===========
        # 获取图像列表
        root_path = "E:/数据集/VOCdevkit/VOC2012/"
        train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=root_path)

        # (RGB)颜色的平均值和均方差
        color_mean = (0.485, 0.456, 0.406)
        color_std = (0.229, 0.224, 0.225)

        # 生成数据集
        train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
        val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
        # print(len(train_dataset))
        # print(len(val_dataset))
        # print(val_dataset.__getitem__(0)[0].shape)
        # print(val_dataset.__getitem__(0)[1].shape)
        # print(train_dataset.__getitem__(0))
        # ===========以上代码段为确认执行结果===========

        # ===========以下代码段为创建数据加载器=============
        batch_size = 8

        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 集中保存到字典型变量中
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

        # 确认执行结果
        batch_iterator = iter(dataloaders_dict["val"])
        images, anno_class_imgs = next(batch_iterator)
        # print(images.size())  # torch.Size([8, 3, 475, 475])
        # print(anno_class_imgs.size())  # torch.Size([8, 475, 475])
        # ===========以上代码段为创建数据加载器=============

        # ===========以下代码段从Dataset文件夹读取图像、读取的图像的结果和标注数据的绘制====================
        import numpy as np
        import matplotlib.pyplot as plt
        
        # 每次运行都会发生变化
        # 导入图像数据
        index = 0
        images, anno_class_imgs = train_dataset.__getitem__(index)

        # 显示图像
        img_val = images  # 此时为PIL类型
        img_val = img_val.numpy().transpose((1, 2, 0))  # 转为numpy，并改变形状为(H, W, C)
        plt.imshow(img_val)
        plt.show()

        # 显示标注图像
        anno_file_path = train_anno_list[0]
        anno_class_img = Image.open(anno_file_path)
        p_palette = anno_class_img.getpalette()

        anno_class_img_val = anno_class_imgs.numpy()
        anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
        anno_class_img_val.putpalette(p_palette)
        plt.imshow(anno_class_img_val)
        plt.show()

        # 绘制验证图像
        # 导入图像数据
        index = 0
        images, anno_class_imgs = val_dataset.__getitem__(index)

        # 显示图像
        img_val = images
        img_val = img_val.numpy().transpose((1, 2, 0))
        plt.imshow(img_val)
        plt.show()

        # 显示标注图像
        anno_file_path = train_anno_list[0]
        anno_class_img = Image.open(anno_file_path)
        p_palette = anno_class_img.getpalette()

        anno_class_img_val = anno_class_imgs.numpy()
        anno_class_img_val = Image.fromarray(np.uint8(anno_class_img_val), mode="P")
        anno_class_img_val.putpalette(p_palette)
        plt.imshow(anno_class_img_val)
        plt.show()
        # ===========以下代码段从Dataset文件夹读取图像、读取的图像的结果和标注数据的绘制====================

    # main()