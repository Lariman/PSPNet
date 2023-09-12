from utils.datapath import make_datapath_list
from DataLoader import DataTransform
from PSPNet import PSPNet
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch

# 创建文件路径列表
rootpath = "E:/数据集/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# 实例化PSPNet模型
net = PSPNet(n_classes=21)

# 载入已经训练完毕的参数
state_dict = torch.load("./weights/pspnet50_30.pth",
                        map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)
print('网络设置完毕：成功载入了训练完毕的权重。')

# 1. 显示原有图像
image_file_path = "./data/cowboy-757575_640.jpg"
img = Image.open(image_file_path)
img_width, img_height = img.size
plt.imshow(img)
plt.show()

# 2.创建预处理类
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

# 3. 预处理
# 准备好适当的标注图像，并从中读取彩色调色板信息
anno_file_path = val_anno_list[0]
anno_class_img = Image.open(anno_file_path)
p_palette = anno_class_img.getpalette()
phase = "val"
img, anno_class_img = transform(phase, img, anno_class_img)

# 4. 使用PSPNet进行推测
net.eval()
x = img.unsqueeze(0)
outputs = net(x)
y = outputs[0]

# 5. 从PSPNet的输出结果求取最大分类，并转换为颜色调色板格式，将图像尺寸恢复为原有尺寸
y = y[0].detach().numpy()
y = np.argmax(y, axis=0)
anno_class_img = Image.fromarray(np.uint8(y), mode="P")
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
anno_class_img.putpalette(p_palette)
plt.imshow(anno_class_img)
plt.show()

# 6. 将图像透明化并重叠在一起
trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
anno_class_img = anno_class_img.convert('RGBA')  # 将彩色调色板格式转换为RGBA格式

for x in range(img_width):
    for y in range(img_height):
        # 获取推测结果的图像的像素数据
        pixel = anno_class_img.getpixel((x, y))
        r, g, b, a = pixel

        # 如果是(0, 0, 0)的背景，直接透明化
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            continue
        else:
            # 将除此之外的颜色写入准备好的图像中
            trans_img.putpixel((x, y), (r, g, b, 150))
            # 150指定的是透明度大小

img = Image.open(image_file_path)  
result = Image.alpha_composite(img.convert('RGBA'), trans_img)
plt.imshow(result)
plt.show()
