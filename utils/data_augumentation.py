import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np

class Compose(object):
    """
        按顺序执行参数transforms中存储的各种预处理的类
        同时转换目标图像和注释图像。
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img

class Scale(object):
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, img, anno_class_img):
        width = img.size[0]  # img.size=[幅][高さ]
        height = img.size[1]  # img.size=[幅][高さ]

        scale = np.random.uniform(self.scale[0], self.scale[1])  # 从scale[0]-scale[1]中随机采样

        scaled_w = int(width * scale)  # img.size=[宽度][高度]
        scaled_h = int(height * scale)  # img.size=[宽度][高度]

        # 调整图像大小
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # 调整标注图像的大小
        anno_class_img = anno_class_img.resize((scaled_w, scaled_h), Image.NEAREST)

        # 将图像设置为原始大小
        # 若缩放后的图像大于原图，则将多余的裁剪掉
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h - height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop((left, top, left+width, top+height))
        # 若缩放后的图像小于原图，则用调色板颜色补足
        else:
            p_palette = anno_class_img.copy().getpalette()  # getpalette():以列表形式返回图像调色板

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width - scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height - scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_class_img = Image.new(anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original, (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)
        
        return img, anno_class_img
    
# 随机旋转
class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):
        # 设定随机旋转角度
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))
        # 旋转
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img

# 随机镜像（50%几率）
class RandomMirror(object):
    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img

# 改变尺寸
class Resize(object):
    def __init__(self, input_size):
        self.input_size = input_size
    
    def __call__(self, img, anno_class_img):
        img = img.resize((self.input_size, self.input_size), Image.BICUBIC)
        anno_class_img = anno_class_img.resize((self.input_size, self.input_size), Image.NEAREST)
        return img, anno_class_img

# 归一化并转成Tensor格式
class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std
    
    def __call__(self, img, anno_class_img):
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, self.color_mean, self.color_std)
        anno_class_img = np.array(anno_class_img)
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0
        anno_class_img = torch.from_numpy(anno_class_img)
        return img, anno_class_img