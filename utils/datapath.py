import os.path as osp
from PIL import Image
import torch.utils.data as data

def make_datapath_list(rootpath):
    """
        创建用于学习、验证的图像数据和标注数据的文件路径列表变量
        输入：rootpath:指向数据文件夹的路径
        输出：ret:train_img_list, train_anno_list, val_img_list, val_anno_list
            保存了指向数据的路径列表变量 
    """

    # 创建指向图像文件和标注数据的路径的模板
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    # 训练和验证，分别获取相应的文件ID（文件名）
    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # 创建指向训练数据的图像文件和标注文件的路径列表变量
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 删除空格和换行符
        img_path = (imgpath_template % file_id)  # 图像的路径
        anno_path = (annopath_template % file_id)  # 标注数据的路径
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
    
    # 创建指向验证数据的图像文件和标注文件的路径列表变量
    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()  # 删除空格和换行符
        img_path = (imgpath_template % file_id)  # 图像的路径
        anno_path = (annopath_template % file_id)  # 标注数据的路径
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
    
    return train_img_list, train_anno_list, val_img_list, val_anno_list

# ========================以下为测试代码段==========================
if __name__ == '__main__':
    # 确认执行结果，获取文件路径列表
    rootpath = "E:/数据集/VOCdevkit/VOC2012/"

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    print(train_img_list[0])
    print(train_anno_list[0])
# ========================以上为测试代码段==========================