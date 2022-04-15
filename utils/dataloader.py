import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

from utils.utils import preprocess_input


class DataGenerator(data.Dataset):
    def __init__(self, txt_path, img_size):
        self.img_size = img_size
        self.txt_path = txt_path

        self.imgs_path, self.words = self.process_labels()
        #    read all img and label info

    def __len__(self):
        return len(self.imgs_path)

    def get_len(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        #   打开图像，获取对应的标签
        img         = Image.open(self.imgs_path[index])
        labels      = self.words[index]
        annotations = np.zeros((0, 15))
        # empty 2d matrix

        if len(labels) == 0:
            return img, annotations
            # no target

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            #   bbox 真实框的位置
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            # x1y1wh---------->x1y1x2y2

            #   landmarks 人脸关键点的位置
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
                #  虽然有框,但是其关键点无法标注(看不清或信息被遮挡)
            else:
                annotation[0, 14] = 1
                #   人脸的关键点信息
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        # [n,15] 坐标大小对应于原图,没有经过任何的归一化处理

        img, target = self.get_random_data(img, target, [self.img_size,self.img_size])
        # [h,w,3] [N,15] x1y1x2y2 10 point 最后一个转态是人脸是否有效,其值为1或者是-1

        img = np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1))
        # 减去均值,但是没有除以方差的数据预处理,那么均值应该是在某个数据集上统计得到的
        # RGB格式数据,是Image读取的,仅后来转化为np格式,但是其图像格式仍未RGB格式
        return img, target

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, targes, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        iw, ih  = image.size
        h, w    = input_shape
        box     = targes

        #   对图像进行缩放并且进行长和宽的扭曲
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        # 缩放比例,小于1或者是大于1
        scale = self.rand(0.25, 3.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        # 对图片随机进行长宽的缩放,不再是保持不失真的resize

        #   将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        # dx,dy 为起始点
        image = new_image
        # 随机补灰条,图像大部分区域都在右下角

        #   翻转图像
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #   对图像进行色域变换
        #   计算色域变换的参数
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #   将图像转到HSV上
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #   应用变换
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        # 色域变换的数据增强

        #   对真实框进行调整
        if len(box)>0:
            np.random.shuffle(box)
            # 随机乱序...
            box[:, [0,2,4,6,8,10,12]] = box[:, [0,2,4,6,8,10,12]]*nw/iw + dx
            # 偶数是x坐标,乘以w的缩放比例再加上x轴方向上的偏移量
            box[:, [1,3,5,7,9,11,13]] = box[:, [1,3,5,7,9,11,13]]*nh/ih + dy
            # 奇数是y坐标,乘以h的缩放比例再加上y轴方向上的偏移量
            if flip: 
                box[:, [0,2,4,6,8,10,12]] = w - box[:, [2,0,6,4,8,12,10]]
                box[:, [5,7,9,11,13]]     = box[:, [7,5,9,13,11]]
                #  画图理解此处坐标变换的意义

            # x1y1x2y2
            center_x = (box[:, 0] + box[:, 2])/2
            center_y = (box[:, 1] + box[:, 3])/2
        
            box = box[np.logical_and(np.logical_and(center_x>0, center_y>0), np.logical_and(center_x<w, center_y<h))]
            # 框的中心在图像内部才可以算作是正常的目标框
            # 只要数据的标注没有问题,这一步显得没必要

            box[:, 0:14][box[:, 0:14]<0] = 0
            # 去除无效关键点,即为坐标为-1的坐标点
            # 左上角越界的话置零处理,但最主要的作用应该是筛除无效关键点
            box[:, [0,2,4,6,8,10,12]][box[:, [0,2,4,6,8,10,12]]>w] = w
            box[:, [1,3,5,7,9,11,13]][box[:, [1,3,5,7,9,11,13]]>h] = h
            # 越界的框置边界处理
            
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # 重新求得框的长宽值
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            # 获取所有的有效的框

        box[:,4:-1][box[:,-1]==-1]=0
        # no need
        box[:, [0,2,4,6,8,10,12]] /= w
        box[:, [1,3,5,7,9,11,13]] /= h
        box_data = box
        # bbox size normalize to (0,1)
        return image_data, box_data
        
    def process_labels(self):
        imgs_path = []
        words = []
        f = open(self.txt_path,'r')
        lines = f.readlines()
        # read txt ,per line a list data
        isFirst = True
        labels = []
        # 单张图片对应的所有的目标信息
        for line in lines:
            line = line.rstrip()
            # delete ' '
            if line.startswith('#'):
                # 图片来了
                if isFirst is True:
                    isFirst = False
                    #   关闭first功能
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                    #  update img label info
                path = line[2:]
                # img path and name 当前的相对path,相对label.txt而言
                path = self.txt_path.replace('label.txt','images/') + path
                # 获取img的相对path,相当于train.py而言
                # data/wideface/iamges/*/*.jpg
                imgs_path.append(path)
                #   update img path list
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
                # x1y1wh + 3*5 + 1 大小没有经过归一化处理的
        words.append(labels)
        # update the last label info
        # 【img1，img2，...】
        # 【【【xywh + 3*5 + 1】，【】，【】】，【【】】
        return imgs_path, words


def detection_collate(batch):
    images  = []
    targets = []
    for img, box in batch:
        if len(box)==0:
            continue
        images.append(img)
        targets.append(box)
    images = np.array(images)
    # [B,3,H,W] [[n1,15],[n2,15].....[nn,15]]
    # 4d matrix
    # list
    return images, targets
