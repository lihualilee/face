from itertools import product as product
from math import ceil

import torch


class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        self.min_sizes  = cfg['min_sizes']
        self.steps      = cfg['steps']
        self.clip       = cfg['clip']
        #   图片的尺寸
        self.image_size = image_size
        #   三个有效特征层高和宽
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        #     80,40,20

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            #   对特征层的高和宽进行循环迭代
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    # anchors与输入大小之间的归一化   [16]---->[16/640]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    # 0.5*8//640--->4 bbox center  [4]----->[4/640]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        # [16800,4] 所有的先验框 大小经过了归一化处理的,坐标的格式为x-y-w-h
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
