import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


#   获得框的左上角和右下角
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)


#   获得框的中心和宽高
def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,boxes[:, 2:] - boxes[:, :2], 1)


#   计算所有真实框和先验框的交面积
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    #   获得交矩形的左上角
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #   获得交矩形的右下角
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    #   计算先验框和所有真实框的重合面积
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    '''
    :param box_a:   [N1,4]
    :param box_b:   [N2,4]
    :return:
    '''
    #   返回的inter的shape为[A,B]
    #   代表每一个真实框和先验框的交矩形
    inter = intersect(box_a, box_b)
    # 【N1,N2】 每一个gt框与先验框之间的重叠面积
    #   计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    # 真实框的面积
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 先验框的面积

    union = area_a + area_b - inter
    #   每一个真实框和先验框的交并比[A,B]
    return inter / union  # [A,B]


def encode(matched, priors, variances):
    # 进行编码的操作
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # 中心编码
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    # 宽高编码
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_landm(matched, priors, variances):
    '''
    :param matched:    (16800,10)---->(16800,5,2)  5个关键点坐标信息
    :param priors:      (16800,4) xywh
    :param variances:
    :return:
    '''
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    # [16800]-->[16800,1]-->[16800,5]---->[16800,5,1]  先验框的中心坐标x
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    # [16800,5,4]

    # 减去中心后除上宽高
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    # 相对于中心坐标的偏移量再对先验框的大小进行归一化,这里跟回归dx是一致的
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # 再次转换为网络输入的格式
    # [16800,5,2]------------>[16800,10]
    return g_cxcy


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
'''
这个操作有点类似于防止softmax防止数据溢出的做法
'''


def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    '''
    iou阈值,[N,4],[16800,4],(0.1,0.2),[N,],[N,10],[b,16800,4],[b,16800],[b,16800,10],i
    真实框,先验框,类别标签,关键点坐标信息,接受损失值的数组
    '''
    #   计算所有的先验框和真实框的重合程度
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    '''
    先验框xywh格式转为x1y1x2y2的格式---->[16800,4]
    得到的结果是[n1,n2]
    '''

    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,1]

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    #   用于保证每个真实框都至少有对应的一个先验框
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    # 很关键,这一操作与原始所看的一些相比确保了每一个gt至少有一个先验框与其匹配

    #   获取每一个先验框对应的真实框[num_priors,4]
    matches = truths[best_truth_idx]
    # [n1,4]------------->[16800,4]  每一个先验框对应的真实标签信息
    # Shape: [num_priors] 此处为每一个anchor对应的label取出来
    conf = labels[best_truth_idx]
    # 每一个先验框对应的真实框的类别信息,这里的类别索引只有两种转态,分别是类别1和类比-1(无效的关键点)
    matches_landm = landms[best_truth_idx]
    # 每一个先验框对应的真实的5个关键点的坐标信息

    '''
    [16800,4]    [16800]    [16800,10]
    '''

    #   如果重合程度小于threhold则认为是背景
    conf[best_truth_overlap < threshold] = 0
    # 原先仅有2种状态,现在有3种类状态
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    loc = encode(matches, priors, variances)
    '''
    [16800,4]-->gt   [16800,4]---->anchors  (0.1,0.2)--->variances
    gt框为xyxy的格式,anchors为xywh的坐标格式
    为每一个先验框赋值bbox框坐标偏移量
    
    '''
    landm = encode_landm(matches_landm, priors, variances)
    '''
    [16800,10]-->gt   [16800,4]---->anchors  (0.1,0.2)--->variances
    anchors为xywh的坐标格式
    为每一个先验框赋值坐标偏移量

    '''

    #   [num_priors, 4]
    loc_t[idx] = loc
    #   [num_priors]
    conf_t[idx] = conf
    #   [num_priors, 10]
    landm_t[idx] = landm


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos, variance, cuda=True):
        super(MultiBoxLoss, self).__init__()
        #   对于retinaface而言num_classes等于2
        self.num_classes    = num_classes

        #   重合程度在多少以上认为该先验框可以用来预测
        #   阈值越低的话可以设置更多的正样本
        self.threshold      = overlap_thresh
        #   正负样本的比率
        self.negpos_ratio   = neg_pos
        self.variance       = variance
        self.cuda           = cuda

    def forward(self, predictions, priors, targets):
        '''
        predictions---------->[[b,16800,4],[b,16800,2],[b,16800,10]]
        priors--------------->[16800,4] 所有的先验框 大小经过了归一化处理的,坐标的格式为x-y-w-h
        targets-------------->[[n1,15],[n2,15].....[nb,15]]
        '''
        #   取出预测结果的三个值：框的回归信息，置信度，人脸关键点的回归信息

        loc_data, conf_data, landm_data = predictions
        #   计算出batch_size和先验框的数量
        num         = loc_data.size(0)
        num_priors  = (priors.size(0))

        #   创建一个tensor进行处理
        loc_t   = torch.Tensor(num, num_priors, 4)
        # [b,16800,4]
        landm_t = torch.Tensor(num, num_priors, 10)
        # [b,16800,10]
        conf_t  = torch.LongTensor(num, num_priors)
        # [b,16800]

        # 对一批次中的每一个样本进行迭代
        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :4].data
            # [N,4]
            labels = targets[idx][:, -1].data
            # [N,]
            landms = targets[idx][:, 4:14].data
            # [N,10]

            # 获得先验框
            defaults = priors.data
            # [16800,4]  xywh normalize to (0,1)
            #   利用真实框和先验框进行匹配。
            #   如果真实框和先验框的重合度较高，则认为匹配上了。
            #   该先验框用于负责检测出该真实框。
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
            '''
            iou阈值,[N,4],[16800,4],(0.1,0.2),[N,],[N,10],[b,16800,4],[b,16800],[b,16800,10],i
            真实框,先验框,类别标签,关键点坐标信息,接受损失值的数组
            设置真实的偏移量,包括类别,回归偏移量,以及关键点坐标偏移量
            '''

        #   转化成Variable
        #   loc_t   (num, num_priors, 4)
        #   conf_t  (num, num_priors)
        #   landm_t (num, num_priors, 10)
        zeros = torch.tensor(0)
        if self.cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            zeros = zeros.cuda()

        #   有人脸关键点的人脸真实框的标签为1，没有人脸关键点的人脸真实框标签为-1
        #   所以计算人脸关键点loss的时候pos1 = conf_t > zeros
        #   计算人脸框的loss的时候pos = conf_t != zeros
        pos1 = conf_t > zeros
        # [b,16800]--------->[b,16800] T/F 找出正样本
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # 2d matrix
        # [b,16800]------->[b,16800,10] 一个批次中的正样本信息 用于回归任务的mask图
        landm_p = landm_data[pos_idx1].view(-1, 10)
        # 预测值,正样本的预测值---->正样本掩码图找出所有的正样本
        # [b,16800,10]------------->[batch_pos_num,10]
        landm_t = landm_t[pos_idx1].view(-1, 10)
        # 真实值,正样本的真实值---->正样本掩码图找出所有的正样本
        # [b,16800,10]------------->[batch_pos_num,10]
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        # 关键点的回归使用smooth l1 损失函数,损失求和的形式求所有关键点回归损失

        # 此处有两种掩码图,一种是关键点正样本掩码图,一种框的正样本的掩码图,类别为-1和类别为0的目标都不需要计算关键点回归loss
        # 重新思考正负样本的定义,以及可忽略样本的定义方式
        
        pos = conf_t != zeros
        # 正样本的掩码图----->[b,16800]--------->[b,16800]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # [b,16800]----------->[b,16800,4]
        loc_p = loc_data[pos_idx].view(-1, 4)
        # 所有的正样本目标框的gt框
        # [b,16800,4]-------------->[pos_anchors_num,4]
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 所有的正样本目标的预测结果
        # [b,16800,4]-------------->[pos_anchors_num,4]
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # 此处也是用的smooth l1损失

        #   batch_conf  (num * num_priors, 2)
        #   loss_c      (num, num_priors)
        conf_t[pos] = 1
        # 此处分类的正样本定义时不管类别为-1和1都是正样本,因为其都是人脸目标
        # 标签的label信息-------->[b,16800]----------->[b*16800,1]
        batch_conf = conf_data.view(-1, self.num_classes)
        # [b,16800,2]-------------->[b*16800,2] 一批次中所有的正负样本数据 网络的输出
        # 这个地方是在寻找难分类的先验框
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        '''
        这里相当于一个评估器的作用,当网络的输出更接近标签值时,其代价会越小,否则其代价越大
        后续通过gather里面的标签索引取出网络的输出预测的真实值位置上所对应的结果
        根据这个代价矩阵来进行难例样本的挖掘,损失越小越好,越大代表越难分类的负样本
        log存在将最低损失降到了0
        [b*16800,1]
        '''

        # 难分类的先验框不把正样本考虑进去，只考虑难分类的负样本
        loss_c[pos.view(-1, 1)] = 0
        # [b*16800,1]======>仅对负样本的分类状态进行难例挖掘
        loss_c = loss_c.view(num, -1)
        # [b*16800,1]--------->[b,16800]

        #   loss_idx    (num, num_priors)
        #   idx_rank    (num, num_priors)
        _, loss_idx = loss_c.sort(1, descending=True)
        # 代价矩阵排序,对每一个样本中的代价矩阵输出进行排序
        # [b,16800] 里面保存的是每排序的索引
        # 第一名idx,第二名idx,....第16800的idx

        _, idx_rank = loss_idx.sort(1)

        # 得到每一个网格的分类输出对应的排序索引

        #   求和得到每一个图片内部有多少正样本
        #   num_pos     (num, )
        #   neg         (num, num_priors)
        num_pos = pos.long().sum(1, keepdim=True)
        # [b, 16800]
        # 统计每一张图片中所对应的正样本目标的数量
        # [b,1] 里面的值是正样本目标的数量

        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        #   求和得到每一个图片内部有多少正样本
        #   pos_idx   (num, num_priors, num_classes)
        #   neg_idx   (num, num_priors, num_classes)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        
        # 选取出用于训练的正样本与负样本，计算loss
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        loss_landm /= N1
        return loss_l, loss_c, loss_landm


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
            #     min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
            #  经过一定的epoch次lr会达到最小值,后续关闭数据增强时lr不再变化,一直维持该最小值
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        # min(max(150*0.1,1),3    预测环节的epoch数大概在1-3之间
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        # 预热的起始的学习率,默认为基准学习率的0.1倍
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        # 在训练结束前15个epoch会关闭数据增强功能,这里设置数据增强关闭的epoch数目
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #   设置学习率
