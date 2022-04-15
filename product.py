from itertools import product as product
import torch
# for i, j in product(range(40), range(40)):
#     print(i,j)


from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt

# image = Image.fromarray(np.ones((10,10)))
# new_image = Image.new('RGB', (100,100), (128,128,128))
# new_image.paste(image, (0, 0))
#
# new_image = np.array(new_image)
# print(new_image[0:10,0:10])
#
# data = np.arange(0,30).reshape(2,15)
# # print(data)
# # box[:, 0:14][box[:, 0:14]<0] = 0
# print([data[:,0:14]<15])
# data[:,0:14][data[:,0:14]<2]=0
# print(data)
# # print(data[[True,False,True,True,False,True]])
# # print(data[True,False,True,True,False,True])
# state = np.logical_and(data[:,0]>5,data[:,1]>6)
# print(data[state])
# # print(state)
# # data

# matrix_iou_b = torch.from_numpy(np.array([0.4,0.5,0.6,0.7,0.8,0.9,0,0,0,0]))
# matrix_idx_b = torch.from_numpy(np.array([0,0,1,2,2,0,0,0,0,0]))
#
# matrix_iou_a = torch.from_numpy(np.array([0.7,0.8,0.9]))
# matrix_idx_a = torch.from_numpy(np.array([1,2,3],np.uint8))
# matrix_idx_a=matrix_idx_a.long()
# print(matrix_idx_a)
# matrix_iou_b.index_fill_(0,matrix_idx_a,2)
#
# print(matrix_idx_a.size(0))
# for i in range(matrix_idx_a.size(0)):
#     matrix_idx_b[matrix_idx_a[i]] = i
# # for j in range(best_prior_idx.size(0)):
# #     best_truth_idx[best_prior_idx[j]] = j
# print(matrix_idx_b)


matched = torch.from_numpy(np.arange(0,30).reshape(3,10))
priors = torch.from_numpy(np.arange(0,12).reshape(3,4))


matched = torch.reshape(matched, (matched.size(0), 5, 2))
# print(matched)

priors = priors.unsqueeze(1).expand(3,5,4)
# priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
# # [16800]-->[16800,1]-->[16800,5]---->[16800,5,1]  先验框的中心坐标x
# priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
# priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
# priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
# priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
# print(priors)


# priors = torch.from_numpy(np.arange(0,12).reshape(3,4))
# zeros = torch.tensor(0)
# print((priors>zeros).dim())


data = torch.from_numpy(np.random.randn(6,6).reshape(6,6)).float()
# print()
print(data)
_,idx = data.sort(1,descending=True)

_, idx_rank = idx.sort(1)
print(idx,idx.shape,idx_rank)

