# import time
#
# import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#
# t1 = time.time()
# x = torch.rand(200,200)
#
# for i in range(x.size(0)):
#     for j in range(x.size(1)):
#         x[i:j]*=2.0
# cnt = time.time() - t1
# print(cnt)
#
# t1 = time.time()
#
# x[:,:]*=2.0
# cnt = time.time() - t1
# print(cnt)