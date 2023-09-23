import numpy as np
import torch
import numpy as np


# 指定随机数组的形状
shape = (1, 24, 18, 2)

# 生成随机数组
random_array = np.random.rand(*shape)
ra2 = np.array([3840,2160])

a = random_array / ra2
arr = []
# 找到每一行中不为0的元素的索引
indices = np.argwhere(arr.sum(axis=1)> 0)

# 将这些索引的数组选出来放到新的数组中
new_arr = arr[indices.T[0], :]

classes = np.load("d:\project_python\Accurate-Interpretable-VAD\data\shanghaitech\shanghaitech_bboxes_test_classes.npy"
              ,allow_pickle=True)
t= np.load("D:\project_python\Accurate-Interpretable-VAD\data\ped2\yolox_track_train_cls_bboxs.npy"
           ,allow_pickle=True)
for item in t:
    for obj in item:
        if int(obj[6][1]) != 0:
            print(obj[6][1])