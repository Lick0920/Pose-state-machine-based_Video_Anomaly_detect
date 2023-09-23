import os
import numpy as np

# 获取文件夹中文件数量
folder_path = 'F:\dataset_anomaly230906\\test_fight_frame\\20_29_4'
num_files = len(os.listdir(folder_path))

# 生成一个长度为文件夹中文件数量的 0 数组
my_array = np.zeros(num_files)
my_array[380:-1] = 1.0
# 将数组保存为一个名为 my_array.npy 的文件
np.save('save_path/own_s4/my_array.npy', my_array)

