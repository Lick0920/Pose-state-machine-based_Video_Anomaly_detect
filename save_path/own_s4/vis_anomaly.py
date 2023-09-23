import numpy as np
import matplotlib.pyplot as plt
scores_arr = np.load("D:\project_python/STG-NF\save_path\own_s4/scores_walk_fight_snachers_ep11.npy", allow_pickle=True)
gt_arr = np.load("D:\project_python/STG-NF\save_path\own_s4/gt_arr_walk_fight_snachers.npy", allow_pickle=True)
clip_lenth = np.load("F:\dataset_anomaly230906\groudtruth\walk_fight_snachers_clip_lengths.npy", allow_pickle=True)
flist = "F:\dataset_anomaly230906\groudtruth\gt_walk_fight_snachers"

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(2, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr

def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc = roc_auc_score(gt, scores_np)

    return auc


gt_np = np.concatenate(gt_arr)  # 40791帧
# scores_arr = np.concatenate(scores_arr)
# for i in range(1, 15):  # 10 0.9707603242399988
#     sigma = i
#     scores_arr = smooth_scores(scores_arr,sigma=sigma)
#     scores_arr = smooth_scores(scores_arr,sigma=3)
#     scores_np = np.concatenate(scores_arr) # 40791
#     auc = score_auc(scores_np, gt_np)
#     print(sigma, auc*100)

# 得到文件夹中文件夹的list
flist = os.listdir(flist)


scores = np.concatenate(scores_arr)
prev = 0
clip_i = 0
auc_list = []
for cur in clip_lenth:
    scores[prev: cur]
    # 一维数组的数据
    data = scores[prev: cur]
    # gt = 1 - gt_np[prev: cur]
    gt = gt_np[prev: cur]
    data_1 = - data
    # 创建横轴标签
    x_labels = range(len(data)//20 + 1)

    # # 创建纵轴标签
    # y_labels = range(1, max(data) + 1)

    # 绘制网格图
    plt.plot(data_1)
    plt.plot(gt)
    # 设置横轴和纵轴标签
    # plt.xticks(range(0, len(data), 10), range(1, len(data)+1, 10),fontsize=4)
    x_range = range(0, len(data), 20) if not len(data) % 20 ==0 else range(0, len(data)+1, 20)
    plt.xticks(x_range, x_labels[::1],fontsize=4)
    # plt.yticks(y_labels)
    # 如果gt 全是 1
    if gt.all() == 0:
        gt[0] = 1
    # 如果gt 全是 0 ,跳过
    # if gt.all() == 1: 
    #     gt[0] = 0
    # if gt.all() == 0: 
    #     gt[0] = 1
    
    # 算的时候换成 1-gt
    auc = roc_auc_score(1-gt, data)
    if not(gt.all() == 1 or gt.all()== 0):
        auc_list.append(auc)
    # 添加标题和标签
    plt.title('Grid Plot' + str(auc))
    plt.xlabel('Index')
    plt.ylabel('Value')

    
    # 添加网格线
    plt.grid(True)
    # 保存图形到指定目录
    plt.savefig('save_path\\own_s4//scores_fig/'+ flist[clip_i].split('.')[0] +'_grid_plot.png')
    # plt.show()
    plt.close()
    clip_i += 1
    prev = cur
print(np.mean(auc_list))