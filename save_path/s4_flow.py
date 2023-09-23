import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
import sys
import os
sys.path.append('D:\\project_python\\STG-NF\\utils_r')
gt = np.load('utils_r\\test_gt.npy',allow_pickle=True)
# 把gt 的 0 1 颠倒
# gt = 1 - gt
vel = np.load('utils_r\\final_scores_velocity.npy',allow_pickle=True)
optical_nf = np.load('D:\\project_python\\STG-NF\\save_path\\shanghaitech\\shanghaitech_5\\normality_scores_ep4.npy',allow_pickle=True)
pose_nf = np.load('utils_r\\scores_nfpose.npy',allow_pickle=True)
apose = np.load('utils_r\\final_scores_pose.npy',allow_pickle=True)
s4_score = np.load('save_path\S4shanghai_d64_f20\\normality_scores_ep12.npy',allow_pickle=True)
# s4_reverse = np.load("save_path\\S4shanghai_reverse\\normality_scores_ep7.npy",allow_pickle=True)
label_score = np.load("save_path\\S4shanghai\\labeled_normality_scores_ep50.npy",allow_pickle=True)  

deep = np.load('utils_r\\final_scores_deep.npy',allow_pickle=True) # 
# deep = np.load('utils\\final_scores.npy',allow_pickle=True) # ###flow 试试超参数

sys.path.append('D:\\project_python\\STG-NF\\utils_r')
root = "D:\project_python\Accurate-Interpretable-VAD\data"

test_clip_lengths = np.load(os.path.join(root, "shanghaitech", 'test_clip_lengths.npy'))

##  ['walking man', 'person',  'skating boy','riding bicycle', 'riding e-bike', 'driving car'] 
person_p = np.load("D:\project_python\Accurate-Interpretable-VAD\models\clip\saved_res\shanghaitech\person_ornot.npy",allow_pickle=True)
# 归一化
# deep = (deep - deep.min()) / (deep.max() - deep.min()) 
# s4_score = (s4_score - s4_score.min()) / (s4_score.max() - s4_score.min())
# vel = (vel - vel.min()) / (vel.max() - vel.min())
def macro_auc(video, test_labels, lengths):
    prev = 0
    auc = 0
    for i, cur in enumerate(lengths):
        cur_auc = roc_auc_score(np.concatenate(([0], test_labels[prev: cur], [1])),
                             np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)

final_score =  - label_score
auc = roc_auc_score(gt, final_score)
print(auc)
print(macro_auc(final_score, gt, test_clip_lengths))

# lam = 0.3
# for lam in np.arange(0.35,0.45,0.01):
#     print("S4 lam:",lam)
#     final_score = - lam * s4_score  + (1-lam)*vel  # 0.8143640082873005  失败，两个精度差别太多, 没有互补，而是和vel可以互补


#     auc = roc_auc_score(gt, final_score)
#     print(auc)


## - S4 score
### 都是漏检，没有互补
# 计算数组的平均值
# mean_value = np.mean(vel)

# max = vel.max()
# min = vel.min()
# # print(int(min),int(max))
# step = 0.1
# auc = roc_auc_score(gt, - s4_score +vel) 
# print(auc)
