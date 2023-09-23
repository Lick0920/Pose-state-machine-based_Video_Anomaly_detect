import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr

def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    ############## 魔改 加点 vecolity信息 ###### 0 1 是反的
    # np.save('save_path\S4shanghai\labeled_normality_scores_ep50.npy', scores_np)
    # vel = np.load('utils\\final_scores_velocity.npy')
    # final_score =  scores_np - vel

    # auc = roc_auc_score(gt, final_score)
    auc = roc_auc_score(gt, scores_np)

    return auc

scores_arr = np.load("save_path\own_s4\\scores\\train_score_72.npy", allow_pickle=True)
gt_arr = np.load("save_path\\own_s4\\20_29_4_gt_fight.npy", allow_pickle=True)
# gt_np = np.concatenate(gt_arr)  # 40791帧
for i in range(1, 30):  # 10 0.9707603242399988
    sigma = i
    scores_arr = smooth_scores(scores_arr,sigma=sigma)
    scores_np = np.concatenate(scores_arr) # 40791
    auc = score_auc(scores_np, gt_arr)
    print(sigma, auc*100)

