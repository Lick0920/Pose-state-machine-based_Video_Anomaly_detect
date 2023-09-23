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

scores_arr = np.load("save_path\\S4shanghai\\smootht_0827_0.9.npy", allow_pickle=True)
gt_arr = np.load("save_path\\S4shanghai\\gt_arr.npy", allow_pickle=True)
gt_np = np.concatenate(gt_arr)  # 40791帧
for i in range(1, 15):  # 10 0.9707603242399988
    sigma = i
    scores_arr = smooth_scores(scores_arr,sigma=sigma)
    scores_arr = smooth_scores(scores_arr,sigma=3)
    scores_np = np.concatenate(scores_arr) # 40791
    auc = score_auc(scores_np, gt_np)
    print(sigma, auc*100)

