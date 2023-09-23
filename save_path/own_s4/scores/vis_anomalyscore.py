import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

def vis(scores, groudt, sigma):
    test_clip_lengths = [2315]
    prev = 0
    clip_i = 0
    auc_list = []
    # 得到文件夹中文件夹的list
    flist = "F:\\dataset_anomaly230906\\data_output\\test_cls_posevibe"
    flist = [os.path.join(flist, f) for f in os.listdir(flist) if os.path.isdir(os.path.join(flist, f))]
    for cur in test_clip_lengths:
        scores[prev: cur]
        # 一维数组的数据
        data = scores[prev: cur]
        gt = 1 - groudt[prev: cur]
        gt_1 = groudt[prev: cur]
        data_1 = - data
        # 创建横轴标签
        x_labels = range(len(data)//10 + 1)

        # # 创建纵轴标签
        # y_labels = range(1, max(data) + 1)

        # 绘制网格图
        plt.plot(data_1)
        plt.plot(gt_1)
        # 设置横轴和纵轴标签
        # plt.xticks(range(0, len(data), 10), range(1, len(data)+1, 10),fontsize=4)
        plt.xticks(range(0, len(data), 10), x_labels[::1],fontsize=4)
        # plt.yticks(y_labels)
        # 如果gt 全是 1
        if gt.all() == 0:
            gt[0] = 1
        auc = roc_auc_score(gt, data)
        auc_list.append(auc)
        # 添加标题和标签
        plt.title('Grid Plot' + str(auc))
        plt.xlabel('Index')
        plt.ylabel('Value')

        # 添加网格线
        plt.grid(True)
        # 保存图形到指定目录
        plt.savefig('save_path\\own_s4//'+ flist[clip_i].split('\\')[-1] + str(sigma) + 'sigma_grid_plot.png')
        # plt.show()
        plt.close()
        clip_i += 1
        prev = cur

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

def macro_auc(video, test_labels, lengths):
    prev = 0
    auc = 0
    for i, cur in enumerate(lengths):
        cur_auc = roc_auc_score(np.concatenate(([0], test_labels[prev: cur], [1])),
                             np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)

scores_arr = np.load("save_path\own_s4\\scores\\train_score_72.npy", allow_pickle=True)
gt_arr = np.load("save_path\\own_s4\\20_29_4_gt_fight.npy", allow_pickle=True)
# gt_np = np.concatenate(gt_arr)  # 40791帧
for i in range(1, 30):  # 10 0.9707603242399988
    sigma = i
    scores_arr = smooth_scores(scores_arr,sigma=sigma)
    scores_np = np.concatenate(scores_arr) # 40791
    auc = score_auc(scores_np, gt_arr)
    vis(scores_np, gt_arr, sigma)
    print(sigma, auc*100)


sys.path.append('D:\\project_python\\STG-NF\\utils_r')

groudt = np.load('D:\project_python\\STG-NF\\save_path\\own_s4\\20_29_4_gt_fight.npy',allow_pickle=True)

vibe_score = np.load('train_score_72.npy',allow_pickle=True)
# print("s4,",s4_score.mean(),s4_score.min(),s4_score.max(),"deep,",deep.mean(),deep.min(),deep.max())
# test_clip_lengths = np.load(os.path.join(root, "shanghaitech", 'test_clip_lengths.npy'))
global test_clip_lengths
test_clip_lengths= [2315]


##  ['walking man', 'person',  'skating boy','riding bicycle', 'riding e-bike', 'driving car'] 
# person_p = np.load("D:\project_python\Accurate-Interpretable-VAD\models\clip\saved_res\shanghaitech\person_ornot.npy",allow_pickle=True)
# def guass_video(scores, lengths, sigma=5):
#     scores_g = np.zeros_like(scores)
#     prev = 0
#     for cur in lengths:
#         scores_g[prev: cur] = gaussian_filter1d(scores[prev: cur], sigma)
#         prev = cur
#     return scores_g

# # s4_score = guass_video(s4_score, test_clip_lengths, sigma=3)
# print(roc_auc_score(1-gt,  s4_score))


