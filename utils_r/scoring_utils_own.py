import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset_backup0823 import shanghaitech_hr_skip


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def score_dataset(score, metadata, args=None):
    gt_arr, scores_arr = get_dataset_scores(score, metadata, args=args)
    # ### 不smooth
    np.save('F:\dataset_anomaly230906\data_output\single_test\\DJI_0144_scores_vibe.npy', scores_arr)
    # np.save('save_path\\own_s4\\gt_arr_walk_fight_snachers.npy', gt_arr)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)  # 40791帧
    scores_np = np.concatenate(scores_arr) # 40791
    auc = score_auc(scores_np, gt_np)
    return auc, scores_np


def get_dataset_scores(scores, metadata, args=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    # per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask_split/'
    # per_frame_scores_root = 'save_path\\own_s4'
    per_frame_scores_root = 'F:\dataset_anomaly230906\data_output\single_test\score_gt_saved'
    # per_frame_scores_root = 'F:/dataset_anomaly230906/groudtruth/gt_walk_fight_snachers'
    clip_list = os.listdir(per_frame_scores_root)
    
    clip_list = sorted([fn for fn in clip_list if fn.endswith('.npy')], key = lambda x: (x.split('.')[0].split('_')[0], x.split('.')[0].split('_')[1]))


    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    return dataset_gt_arr, dataset_scores_arr


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()

    auc = roc_auc_score(gt, scores_np)

    return auc


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):
    
    # scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
    scene_id, clip_id = clip.split('.')[0].split('_')[0:2]
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]   ## metadata 记录了属于人物属于哪一帧的索引
    clip_metadata = metadata[clip_metadata_inds] # ## 453408 4?
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn,allow_pickle=True)
    # if args.dataset != "UBnormal":
    #     clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where( # 根据人物的图形索引从元数据中获取人物的元数据索引
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds] # 获取人物的得分

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int) # 获取人物在剪辑中的帧索引
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))  # 5 个人， 其实就是取每一帧中的得分最低的人
    clip_score = np.amin(clip_ppl_score_arr, axis=0)  # 最小的值 0为异常 1为正常 

    return clip_gt, clip_score
