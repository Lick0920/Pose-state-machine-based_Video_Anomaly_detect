import os
import json
from posixpath import basename
import argparse
import shutil
import numpy as np

def convert_data_format(data, split='None'):
    if split == 'testing':
        num_digits = 3
    elif split == 'training':
        num_digits = 4
    elif split == 'None':
        num_digits = 4

    data_new = dict()
    for item in data:
        frame_idx_str = item['image_id'][:-4]  # '0.jpg' -> '0'
        frame_idx_str = frame_idx_str.zfill(num_digits)  # '0' -> '000'
        person_idx_str = str(item['idx'])
        keypoints = item['keypoints']
        scores = item['score']
        if not person_idx_str in data_new:
            data_new[person_idx_str] = {frame_idx_str: {'keypoints': keypoints, 'scores': scores}}
        else:
            data_new[person_idx_str][frame_idx_str] = {'keypoints': keypoints, 'scores': scores}

    return data_new

def convert_data_format_2vibe(data, split='None'):
    if split == 'testing':
        num_digits = 3
    elif split == 'training':
        num_digits = 4
    elif split == 'None':
        num_digits = 4

    data_new = dict()
    for item in data:
        frame_idx_str = item['image_id'][:-4]  # '0.jpg' -> '0'
        frame_idx_str = frame_idx_str.zfill(num_digits)  # '0' -> '000'
        # person_idx_str = str(item['idx'])
        person_idx_str = item['idx']
        keypoints = item['keypoints']
        bbox = item['box']
        ##########¸ñÊ½¶ÔÆë
        w,h = bbox[2], bbox[3]
        c_x, c_y = bbox[0] + w/2, bbox[1] + h/2
        w = h = np.where(w / h > 1, w, h).tolist()
        bbox = [c_x, c_y, w, h]
        scores = item['score']

        if not person_idx_str in data_new:
            data_new[person_idx_str] = {frame_idx_str: {'bbox': bbox, 'scores': scores}}
        else:
            data_new[person_idx_str][frame_idx_str] = {'bbox': bbox, 'scores': scores}
    
    data_vibe = dict()
    for idx in (data_new):
        frame = list(data_new[idx].keys())
        bbox = []
        for i in frame:
            bbox.append(data_new[idx][i]['bbox'])
        data_vibe[idx] = {'frames':frame, 'bbox':bbox}

    return data_vibe

def read_convert_write(in_full_fname, out_full_fname):
    # Read results file
    with open(in_full_fname, 'r') as f:
        data = json.load(f)

    # Convert reults file format
    data_new = convert_data_format_2vibe(data)

    # 3. Write
    save = True  # False
    if save:
        with open(out_full_fname, 'w') as f:
            json.dump(data_new, f)

def main():
    # parse command line
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='dir', default="D:\\project_python\\STG-NF\\data\\ShanghaiTech\\pose\\test",type=str, help='video\images dir')
    ap.add_argument('--outdir', dest='outdir', default= "D:\\project_python\\STG-NF\\data\\ShanghaiTech\\pose\\test",type=str, help='video\images outdir')
    ap.add_argument('--alphapose_dir', dest='alphapose_dir', type=str, help='alphapose_dir')
    ap.add_argument('--video', dest='video', action='store_true', help='is video')

    args = ap.parse_args()
    print(args)
    root = args.dir
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    img_dirs = []
    output_files = os.listdir(args.outdir)
    
    in_path = "D:\\project_python\\STG-NF\\data\\ShanghaiTech\\pose\\test\\01_0014_alphapose-results.json"
    # in_path = "D:\\project_python\\STG-NF\\data\\ShanghaiTech\\pose\\train\\01_001_alphapose-results.json"
    out_path = "D:\\project_python\\STG-NF\\data\\vibe_data\\ShanghaiTech\\test\\try_001_0014.json"
    read_convert_write(in_path, out_path)

    # for path, subdirs, files in os.walk(root):
    #     for name in files:
    #         run_pose = False
    #         if args.video and name.endswith(".mp4") or name.endswith("avi"):
    #             video_filename = os.path.join(path, name)
    #             video_basename = basename(video_filename)[:-4]
    #             run_pose = True
    #         elif name.endswith(".jpg") or name.endswith(".png"):
    #             if path not in img_dirs:
    #                 video_filename = path
    #                 img_dirs.append(path)
    #                 video_basename = basename(video_filename)
    #                 run_pose = True
    #         if run_pose:
    #             # Rename constants
    #             alphapose_orig_results_filename = 'alphapose-results.json'
    #             alphapose_tracked_results_filename = video_basename + '_alphapose_tracked_person.json'
    #             alphapose_results_filename = video_basename + '_alphapose-results.json'
    #             print(alphapose_results_filename)
    #             if alphapose_results_filename in output_files:
    #                 continue
    #             # Change to AlphaPose dir
    #             os.chdir(args.alphapose_dir)

    #             # Build command line
    #             command = create_command(args.alphapose_dir, video_filename, args.outdir, is_video=args.video)
    #             # Run command
    #             print('\n$', command)
    #             os.system(command)

    #             # Change back to directory containing this script (main_alpahpose.py)
    #             os.chdir(args.outdir)

    #             # Convert alphapose-results.json to *_alphapose_tracked_person.json
    #             read_convert_write(alphapose_orig_results_filename, alphapose_tracked_results_filename)
    #             # Optionally, rename generic filename 'alphapose-results.json' by adding video filename prefix
    #             os.rename("alphapose-results.json", alphapose_results_filename)
    #             shutil.rmtree('poseflow', ignore_errors=True)
    #             os.chdir(curr_dir)


if __name__ == '__main__':
    main()
