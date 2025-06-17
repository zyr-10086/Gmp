import sys
import os

import argparse     

import matplotlib.transforms as transforms

import cv2
import torch
import numpy as np
from PIL import Image
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points
from scipy.spatial import ConvexHull
from PIL import Image
import cv2
import imageio
import math
from .cmap_utils.utils import *
from .cmap_utils.match_utils import *
from .cmap_utils.merge_utils import *

cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}

def collect_pred(data,thr):
    vectors = {label: [] for label in cat2id.values()}
    scores = {label: [] for label in cat2id.values()}
    for i in range(len(data['labels'])):
        score, label, v = data['scores'][i], data['labels'][i], data['vectors'][i]
        if score > thr:
            vectors[label].append(np.array(v))
            scores[label].append(score)
    return vectors, scores



def match_two_consecutive_frames_pred(prev_data,  curr_data, roi_size, origin):

    #获得到当前帧的变换矩阵
    prev2curr_matrix ,prev2glb_matrix  = get_prev2curr_matrix(prev_data,curr_data)

    #根据预测的置信度滤除
    prev_vectors = filter_vectors(prev_data,origin,roi_size)
    curr_vectors = filter_vectors(curr_data,origin,roi_size)
    # import ipdb;ipdb.set_trace()
    #统一变换到当前帧
    prev_vectors, curr_vectors, prev2curr_vectors = get_consecutive_vectors(prev_vectors,curr_vectors,
                                    prev2curr_matrix,origin,roi_size) 

    #为前一帧和当前帧的所有vectors画栅格化mask，存储到一个字典中；然后所有的栅格化的结果绘制到一个canvas返回（没用）

    prev2curr_masks, prev2curr_viz = draw_instance_masks(prev2curr_vectors, roi_size, origin)
    curr_masks, curr_viz = draw_instance_masks(curr_vectors, roi_size, origin)
    # import ipdb;ipdb.set_trace()


    #最小化成本矩阵的方式进行匹配分配（先最小化成本矩阵，然后再用阈值限制）
    prev2curr_matchings = find_matchings_iou(prev2curr_masks, curr_masks, thresh=0.001)
    curr2prev_matchings = {label:[match_info[1],match_info[0]]  for label,match_info in prev2curr_matchings.items()}
    return curr2prev_matchings

def get_scene_matching_result(data_container):
    ### obtain local id sequence matching results of predictions
    vectors_seq = []
    scores_seq = []

    roi_size = (30, 60) # bev range, 60m in x-axis, 30m in y-axis
    pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]
    roi_size = torch.tensor(roi_size).numpy()
    origin = torch.tensor(pc_range[:2]).numpy()

    ids_seq = []
    global_map_index = {
        0: 0,
        1: 0,
        2: 0,
    }
    frame_token_list = []
    pred_data_list = []
    meta_list = []

    # for idx in scene_name2idx:
    #     token = dataset[idx]['img_metas'].data['token']
    #     pred_data = pred_results[token]
    #     frame_token_list.append(token)
    #     meta_list.append(dataset[idx]['img_metas'].data)
    #     pred_data_list.append(pred_data)

    #自定义数据
    # data_container = []
    pred_data_list = []
    for data in data_container:
        pred = {}
        vectors = data['pred']['box']
        labels = data['pred']['label']
        vec_dic = {0:[],1:[],2:[]}
        for i in range(len(vectors)):
            lb = 0
            if labels[i] == 0:
                lb = 1
            elif labels[i] == 1:
                lb = 0
            elif labels[i] == 2:
                lb = 2
            vec_dic[lb].append(vectors[i])

        # pred['vectors'] = data['pred']['box']
        # pred['labels'] = data['pred']['label']

        pred['vectors'] = vec_dic
        pred['pose'] = data['gt']['pose']
        pred['yaw'] = data['gt']['yaw']
        pred_data_list.append(pred)


    for local_idx in range(len(pred_data_list)):
        curr_pred_data = pred_data_list[local_idx]
        # print(local_idx)
        # vectors_info, scores = collect_pred(curr_pred_data,args.thr)
        # vectors_seq.append(vectors_info)
        # scores_seq.append(scores)

        ### assign global id for the first frame
        if local_idx == 0:
            ids_0 = dict()
            for lb in range(3):
                ids_0[lb] = dict()

            # for idx in  range(len(curr_pred_data['labels'])):
            #     label = curr_pred_data['labels'][idx]
            #     glb_idx = global_map_index[label]
            #     global_map_index[label] += 1
            #     tmp_dict = {idx:glb_idx}

            #     ids_0[label].update(tmp_dict)

            for label, vectors in  curr_pred_data['vectors'].items():

                id_mapping = dict()
                for i, _ in enumerate(vectors):
                    id_mapping[i] = global_map_index[label]
                    global_map_index[label] += 1
                ids_0[label] = id_mapping

            ids_seq.append(ids_0)
            # import ipdb;ipdb.set_trace()
            continue
        
        #过去时刻的帧数
        cons_frames = 5

        ### from the farthest to the nearest
        history_range = range(max(local_idx-cons_frames,0),local_idx)
        tmp_ids_list = []

        for comeback_idx,prev_idx in enumerate(history_range):

            tmp_ids = {label:{} for label in cat2id.values()} 
            curr_pred_data = pred_data_list[local_idx]
            comeback_pred_data = pred_data_list[prev_idx]

            # import ipdb;ipdb.set_trace()

            # curr_meta = meta_list[local_idx]
            # comeback_meta = meta_list[prev_idx]

            #在这里根据前一帧的匹配结果，然后分配当前帧的id，如果匹配上了，就给上一个id，否则就分配一个新的id
            curr2prev_matching = match_two_consecutive_frames_pred(comeback_pred_data, curr_pred_data, roi_size, origin)
            # import ipdb;ipdb.set_trace()
            for label,match_info in curr2prev_matching.items():
                for curr_match_local_idx,prev_match_local_idx in enumerate(match_info[0]):
                    if prev_match_local_idx == -1:
                        tmp_ids[label][curr_match_local_idx] = -1
                    else:
                        prev_match_global_idx = ids_seq[prev_idx][label][prev_match_local_idx]
                        tmp_ids[label][curr_match_local_idx] = prev_match_global_idx

            tmp_ids_list.append(tmp_ids)

        ids_n = {label:{} for label in cat2id.values()}

        ### assign global id based on previous k frames' global id
        missing_matchings = {label:[] for label in cat2id.values()}
        for tmp_match in tmp_ids_list[::-1]:
            for label, matching in tmp_match.items():
                for vec_local_idx, vec_glb_idx in matching.items():
                    if vec_local_idx not in ids_n[label].keys():
                        if vec_glb_idx != -1 and vec_glb_idx not in ids_n[label].values():
                            ids_n[label][vec_local_idx] = vec_glb_idx
                            if vec_local_idx in missing_matchings[label]:
                                missing_matchings[label].remove(vec_local_idx)
                        else:
                            missing_matchings[label].append(vec_local_idx)

        ### assign new id if one vector is not matched 
        for label,miss_match in missing_matchings.items():
            for miss_idx in miss_match:
                if miss_idx not in ids_n[label].keys():
                    ids_n[label][miss_idx] = global_map_index[label]
                    global_map_index[label] += 1
        ids_seq.append(ids_n)

    return ids_seq, pred_data_list
def generate_results(ids_info,pred_data_list):

    ### assign global id 

    global_gt_idx = {}
    result_list = []
    instance_count = 0
    for f_idx in range(len(ids_info)):
        output_dict = {'vectors':[],'global_ids':[],'labels':[],'scores':[],'local_idx':[]}
        output_dict['pose'] = pred_data_list[f_idx]['pose']
        output_dict['yaw'] = pred_data_list[f_idx]['yaw']

        for label in cat2id.values():
            for local_idx, global_label_idx in ids_info[f_idx][label].items():
                overall_count_idx = label*100 + global_label_idx
                if overall_count_idx not in global_gt_idx.keys():
                    overall_global_idx = instance_count
                    global_gt_idx[overall_count_idx] = overall_global_idx
                    instance_count += 1
                else:
                    overall_global_idx = global_gt_idx[overall_count_idx]
                output_dict['global_ids'].append(overall_global_idx)
                output_dict['vectors'].append(pred_data_list[f_idx]['vectors'][label][local_idx])
                # output_dict['labels'].append(pred_data_list[f_idx]['labels'][local_idx])
                output_dict['labels'].append(label)
        output_dict['local_idx'] = f_idx

        result_list.append(output_dict)
    return result_list

def vis_pred_data(pred_results=None, if_vis=False, pred_save_path=None):

    
    roi_size = (60, 30) # bev range, 60m in x-axis, 30m in y-axis
    pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]
    roi_size = torch.tensor(roi_size).numpy()
    origin = torch.tensor(pc_range[:2]).numpy()
    
    # get the item index of the scene
    index_list = []
    for index in range(len(pred_results)):
        # if pred_results[index]["scene_name"] == scene_name:
        index_list.append(index)
    
    car_trajectory = []
    id_prev2curr_pred_vectors = defaultdict(list)
    id_prev2curr_pred_frame_info = defaultdict(list)
    id_prev2curr_pred_frame = defaultdict(list)

    # iterate through each frame
    last_index = index_list[-1]
    for index in index_list:
        # import ipdb;ipdb.set_trace()

        # vectors = np.array(pred_results[index]["vectors"]).reshape((len(np.array(pred_results[index]["vectors"])), 20, 2))
        vectors = np.array(pred_results[index]["vectors"])
        # if abs(vectors.max()) <= 1:
        #     curr_vectors = vectors * roi_size + origin
        # else:
        curr_vectors = vectors
            
        # get the transformation matrix of the last frame

        #逐帧到最后一帧位姿对齐（得到车移动位姿，这里要改）
        prev2curr_matrix, prev2glb_matrix = get_prev2curr_matrix(pred_results[index],pred_results[last_index])
        prev2curr_pred_vectors = get_prev2curr_vectors(curr_vectors, prev2glb_matrix,origin,roi_size,False,False)
        # prev2curr_pred_vectors = prev2curr_pred_vectors * roi_size + origin
        
        rotation_degrees = np.degrees(np.arctan2(prev2curr_matrix[:3, :3][1, 0], prev2curr_matrix[:3, :3][0, 0]))
        car_center = get_prev2curr_vectors(np.array((0,0)).reshape(1,1,2), prev2curr_matrix,origin,roi_size,False,False)* roi_size + origin
        
        # car_trajectory.append([car_center.squeeze(), rotation_degrees])
        car_trajectory.append([np.array(pred_results[index]['pose']), (pred_results[index]['yaw'] - np.pi/2) * 180 / np.pi])

        
        for i, (label, vec_glb_idx) in enumerate(zip(pred_results[index]['labels'], pred_results[index]['global_ids'])):
            dict_key = "{}_{}".format(label, vec_glb_idx)
            id_prev2curr_pred_vectors[dict_key].append(prev2curr_pred_vectors[i])
            id_prev2curr_pred_frame_info[dict_key].append([pred_results[index]["local_idx"], len(id_prev2curr_pred_frame[dict_key])])

        for key, frame_info in id_prev2curr_pred_frame_info.items():
            frame_localIdx = dict()
            for frame_time, local_index in frame_info:
                frame_localIdx[frame_time] = local_index
            id_prev2curr_pred_frame[key] = frame_localIdx
        
    
    # sort the id_prev2curr_pred_vectors
    id_prev2curr_pred_vectors = {key: id_prev2curr_pred_vectors[key] for key in sorted(id_prev2curr_pred_vectors)}

    return plot_fig_merged(car_trajectory, id_prev2curr_pred_vectors, if_vis, pred_save_path)

def plot_fig_merged(car_trajectory, id_prev2curr_pred_vectors, if_vis, pred_save_path =  'pred_merge_result.png'):
    

    simplify = 0.5
    line_opacity = 0.75

    pred_maps = merge_vectors(id_prev2curr_pred_vectors)

    if if_vis:
        roi_size = (60, 30)
        # set the size of the image
        x_min = -roi_size[0] / 2
        x_max = roi_size[0] / 2
        y_min = -roi_size[1] / 2
        y_max = roi_size[1] / 2

        all_points = []
        for vecs in id_prev2curr_pred_vectors.values():
            points = np.concatenate(vecs, axis=0)
            all_points.append(points)
        all_points = np.concatenate(all_points, axis=0)

        x_min = all_points[:,0].min()
        x_max = all_points[:,0].max()
        y_min = all_points[:,1].min()
        y_max = all_points[:,1].max()

        fig = plt.figure(figsize=(int(x_max - x_min) + 10 , int(y_max - y_min) + 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        car_img = Image.open('resources/car-orange.png')
        
        faded_rate = np.linspace(0.2, 1, num=len(car_trajectory))

        # trace the path reversely, get the sub-sampled traj for visualizing the car
        pre_center = car_trajectory[-1][0]
        selected_traj = []
        selected_timesteps = []
        for timestep, (car_center, rotation_degrees) in enumerate(car_trajectory[::-1]):
            if np.linalg.norm(car_center - pre_center) < 5 and timestep > 0 and timestep < len(car_trajectory)-1:
                continue
            selected_traj.append([car_center, rotation_degrees])
            selected_timesteps.append(len(car_trajectory)-1-timestep)
            pre_center = car_center
        selected_traj = selected_traj[::-1]
        selected_timesteps = selected_timesteps[::-1]

        for selected_t, (car_center, rotation_degrees) in zip(selected_timesteps, selected_traj):
            translation = transforms.Affine2D().translate(car_center[0], car_center[1])
            rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
            rotation_translation = rotation + translation
            ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation+ ax.transData, 
                    alpha=faded_rate[selected_t])

        for category , vecs in pred_maps.items():

            if category == 'ped_crossing': # ped_crossing
                color = 'b'
            elif category == 'divider': # divider
                color = 'orange'
            elif category == 'boundary': # boundary
                color = 'r'

            for vec in vecs:
                pts = vec[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)

        transparent = False
        dpi = 20
        plt.grid(False)
        plt.savefig(pred_save_path, bbox_inches='tight', transparent=transparent, dpi=dpi)
        plt.clf() 
        plt.close(fig)
        print("image saved to : ", pred_save_path)
    
    return pred_maps


def merge_vectors(id_prev2curr_pred_vectors):

    simplify = 0.5
    line_opacity = 0.75
    pred_datas = {'divider': [], 'ped_crossing': [], 'boundary': []}

    for tag, vecs in id_prev2curr_pred_vectors.items():

        label, vec_glb_idx = tag.split('_')
        label = int(label)
        vec_glb_idx = int(vec_glb_idx)

    
        # get the vectors belongs to the same instance
        polylines = []
        for vec in vecs:
            polylines.append(LineString(vec))
        if len(polylines) <= 0:
            continue
        # print(tag, vec_num)

        if label == 0: # crossing, merged by convex hull
            polygon = merge_corssing(polylines)
            if polygon.area < 2:
                continue
            polygon = polygon.simplify(simplify)
            vector = np.array(polygon.exterior.coords) 
            pred_datas['ped_crossing'].append(vector)
            
        elif label == 1: # divider, merged by interpolation
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            polylines_vecs = merge_divider(polylines_vecs)
            for one_line in polylines_vecs:
                one_line = np.array(LineString(one_line).simplify(simplify).coords)
                pred_datas['divider'].append(one_line)
                
        elif label == 2: # boundary, merged by interpolation
            polylines_vecs = [np.array(one_line.coords) for one_line in polylines]
            # if tag == '2_1494':
            # import ipdb;ipdb.set_trace()
            polylines_vecs = merge_boundary(polylines_vecs,tag)
            for one_line in polylines_vecs:
                one_line = np.array(LineString(one_line).simplify(simplify).coords)
                pred_datas['boundary'].append(one_line)
                
    return pred_datas            