import sys
sys.path.append('..')
from map_stitch_libs import post_process_lib as pplb
from map_stitch_libs import Stitch_lib as slb
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev
from progress.bar import Bar
import time

def merge_by_frame(global_map):
    step = 10
    if step > len(global_map[0]):
        return slb.refine_token_v3(global_map, proximity_th = 0.8)
    temp_map = [[],[],[]]
    for i in range(0,len(global_map[0]),step):
        subset = [[],[],[]]
        for cate in range(3):
            subset[cate] = global_map[cate][i:i+step]
        # import ipdb;ipdb.set_trace()

        subset_merge = slb.refine_token_v3(subset, proximity_th = 0.8)
        for cate in range(3):
            temp_map[cate] = temp_map[cate] + subset_merge[cate]
    return merge_by_frame(temp_map)

def pre_fit(data):
    for cate in range(0, 1):
        temp_local_old = data[cate]
        for i, vecs in enumerate(temp_local_old):
            for j, vec in enumerate(vecs):
                # 对 vec 进行拟合，并直接更新 data 中的值
                vec = pplb.fit_2d_rotate(vec)
                vec = pplb.reorder_polyline(vec)
                data[cate][i][j] = vec

def merge_by_frame_v2(global_map):
    sub_map = [[],[],[]]
    start = 5
    bar = Bar('merge', max=(len(global_map[0]) - start - 1))
    start_time = time.time()
    for cate in range(3):
        
        sub_map[cate] += (global_map[cate][0:start])

    for i in range(start,len(global_map[0]) - 1):
        for cate in range(3):
            sub_map[cate].append(global_map[cate][i + 1])
        sub_map = slb.refine_token_v3(sub_map, proximity_th = 0.8)
        bar.next()
        elapsed_time = time.time() - start_time
        bar.message = f'merge: {elapsed_time:.2f} s'
    bar.finish()

    return sub_map


if __name__ == '__main__':

    pkl_name = 's1'
    save = 0
    data_path = '../../../local_data/nuscenes/result_keep_scene-0077.pkl'
    data_path = '../../../local_data/nuscenes/result_keep_all_nuscenes.pkl'
    data = slb.load_data(data_path)
    global_map = [[], [], []]
    
    car_trajectory = []

    # data = data[:10]
    skip = 1
    add_last = False
    for index in range(0,len(data['scene-0279']['local']) // skip):
        if index == len(data['scene-0279']['local']) // skip and index * skip != len(data['scene-0279']['local']):
            add_last = True
        frame = data['scene-0279']['local'][index * skip]
        loc = frame['gt']['pose']
        yaw = frame['gt']['yaw'] - np.pi / 2
        car_trajectory.append([np.array(loc), yaw * 180 / np.pi])
        message = slb.preprocess_frame(frame)        
        slb.update_global_map(global_map, message, type='pred')

        if add_last:
            frame = data['scene-0279']['local'][-1]
            loc = frame['gt']['pose']
            yaw = frame['gt']['yaw'] - np.pi / 2
            car_trajectory.append([np.array(loc), yaw * 180 / np.pi])
            message = slb.preprocess_frame(frame)
    
    # global_map[2] = []
    #这里采用的是逐帧拼接的策略，即按照时间顺序，不断依次合并帧间元素，之前的方法是所有的帧采用二分的方式进行拼接
    merged_map = merge_by_frame_v2(global_map)

    # #保存拼接数据
    # if save:
    #     with open('merge/merge_' + pkl_name + '.pkl', 'wb') as file: 
    #         pickle.dump(merged_map, file)

    # #对拼接结果进行绘制
    # range_x, range_y = pplb.get_range(pkl_name)
    # slb.plot_map(global_map, 'unmerged_'+ pkl_name , range_x, range_y, save = False, fit = False, show = True)

    all_points = []

    for catogory in merged_map:
        for frame in catogory:
            for vec in frame:
                all_points.append(vec)
    all_points = np.concatenate(all_points, axis=0)
    x_min = all_points[:,0].min()
    x_max = all_points[:,0].max()
    y_min = all_points[:,1].min()
    y_max = all_points[:,1].max()


    pred_save_path = 'merge_global_polymerge.png'
    slb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, merged_map)



