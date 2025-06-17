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

if __name__ == '__main__':


    data_path = '/home/zyr/globalmap/Merge_Libs/local_data/datas_v2/result_keep_s1.pkl'
    save = 0

    data = slb.load_data(data_path)

    global_map = [[], [], []]

    global_map_2 = [[], [], []]
    
    #读取预测结果，并转换到全局坐标系
    car_trajectory = []
    # data = data[:10]
    # import ipdb;ipdb.set_trace()

    vis_num = 3
    for index in range(0,len(data)):
        # if index < vis_num:
        #     continue
        
        frame = data[index]
        loc = frame['gt']['pose']
        yaw = frame['gt']['yaw'] - np.pi / 2
        car_trajectory.append([np.array(loc), yaw * 180 / np.pi])
        message = slb.preprocess_frame(frame)        
        slb.update_global_map(global_map, message, type='pred')
    # car_trajectory = []

    # slb.update_global_map_all(global_map_2, data['global'])
    global_map[2] = global_map[1]
    global_map[1] = []

    # del global_map[2][0][2]
    # del global_map[2][0][0]
    
    #将数据进行合并、拼接（这里针对boundary和divider）
    # refined_global_map = slb.refine_token_v2(global_map, proximity_th = 10, save = False, name = 'pkls/match_.pkl')
    # refined_global_map = slb.refine_token_v2(global_map, proximity_th = 0.5, save = False, name = 'match_ss.pkl')
    # refined_global_map = slb.refine_token_v2(global_map, proximity_th = 0.8, pre =True, denoise = True)
    refined_global_map = global_map

    # for i in range(0, len(refined_global_map)):
    #     refined_global_map[i].append(global_map_2[i][0])

    all_points = []

    for catogory in refined_global_map:
        for frame in catogory:
            for vec in frame:
                all_points.append(vec)
    # import ipdb;ipdb.set_trace()

    all_points = np.concatenate(all_points, axis=0)
    x_min = all_points[:,0].min()
    x_max = all_points[:,0].max()
    y_min = all_points[:,1].min()
    y_max = all_points[:,1].max()


    pred_save_path = 'pred_merge_global_polymerge_scene.png'
    slb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, refined_global_map)