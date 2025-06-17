import sys
sys.path.append('..')
from map_stitch_libs import post_process_lib as pplb
from map_stitch_libs import Stitch_lib as slb
from map_stitch_libs import PolyMergelib as plb
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev

if __name__ == '__main__':

    pkl_name = 's1'
    save = 0
    data_path = '../../../local_data/nuscenes/result_keep_scene-0077.pkl'

    data = slb.load_data(data_path)

    global_map = [[], [], []]
    
    #读取预测结果，并转换到全局坐标系
    car_trajectory = []
    # data = data[:10]
    for index in range(0,len(data['local'])//3):
        frame = data['local'][3*index]
        loc = frame['gt']['pose']
        yaw = frame['gt']['yaw'] - np.pi / 2
        car_trajectory.append([np.array(loc), yaw * 180 / np.pi])
        message = slb.preprocess_frame(frame)        
        slb.update_global_map(global_map, message, type='pred')
    
    # global_map[2] = []
    #将数据进行合并、拼接（这里针对boundary和divider）
    refined_global_map = plb.refine_token_v2(global_map, proximity_th = 0.8)
    # import ipdb; ipdb.set_trace()
    #保存拼接数据
    # if save:
    #     with open('merge/merge_' + pkl_name + '.pkl', 'wb') as file: 
    #         pickle.dump(refined_global_map, file)

    all_points = []
    import ipdb; ipdb.set_trace()

    for catogory in refined_global_map:
        for frame in catogory:
            for vec in frame:
                all_points.append(vec)
    all_points = np.concatenate(all_points, axis=0)
    x_min = all_points[:,0].min()
    x_max = all_points[:,0].max()
    y_min = all_points[:,1].min()
    y_max = all_points[:,1].max()
    

    pred_save_path = 'merge_global_polymerge.png'
    plb.plot_fig_merged(car_trajectory, x_min, x_max, y_min, y_max, pred_save_path, refined_global_map)