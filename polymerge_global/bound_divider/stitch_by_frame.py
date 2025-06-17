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

    pkl_name = 's1'

    for cate in range(3):
        sub_map[cate] += (global_map[cate][0:start])

    for i in range(start,len(global_map[0]) - 1):
        for cate in range(3):
            sub_map[cate].append(global_map[cate][i + 1])
        sub_map = slb.refine_token_v3(sub_map, proximity_th = 0.8)

    return sub_map


if __name__ == '__main__':

    pkl_name = 's1'
    save = 0

    data = slb.load_data('datas/result_keep_'+ pkl_name  + '.pkl')
    global_map = [[], [], []]
    
    for index in range(0,len(data)//3):
        frame = data[3*index]
        message = slb.preprocess_frame(frame)        
        slb.update_global_map(global_map, message, type='pred')

    #这里采用的是逐帧拼接的策略，即按照时间顺序，不断依次合并帧间元素，之前的方法是所有的帧采用二分的方式进行拼接
    merged_map = merge_by_frame_v2(global_map)

    #保存拼接数据
    if save:
        with open('merge/merge_' + pkl_name + '.pkl', 'wb') as file: 
            pickle.dump(merged_map, file)

    #对拼接结果进行绘制
    range_x, range_y = pplb.get_range(pkl_name)
    slb.plot_map(global_map, 'unmerged_'+ pkl_name , range_x, range_y, save = False, fit = False, show = True)


