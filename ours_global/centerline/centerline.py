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
from scipy.optimize import least_squares
from sklearn.decomposition import PCA



if __name__ == '__main__':
    pkl_name = 's1'
    save = 0

    # 读取前面拼接完成后包含divider和boundary的结果

    data = slb.load_data('pkls/merge_'+ pkl_name+'.pkl')
    global_map = [[],[],[]]
    # import ipdb;ipdb.set_trace()

    # 对divider进行直线拟合，并且统一排列顺序
    for cate in range(0, 1):
        temp_local_old = data[cate]
        for i, vecs in enumerate(temp_local_old):
            for j, vec in enumerate(vecs):
                # 对 vec 进行拟合，并直接更新 data 中的值
                vec = pplb.fit_2d_rotate(vec)
                vec = pplb.reorder_polyline(vec)
                data[cate][i][j] = vec

    # 6, 10
    # 后处理得到中心线
    straight = pplb.get_center_straight(data, 6, 10, save = False, name = 'pkls/center_v2/merge_center_' + pkl_name + '.pkl')

    #保存结果
    if save:
        with open('merge/merge_center_' + pkl_name + '.pkl', 'wb') as file: 
            pickle.dump(data, file)

    # 绘制结果
    range_x, range_y = pplb.get_range(pkl_name)
    # slb.plot_map(straight, 'centers/version2/straight_line/merged_center_'+ pkl_name, range_x, range_y, save = True, fit = True, show= True, is_global=True)
    slb.plot_map(data, 'centers/version2/intersections/merged_center_'+ pkl_name, range_x, range_y, save = False, fit = True, show= True, is_global=True)