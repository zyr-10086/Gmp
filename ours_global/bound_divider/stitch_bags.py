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
    global_map = [[],[],[]]
    save = 0

    #这里是针对不同包合并好的结果进行拼接，merge_XX指的是每个对应的包合并完成的结果
    for i in range(8):
        if i == 7:
            datas_temp = (slb.load_data('results/pickles/merge_l1.pkl'))
        else:
            datas_temp = (slb.load_data('results/pickles/merge_s' + str(i + 1) + '.pkl'))
        
        for category in range(3):
            global_map[category] += datas_temp[category]

    #对元素进行拟合，并且稠密化
    for category in range(3):
        instance_set = global_map[category]
        for frame in instance_set:
            for i in range(len(frame)):
                vec = frame[i]
                # import ipdb; ipdb.set_trace()

                pts_x = vec[:, 0]  
                pts_y = vec[:, 1]
                if len(pts_x) >3:
                    tck, u = splprep([pts_x, pts_y], s=1)
                    x_smooth, y_smooth = splev(np.linspace(0, 1, 50), tck)
                    new_vec = np.column_stack((x_smooth, y_smooth))
                    frame[i] = new_vec

    #采用类似策略进行拼接
    refined_global_map = slb.refine_token_v3(global_map, proximity_th = 2 , save = True)

    #保存结果
    if save:
        with open('merge/merge_global.pkl', 'wb') as file: 
            pickle.dump(refined_global_map, file)

    #绘制结果
    # slb.plot_map(global_map, 'centers/unmerged_gg', save = True, fit = True)
    slb.plot_map(refined_global_map, 'merged_global', save = True, fit = True)