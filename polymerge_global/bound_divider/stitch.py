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

    pkl_name = 's1'
    save = 0

    data = slb.load_data('../../datas/result_keep_'+ pkl_name  + '.pkl')
    global_map = [[], [], []]
    
    #读取预测结果，并转换到全局坐标系
    for index in range(0,len(data)//3):
        frame = data[3*index]
        message = slb.preprocess_frame(frame)        
        slb.update_global_map(global_map, message, type='pred')
    
    global_map[2] = []
    #将数据进行合并、拼接（这里针对boundary和divider）
    refined_global_map = slb.refine_token_v2(global_map, proximity_th = 0.8, save = False, name = 'pkls/match_'+ pkl_name + '.pkl')

    #保存拼接数据
    if save:
        with open('merge/merge_' + pkl_name + '.pkl', 'wb') as file: 
            pickle.dump(refined_global_map, file)
            
    #对拼接结果进行绘制
    range_x, range_y = pplb.get_range(pkl_name)
    slb.plot_map(refined_global_map, 'unmerged_'+ pkl_name , range_x, range_y, save = False, fit = False, show = True)
    # slb.plot_map(global_map, 'merged_'+ pkl_name, range_x, range_y, save = True, fit = False)