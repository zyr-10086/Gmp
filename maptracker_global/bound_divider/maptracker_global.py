import sys

from ..map_stitch_libs_global  import Stitch_lib as slb
from ..map_stitch_libs_global  import Maptracker_lib as mlb
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from scipy.interpolate import splprep, splev

class MapTracker_Global:
    def __init__(self):
        self.stitch_data = None
    def global_map_stitch(self,global_data, if_vis=False, pred_save_path=None):
        match_seq, pred_list = mlb.get_scene_matching_result(global_data)
        match_result = mlb.generate_results(match_seq, pred_list)
        merge_result = mlb.vis_pred_data(match_result, if_vis, pred_save_path=None)
        self.stitch_data = merge_result
        return merge_result
