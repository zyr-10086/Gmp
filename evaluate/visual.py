from progress.bar import Bar
from shapely.geometry import LineString
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
sys.path.append('..')
from ours_global.bound_divider import ours_global as OURS_GLOBAL


categories = ['divider', 'ped_crossing', 'boundary']
import pickle as pkl
from PIL import Image

def vis_per_method(dataset_name, eval_data_per_method, method_name):

    simplify = 0.5
    line_opacity = 0.75

    all_points = []

    all_pred_data_path = '../local_data/nuscenes/result_keep_all_' + dataset_name +'_glb.pkl'
    with open(all_pred_data_path, 'rb') as f:
        all_pred_data = pkl.load(f)
    method = OURS_GLOBAL.Ours_Global()

    savec_path = 'save_' + dataset_name + '/'
    if not os.path.exists(savec_path):
            os.makedirs(savec_path)


    if not os.path.exists(savec_path + method_name):
        # 如果文件夹不存在，则创建
        os.makedirs(savec_path + method_name)
    
    for scene_name, eval_data in eval_data_per_method.items():

        pred_save_path = savec_path + method_name + '/vis_result_' + scene_name + '.png'

        pred_global_maps = eval_data['pred']
        pred_data = all_pred_data[scene_name]['local']
        # import ipdb; ipdb.set_trace()
        global_data, car_trajectory = method.preprocess_data(pred_data)

        all_points = []
        for catogory in categories:
            pred_map_vectors = pred_global_maps[catogory]
            for vec in pred_map_vectors:
                all_points.append(vec)

        all_points = np.concatenate(all_points, axis=0)
        x_min = all_points[:,0].min()
        x_max = all_points[:,0].max()
        y_min = all_points[:,1].min()
        y_max = all_points[:,1].max()

        fig = plt.figure(figsize=(int(x_max - x_min) + 10 , int(y_max - y_min) + 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if len(car_trajectory) != 0:
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
            import matplotlib.transforms as transforms

            for selected_t, (car_center, rotation_degrees) in zip(selected_timesteps, selected_traj):
                translation = transforms.Affine2D().translate(car_center[0], car_center[1])
                rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
                rotation_translation = rotation + translation
                ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation+ ax.transData, 
                        alpha=faded_rate[selected_t])


        for label in categories:
            if label == 'ped_crossing': # ped_crossing
                color = 'b'
            elif label == 'divider': # divider
                color = 'orange'
            elif label == 'boundary': # boundary
                color = 'r'

            pred_map_vectors = pred_global_maps[label]
            # gt_map_vectors = gt_global_maps[label]

            for pred_map_vector in pred_map_vectors:
                polyline = LineString(pred_map_vector)
                polyline = np.array(polyline.simplify(simplify).coords)
                pts = polyline[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)

        transparent = False
        dpi = 20
        plt.grid(False)
        plt.axis('off') 
        plt.savefig(pred_save_path, bbox_inches='tight', transparent=transparent, dpi=dpi)
        plt.clf() 
        plt.close(fig)
        print("image saved to : ", pred_save_path)

def vis_gt(dataset_name, eval_data_per_method, method_name):

    simplify = 0.5
    line_opacity = 0.75

    all_points = []

    all_pred_data_path = '../local_data/nuscenes/result_keep_all_' + dataset_name +'_glb.pkl'
    with open(all_pred_data_path, 'rb') as f:
        all_pred_data = pkl.load(f)
    method = OURS_GLOBAL.Ours_Global()

    store_gt = 'save_' + dataset_name +'/gts'

    if not os.path.exists(store_gt):
        # 如果文件夹不存在，则创建
        os.makedirs(store_gt)
    
    for scene_name, eval_data in eval_data_per_method.items():

        pred_save_path = store_gt + '/vis_result_gt_' + scene_name + '.png'

        gt_global_maps = eval_data['gt']
        pred_data = all_pred_data[scene_name]['local']
        global_data, car_trajectory = method.preprocess_data(pred_data)
        # import ipdb; ipdb.set_trace()

        all_points = []
        for catogory in categories:
            gt_map_vectors = gt_global_maps[catogory]
            for vec in gt_map_vectors:
                all_points.append(vec)

        all_points = np.concatenate(all_points, axis=0)
        x_min = all_points[:,0].min()
        x_max = all_points[:,0].max()
        y_min = all_points[:,1].min()
        y_max = all_points[:,1].max()

        fig = plt.figure(figsize=(int(x_max - x_min) + 10 , int(y_max - y_min) + 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if len(car_trajectory) != 0:
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
            import matplotlib.transforms as transforms

            for selected_t, (car_center, rotation_degrees) in zip(selected_timesteps, selected_traj):
                translation = transforms.Affine2D().translate(car_center[0], car_center[1])
                rotation = transforms.Affine2D().rotate_deg(rotation_degrees)
                rotation_translation = rotation + translation
                ax.imshow(car_img, extent=[-2.2, 2.2, -2, 2], transform=rotation_translation+ ax.transData, 
                        alpha=faded_rate[selected_t])

        for label in categories:
            if label == 'ped_crossing': # ped_crossing
                color = 'b'
            elif label == 'divider': # divider
                color = 'orange'
            elif label == 'boundary': # boundary
                color = 'r'

            gt_map_vectors = gt_global_maps[label]
            # gt_map_vectors = gt_global_maps[label]

            for gt_map_vector in gt_map_vectors:
                polyline = LineString(gt_map_vector)
                polyline = np.array(polyline.simplify(simplify).coords)
                pts = polyline[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=20, markersize=50, alpha=line_opacity)
                ax.plot(x, y, "o", color=color, markersize=50)

        transparent = False
        dpi = 20
        plt.grid(False)
        plt.axis('off') 
        plt.savefig(pred_save_path, bbox_inches='tight', transparent=transparent, dpi=dpi)
        plt.clf() 
        plt.close(fig)
        print("image saved to : ", pred_save_path)