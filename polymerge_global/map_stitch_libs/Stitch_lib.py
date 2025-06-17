import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import directed_hausdorff
import networkx as nx
from tqdm import tqdm
import random
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

def load_data(data_path = 'result_keep_v1.pkl'):
    with open(data_path, 'rb') as f:
        return pickle.load(f)
    
def preprocess_frame(frame):
    position = frame['gt']['pose']
    yaw = frame['gt']['yaw']
    gt_box = frame['gt']['box']
    gt_label = frame['gt']['label']
    pred_box = frame['pred']['box']
    pred_label = frame['pred']['label']
    message = {'position': position, 'yaw': yaw, 'gt_box': gt_box, 'gt_label': gt_label, 'pred_box': pred_box, 'pred_label': pred_label}
    return message

def transform_coords(coords, position, yaw):
    rotation_matrix = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return np.dot(coords, rotation_matrix.T) + position

def update_global_map(global_map, message, type = "pred"):
    frame_map = [[], [], []]
    if type == "pred":
        box = message['pred_box']
        label = message['pred_label']
    else:
        box = message['gt_box']
        label = message['gt_label']

    for p_box, p_label in zip(box, label):
        global_coords = transform_coords(p_box, message['position'], message['yaw'] -1.57)
        # import ipdb;ipdb.set_trace()
        frame_map[p_label].append(global_coords)

    for i in range(len(global_map)):
        global_map[i].append(frame_map[i])

def plot_map(map_data, name, range_x = [-2300, -2200], range_y = [-580, -510] , save = False, fit = False, show = False, is_global = False):


    fig, ax = plt.subplots()
    # plt.figure(figsize=(2,2))
    # -2228, -2276
    # -528, -554
    plt.xlim(range_x[0], range_x[1])
    plt.ylim(range_y[0], range_y[1])
    plt.axis('off')

    colors_plt = ['orange',  'r', 'g']
    label_plt = ['divider',  'boundary', 'centerline']

    linewidth = 1.0
    dpi = 600
    mutation_scale=5
    freq = 2
    if (is_global):
        linewidth = 0.5
        dpi = 3600
        mutation_scale=1
        freq = 8

    for idx, local_map in enumerate(map_data):
        for vecs in local_map:
            for vec in vecs:
                # import ipdb;ipdb.set_trace()
                pts_x = vec[:, 0]  
                pts_y = vec[:, 1]
                if fit and len(pts_x) > 3:
                    tck, u = splprep([pts_x, pts_y], s=0.1)
                    x_smooth, y_smooth = splev(np.linspace(0, 1, 50), tck)
                    ax.plot(x_smooth, y_smooth, color=colors_plt[idx], label=label_plt[idx], linewidth=linewidth)
                else:
                    ax.plot(pts_x, pts_y, color=colors_plt[idx], label=label_plt[idx],linewidth=linewidth)
                    # plt.scatter(pts_x, pts_y, c=colors_plt[idx], s=0.01)
                import matplotlib.patches as patches
                
                if(idx == 2):
                    if len(pts_x) > 3:
                        freq = 3
                    else:
                        freq = 1
                    for k in range(len(pts_x)//freq -1) :
                        i = k*freq
                        mid_x = (pts_x[i] + pts_x[i+1]) / 2
                        mid_y = (pts_y[i] + pts_y[i+1]) / 2
                        mid_x_1 = (mid_x + pts_x[i])/2
                        mid_y_1 = (mid_y + pts_y[i])/2
                        mid_x_2 = (mid_x + pts_x[i+1])/2
                        mid_y_2 = (mid_y + pts_y[i+1])/2

                        # import ipdb;ipdb.set_trace()
                        arrow = patches.FancyArrowPatch((mid_x_1, mid_y_1), (mid_x_2, mid_y_2),
                                        mutation_scale=mutation_scale, arrowstyle='-|>', color=colors_plt[idx])
                        ax.add_patch(arrow)


    ax.set_aspect('equal', 'box')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys())

    if(save):
        plt.savefig(name + '.png', bbox_inches='tight', format='png', dpi=dpi)
    if show:
        plt.show()
    plt.close()

def vector_proj_in(p1, p2, p3): 
    """
    Computes the vector projection of point p1 onto the line segment defined by points p2 and p3.
    
    和vector_proj类似,但是限制了投影点一定在被投影线段内

    Parameters:
    - p1: A numpy array representing the coordinates of point p1.
    - p2: A numpy array representing the coordinates of the first endpoint of the line segment.
    - p3: A numpy array representing the coordinates of the second endpoint of the line segment.

    Returns:
    - p4_in: A numpy array representing the coordinates of the closest point on the line segment (p2-p3) to point p1.

    Note:
    - If point p1 coincides with either point p2 or p3, point p1 is returned.
    - If the projection point is outside the line segment, it's clamped to the closest endpoint.
    - Adapted from: https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
    """
    if all(p1 == p2) or all(p1 == p3):
        return p1
    else:
        p2p1 = p1 - p2
        p2p3 = p3 - p2
        t = np.dot(p2p1, p2p3) / np.dot(p2p3, p2p3)
        # if you need the the closest point belonging to the segment
        t_in = max(0, min(1, t))
        p4_in = p2 + t_in * p2p3
        return p4_in 

def is_monotonic(lst):
    # 检查是否单调递增
    increasing = all(x <= y for x, y in zip(lst, lst[1:]))
    # 检查是否单调递减
    decreasing = all(x >= y for x, y in zip(lst, lst[1:]))
    return increasing or decreasing

def get_p2p3p4_in(p, poly):
    """
    Gets the closest line segment of a given polyline to point p.

    Inputs:
    - p: A numpy array representing the coordinates of point p.
    - poly: NumPy array representing the vertices of the polyline.
    
    Outputs:
    - p2_in: A numpy array representing the coordinates of the first endpoint of the closest line segment to point p.
    - p3_in: A numpy array representing the coordinates of the second endpoint of the closest line segment to point p.
    - p4_in: A numpy array representing the coordinates of the closest point on the closest line segment to point p.

    Note:
    - Returns the closest projection point (p4_in) and the line segment endpoints (p2_in, p3_in).
    """
    min_dist_in = np.inf
    all_dist_in = []
    # Iterate through each line segment defined by consecutive pairs of points in the polyline
    for p2_temp, p3_temp in zip(poly[:-1], poly[1:]):
        # Calculate projection points using vector_proj_in function
        p4_in_temp = vector_proj_in(p, p2_temp, p3_temp)
        # Calculate distance between point p and in-line projection point
        dist_in = math.dist(p, p4_in_temp)
        # Update minimum distance and corresponding points if necessary
        # all_dist_in.append(dist_in)
        if dist_in < min_dist_in:
            min_dist_in = dist_in
            p2_in = p2_temp
            p3_in = p3_temp
            p4_in = p4_in_temp
    monotonic = is_monotonic(all_dist_in)
    if min_dist_in >= 1 :
        if all(p2_in == poly[0]) or all(p3_in == poly[0]):
            return p2_in, p3_in, poly[0]
        elif all(p2_in == poly[-1]) or all(p3_in == poly[-1]):
            return p2_in, p3_in, poly[-1]

    return p2_in, p3_in, p4_in



def point_to_polyline_dist(p1, poly):
    """
    Determines the distance between point p1 and the given polyline.

    计算点p1到给定的polyline的距离。

    Inputs:
    - p1: A numpy array representing the coordinates of point p1.
    - poly: NumPy array representing the vertices of the polyline.

    Outputs:
    - Distance between p1 and its projection on the polyline.

    Note:
    - Uses the get_p2p3p4_in function to find the projection point.
    """
    # Get the projection point (p4) on the polyline and the corresponding line segment endpoints (p2, p3)
    p2, p3 , p4 = get_p2p3p4_in(p1, poly)
    # Calculate the distance between point p1 and its projection point (p4)
    return math.dist(p1, p4)

def polyline_merge_check(poly1, poly2, T):
    """
    Function that checks if the two polylines should be merged or not. 
    Inputs: 
    polyline1, polyline2 ->
    T -> threshold (minimum distance between given point and polyline to determine if they shall be merged or not)

    Outputs:
    check: True (should be merged) / False (shouldn't be merged)
    """
    check = "False"

    for i, p1 in enumerate(poly1):
        dist1 = point_to_polyline_dist(p1, poly2)
        if dist1 < T:
            check = "True"
            break

    for j, p2 in enumerate(poly2):
        dist2 = point_to_polyline_dist(p2, poly1)
        if dist2 < T:
            check = "True"
            break
    return dist1,dist2, check

def delete_duplicate(poly):
    """
    Function that removes duplicate points from a polyline.

    Inputs:
    - poly: numpy array representing the polyline with duplicate points.

    Outputs:
    - poly_unique: numpy array representing the polyline with duplicate points removed.
    """
    # Initialize an array to store unique points
    poly_unique = np.array([[poly[0,0], poly[0,1]]])
    # Iterate over each point in the polyline
    for point in poly:
        # Check if the point is already in the unique array
        if point in poly_unique:
            pass  # Skip if the point is a duplicate
        else:
            # Add the point to the unique array if it's not a duplicate
            new_point = np.array([[point[0], point[1]]])
            poly_unique = np.append(poly_unique, new_point, axis=0)
    return poly_unique

def uniform_resample_by_scale(points, scale):
    # 计算每段长度
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_lengths = np.hstack([[0], np.cumsum(segment_lengths)])
    
    # 计算所需的采样点数
    total_length = cumulative_lengths[-1]
    num_samples = max(int(np.round(total_length / scale)), 2)
    
    # 均匀分布采样点
    uniform_lengths = np.linspace(0, total_length, num_samples)
    resampled_x = np.interp(uniform_lengths, cumulative_lengths, points[:, 0])
    resampled_y = np.interp(uniform_lengths, cumulative_lengths, points[:, 1])
    
    return np.column_stack((resampled_x, resampled_y))

def calculate_signed_area(polyline):
    n = len(polyline)
    area = 0.0
    for i in range(n):
        x1, y1 = polyline[i]
        x2, y2 = polyline[(i + 1) % n]  # 循环处理点，确保首尾相连
        area += (x2 - x1) * (y2 + y1)
    return area

# 确保两条 polyline 方向一致
def ensure_same_direction(polyline1, polyline2):

    poly1, poly2 = polyline1.copy(), polyline2.copy()

    area1 = calculate_signed_area(poly1)
    area2 = calculate_signed_area(poly2)
    
    # 如果方向不一致，则反转其中一条 polyline
    if area1 * area2 < 0:
        polyline2 = polyline2[::-1]
    
    return polyline1, polyline2

def ensure_same_direction_v2(polyline1, polyline2):

    p1_s = polyline1[0]
    p1_e = polyline1[-1]

    p2_s_1, p3_s_1, p4_s_1 = get_p2p3p4_in(p1_s, polyline2)
    p2_e_1, p3_e_1, p4_e_1 = get_p2p3p4_in(p1_e, polyline2)
    
    dist1_s = math.dist(p1_s, p4_s_1)
    dist1_e = math.dist(p1_e, p4_e_1)

    p2_s = polyline2[0]
    p2_e = polyline2[-1]

    p2_s_2, p3_s_2, p4_s_2 = get_p2p3p4_in(p2_s, polyline1)
    p2_e_2, p3_e_2, p4_e_2 = get_p2p3p4_in(p2_e, polyline1)
    
    dist_s_1 = math.dist(p1_s, p4_s_1)
    dist_e_1 = math.dist(p1_e, p4_e_1)

    dist_s_2 = math.dist(p2_s, p4_s_2)
    dist_e_2 = math.dist(p2_e, p4_e_2)
    
    # 如果方向不一致，则反转其中一条 polyline
    if dist_s_1 > dist_e_1:
        polyline1 = np.flip(polyline1, axis=0)  # 沿着第一个轴反转
    if dist_s_2 > dist_e_2:
        polyline2 = np.flip(polyline2, axis=0)  # 沿着第一个轴反转
    
    return polyline1, polyline2

def fit(poly):
    pts_x = poly[:, 0]  
    pts_y = poly[:, 1]
    if(len(pts_x) <= 3):
        return poly
    tck, u = splprep([pts_x, pts_y], s=1)
    x_smooth, y_smooth = splev(np.linspace(0, 1, 50), tck)
    new_vec = np.column_stack((x_smooth, y_smooth))
    return new_vec

def merge_polys_v3(poly1, poly2):
    """
    Merge two polygons into a single polygon.

    Parameters:
    - poly1: NumPy array representing the vertices of the first polygon. Each row contains the (x, y) coordinates of a vertex.
    - poly2: NumPy array representing the vertices of the second polygon. Each row contains the (x, y) coordinates of a vertex.

    Returns:
    - NumPy array representing the vertices of the merged polygon. Each row contains the (x, y) coordinates of a vertex.

    This function merges two polygons, poly1 and poly2, into a single polygon. It starts by arranging the vertices of poly1 
    to form the initial merged polygon. The function then iterates over each vertex of poly1 and computes its corresponding 
    projection point (p4) onto poly2. If p4 does not coincide with either endpoint of the line segment (p2, p3) in poly2, 
    the function averages the vertex with p4 and updates poly1. Next, the function iterates over each vertex of poly2, 
    determining its projection onto poly1 and inserting it into the merged polygon based on its position relative to the edges 
    of the merged polygon. If the projection coincides with the starting or ending vertex of the merged polygon, the vertex is 
    inserted accordingly. Otherwise, the midpoint between the vertex and its projection onto the merged polygon is computed 
    and inserted into the merged polygon. Finally, any duplicate vertices are removed, and the merged polygon is returned.
    """
    
    # Initialize the merged polygon with poly1
    poly1, poly2 = ensure_same_direction_v2(poly1,poly2)
    # poly1 = fit(poly1)
    # poly2 = fit(poly2)

    polyM = uniform_resample_by_scale(poly1, 0.5)
    polyN = uniform_resample_by_scale(poly2, 0.5)
    # polyM = poly1.copy()
    # polyN = poly2.copy()

    # Average the vertices of poly1 that correspond to the projection points (p4) onto poly2
    for i in range(len(polyM)):
        p = polyM[i]
        p2, p3, p4 = get_p2p3p4_in(p, poly2)
        dist = math.dist(p,p4)
        if not(all(p4==p2) or all(p4==p3)) and dist < 1.5:
            polyM[i] = (p+p4)/2
    
    for i in range(len(polyN)):
        p = polyN[i]
        p2, p3, p4 = get_p2p3p4_in(p, poly1)
        dist = math.dist(p,p4)
        if not(all(p4==p2) or all(p4==p3)) and dist < 1.5:
            polyN[i] = (p+p4)/2
    
    # Get the first and last vertices of poly1

    # Get the projection point (p4) of p onto poly1
 
    # Iterate over vertices of poly2
    if len(polyM) < len(polyN):
        polyM, polyN = polyN, polyM

    for p in polyN:
        polyM_e0 = polyM[0]     # first edge (Head) of main poly
        polyM_e1 = polyM[-1]    # last edge (Tail) of main poly
        polyMM = polyM.copy()        
        p2, p3, p4 = get_p2p3p4_in(p, polyMM)
        p2_index = np.argwhere(np.all(polyMM == p2, axis=1))[0][0]
        p3_index = np.argwhere(np.all(polyMM == p3, axis=1))[0][0]
        dist = math.dist(p,p4)

        # Insert p into the merged polygon based on its position relative to poly1
        if all(p4==polyM_e0):
            polyM = np.insert(polyM, p2_index, p, axis=0)
        elif all(p4==polyM_e1):
            polyM = np.insert(polyM, p3_index + 1, p, axis=0)
        elif  dist < 0.9 :
            pm = (p+p4)/2
            polyM = np.insert(polyM, p2_index+1,pm, axis=0)

    # Remove duplicate vertices
    polyM = delete_duplicate(polyM)
    polyM_resample = uniform_resample_by_scale(polyM, 0.5)
    return polyM_resample

def merge_polys_mask(poly1, poly2):
    # Initialize the merged polygon with poly1
    polyM = poly1.copy() 
    polyN = poly2.copy()

    # Average the vertices of poly1 that correspond to the projection points (p4) onto poly2
    for i in range(len(polyM)):
        p = polyM[i]
        p2, p3, p4 = get_p2p3p4_in(p, poly2)
        if not(all(p4==p2) or all(p4==p3)):
            polyM[i] = (p+p4)/2
    
    for i in range(len(polyN)):
        p = polyN[i]
        p2, p3, p4 = get_p2p3p4_in(p, poly1)
        if not(all(p4==p2) or all(p4==p3)):
            polyN[i] = (p+p4)/2
    polyM = uniform_resample_by_scale(polyM, 0.05)
    polyN = uniform_resample_by_scale(polyN, 0.05)

    from scipy.spatial import ConvexHull
    points = np.concatenate((polyM, polyN)) 
    hull = ConvexHull(points)
    contour_points = points[hull.vertices]
    contour_points = contour_points[np.argsort(hull.simplices[:, 0])]


    midpoints = []
    for i in range(len(contour_points)):
        start = contour_points[i]
        end = contour_points[(i + 1) % len(contour_points)]
        midpoint = (start + end) / 2.0
        midpoints.append(midpoint)

    midpoints = np.array(midpoints)

        # 可视化凸包轮廓与中点（作为中心线）
    plt.scatter(points[:, 0], points[:, 1], c='red', label='Original Points')
    plt.plot(np.append(contour_points[:, 0], contour_points[0, 0]), 
            np.append(contour_points[:, 1], contour_points[0, 1]), c='blue', label='Convex Hull')
    plt.plot(midpoints[:, 0], midpoints[:, 1], 'g--', label='Centerline Midpoints')
    plt.title('Convex Hull and Centerline Midpoints')
    plt.legend()
    plt.show()

    # Remove duplicate vertices
    midpoints = delete_duplicate(midpoints)
    midpoints_resample = uniform_resample_by_scale(midpoints, 0.5)

    return midpoints_resample


def merge_recursive(polys, result):

    # 基本递归终止条件：如果只有一个多边形，直接返回这个多边形
    if len(polys) == 1:
        return polys[0], None, None

    
    # 将多边形列表分成两半
    mid = len(polys) // 2
    left_half = polys[:mid]
    right_half = polys[mid:]
    
    # 递归地合并两半的多边形
    left_merged, _ , _ = merge_recursive(left_half, result)
    right_merged, _ , _ = merge_recursive(right_half, result)
    result[0].append(left_merged)
    result[1].append(right_merged)
 
    # 合并两个递归调用的结果
    return merge_polys_v3(left_merged, right_merged), left_merged, right_merged

def merge_individuals(individuals):
    if len(individuals) < 2:
        return individuals
    
    merged_list = []
    
    while len(individuals) > 1:
        merged_round = []
        # 在每一轮中，找到每个个体的最近邻进行合并
        i = 0
        while i < len(individuals):
            if i == len(individuals) - 1:
                # 如果剩下最后一个个体，直接放入结果列表中
                merged_round.append(individuals[i])
                break
            
            # 找到个体 i 的最近邻 j
            min_distance = float('inf')
            nearest_j = i + 1
            for j in range(i + 1, len(individuals)):
                d1, d2, check = polyline_merge_check(np.array(individuals[i]), np.array(individuals[j]), 0.8)
                if check:
                    min_distance = (d1 + d2)/2
                    nearest_j = j
            
            # 合并个体 i 和最近邻 j
            merged = merge_polys_v3(np.array(individuals[i]), np.array(individuals[nearest_j]))
            merged_round.append(merged)
            
            # 移除 i 和 j
            individuals.pop(nearest_j)
            individuals.pop(i)
        
        # 更新个体列表并进入下一轮合并
        individuals = merged_round
        merged_list = merged_round
    
    return merged_list

def merge_individuals_v2(individuals):
    n = len(individuals)
    if n < 2:
        return individuals

    # 初始化距离矩阵
    distance_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d1, d2, check = polyline_merge_check(np.array(individuals[i]), np.array(individuals[j]), 0.8)
            if check:
                distance_matrix[i, j] = (d1 + d2) / 2
                distance_matrix[j, i] = distance_matrix[i, j]  # 对称矩阵

    while n > 1:
        # 找到最小距离的那对个体
        min_dist_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        i, j = min_dist_idx
        if i > j:  # 保证 i < j
            i, j = j, i
        
        # 合并最近的两个个体
        merged = merge_polys_v3(np.array(individuals[i]), np.array(individuals[j]))
        individuals[i] = merged
        
        # 更新距离矩阵：更新与新合并个体相关的距离
        for k in range(n):
            if k != i and k != j:
                d1, d2, check = polyline_merge_check(np.array(individuals[i]), np.array(individuals[k]), 0.8)
                if check:
                    distance_matrix[i, k] = (d1 + d2) / 2
                    distance_matrix[k, i] = distance_matrix[i, k]
                else:
                    distance_matrix[i, k] = np.inf
                    distance_matrix[k, i] = np.inf
        
        # 删除已合并的个体 j，更新列表和距离矩阵
        individuals.pop(j)
        distance_matrix = np.delete(distance_matrix, j, axis=0)
        distance_matrix = np.delete(distance_matrix, j, axis=1)
        n -= 1

    return individuals

def refine_token_v2(global_map, proximity_th, save = False, name = 'data.pkl'):
    """
    Refines a token containing polylines by merging polylines that are in close proximity to each other.

    Parameters:
    - global_map: a list, [[],[],[]], containing polylines and the first index is the type of polyline.
    - proximity_th: Proximity threshold determining when to merge polylines.

    Returns:
    -  refined polylines after merging.

    This function refines a token containing polylines by merging polylines that are in close proximity to each other. 
    It first creates a graph to store the connected polylines (nodes) to be merged. Then, it iterates over each pair 
    of polylines and checks if they satisfy the proximity threshold condition for merging. If the condition is met, 
    an edge is added to the graph connecting the two polylines. After identifying connected components in the graph, 
    polylines within each component are merged together. Finally, the refined token containing merged polylines is returned.
    """

    import networkx as nx
    from tqdm import tqdm
    from progress.bar import Bar
    import time
    # Create graph to store the connected polylines (nodes) to merge them at the end
    start_time = time.time()

    merged_global = [[],[],[]]
    name_list = ['divider',  'boundary', 'centerline']
    jjj = 0
    ins = [[],[],[]]
    for category in range(0,2):
        jjj +=1 
        print("type: "+ name_list[category])

        instances = global_map[category]
        G = nx.Graph()

        flattened_list = [item for sublist in instances for item in sublist]
        bar = Bar('matching', max=len(flattened_list))

        for idx1, poly1 in enumerate(flattened_list):
            # import ipdb; ipdb.set_trace()
            poly2_list = flattened_list[idx1 + 1:]
            # del poly2_list[idx1]
            for idx2, poly2 in enumerate(poly2_list):
                d1, d2, check = polyline_merge_check(np.array(poly1), np.array(poly2), proximity_th)
                # print(d1, d2, check)
                if check=='True':
                    G.add_edge(idx1, idx2 + idx1 + 1)
            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f'matching: {elapsed_time:.2f} s'
        bar.finish()
        # exit(0)
        C = [list(c) for c in nx.connected_components(G)]

        # Merge polylines within each connected component
        bar = Bar('merging', max=len(C))
        merged_frame = []

        # flattened_index = [item for sublist in C for item in sublist]
        # not_match_list = [item for idx, item in enumerate(flattened_list) if idx not in flattened_index]

        # merged_frame += not_match_list
        k = 0
        import pickle as pkl
        
        for idxs2merge in C:
            k +=1 
            idxs2merge.sort()
            # merged_poly = flattened_list[idxs2merge[0]]
            # for idx2 in idxs2merge[1:]:
            #   merged_poly = merge_polys_v3(np.array(merged_poly), np.array(flattened_list[idx2]))
            result = [[],[]]
            instance_set = [np.array(flattened_list[idx]) for idx in idxs2merge]
            ins[category].append(instance_set)
            merged_poly, left, right = merge_recursive(instance_set, result)
            # import ipdb;ipdb.set_trace()
            # temp_1 = [[],[],[]]
            # temp_1[0].append([result[1][-5]])
            # temp_1[1].append([result[1][-6]])
            # plot_map(temp_1, 'merge_' + str(jjj) + '_' +str(k), save = True, fit = False)    
            merged_frame.append(merged_poly)
            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f'merging: {elapsed_time:.2f} s'
        # import ipdb; ipdb.set_trace()
        merged_global[category].append(merged_frame)
        # merged_global[category].append(flattened_list)
        bar.finish()
    if save:
        with open(name, 'wb') as f:
            pkl.dump(ins, f)
    return merged_global

def refine_token_v3(global_map, proximity_th, save = False, name = 'data.pkl'):

    import networkx as nx
    from tqdm import tqdm
    from progress.bar import Bar
    import time
    # Create graph to store the connected polylines (nodes) to merge them at the end
    start_time = time.time()

    merged_global = [[],[],[]]
    name_list = ['divider',  'boundary', 'centerline']
    jjj = 0
    ins = [[],[],[]]
    for category in range(0,2):
        jjj +=1 
        print("type: "+ name_list[category])

        instances = global_map[category]
        G = nx.Graph()

        flattened_list = [item for sublist in instances for item in sublist]
        bar = Bar('matching', max=len(flattened_list))

        for idx1, poly1 in enumerate(flattened_list):
            # import ipdb; ipdb.set_trace()
            poly2_list = flattened_list[idx1 + 1:]
            # del poly2_list[idx1]
            for idx2, poly2 in enumerate(poly2_list):
                # check = hausdorff_distance(poly1 , poly2, proximity_th)
                d1, d2, check = polyline_merge_check(np.array(poly1), np.array(poly2), proximity_th)
                # print(d1, d2, check)
                if check == 'True':
                    G.add_edge(idx1, idx2 + idx1 + 1)
            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f'matching: {elapsed_time:.2f} s'
        bar.finish()
        # exit(0)
        C = [list(c) for c in nx.connected_components(G)]

        # Merge polylines within each connected component
        bar = Bar('merging', max=len(C))
        merged_frame = []

        flattened_index = [item for sublist in C for item in sublist]
        not_match_list = [item for idx, item in enumerate(flattened_list) if idx not in flattened_index]

        merged_frame += not_match_list
        k = 0
        import pickle as pkl
        
        for idxs2merge in C:
            k +=1 
            idxs2merge.sort()
            result = [[],[]]
            instance_set = [np.array(flattened_list[idx]) for idx in idxs2merge]
            ins[category].append(instance_set)
            merged_poly, left, right = merge_recursive(instance_set, result)
  
            merged_frame.append(merged_poly)
            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f'merging: {elapsed_time:.2f} s'
        # import ipdb; ipdb.set_trace()
        merged_global[category].append(merged_frame)
        # merged_global[category].append(flattened_list)
        bar.finish()
    if save:
        with open(name, 'wb') as f:
            pkl.dump(ins, f)
    return merged_global

def is_polyline_straight(points, threshold=0.1):

    if len(points) < 2:
        return True

    # 计算起点和终点的连线
    start_point = points[0]
    end_point = points[-1]
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # 计算每个点到连线的距离
    distances = []
    for point in points:
        distance = abs(dy * point[0] - dx * point[1] + end_point[0] * start_point[1] - end_point[1] * start_point[0]) / np.sqrt(dx**2 + dy**2)
        distances.append(distance)

    max_distance = max(distances)
    return max_distance < threshold

def get_converge(poly1, poly2):
    s_1, e_1 = 0 , len(poly1) - 1

    for i in range(len(poly1)):
        point1 = poly1[i]
        point2 = poly1[len(poly1) - i - 1]
        p12, p13 , p14 = get_p2p3p4_in(point1, poly2)
        p22, p23 , p24 = get_p2p3p4_in(point2, poly2)

        if all(p12 == poly2[0]):
            s_1 = i
        elif all(p22 == poly2[-1]):
            e_1 = len(poly1) - i - 1

        if s_1 >= e_1:
            return np.array([]) 
    
    
    new_poly1 = poly1[s_1:e_1+1]
    return new_poly1

def polyline_merge_check_center(polyline1, polyline2, th_dist):
    # polyline1 和 polyline2 是两个多段线点集的 numpy 数组
    # poly1 = uniform_resample_by_scale(polyline1, 0.5)
    # poly2 = uniform_resample_by_scale(polyline2, 0.5)
    poly1 = polyline1.copy()
    poly2 = polyline2.copy()
    # flag1 = is_polyline_straight(poly1, 0.1)
    # flag2 = is_polyline_straight(poly2, 0.1)

    new_poly1 = get_converge(poly1, poly2)
    new_poly2 = get_converge(poly2, poly1)

    # new_poly1 = poly1
    # new_poly2 = poly2

    if len(new_poly1) == 0 or len(new_poly2) == 0:
        return False

    # if flag1 == flag2: #都为直线或者曲线，所以重合范围需要更大
    #     th_converge = 0.5
    # else: #其中一个是直线或者曲线，另一个是曲线，所以重合范围需要更小
    #     th_converge = 0.3

    # import ipdb; ipdb.set_trace()
    th_converge = 0

    if len(new_poly1) / len(poly1) > th_converge or len(new_poly2) / len(poly2) > th_converge:
        for p1 in new_poly1:
            dist = point_to_polyline_dist(p1, new_poly2)
            # print(dist)
            if dist > th_dist:
                return False
        return True
    else:
        return False



def hausdorff_distance(polyline1, polyline2, th):
    # polyline1 和 polyline2 是两个多段线点集的 numpy 数组
    d1 = directed_hausdorff(polyline1, polyline2)[0]
    d2 = directed_hausdorff(polyline2, polyline1)[0]
    dist = max(d1, d2)
    if dist < th:
        # print(dist)
        return True
    else:
        return False


def refine_token_center(global_map, proximity_th, save = False, name = 'data.pkl'):

    import networkx as nx
    from tqdm import tqdm
    from progress.bar import Bar
    import time
    # Create graph to store the connected polylines (nodes) to merge them at the end
    start_time = time.time()

    merged_global = [[],[],[]]
    name_list = ['divider',  'boundary', 'centerline']
    jjj = 0
    ins = [[],[],[]]
    for category in range(2,3):
        jjj +=1 
        print("type: "+ name_list[category])

        instances = global_map[category]
        G = nx.Graph()

        flattened_list = [item for sublist in instances for item in sublist]
        bar = Bar('matching', max=len(flattened_list))

        for idx1, poly1 in enumerate(flattened_list):
            # import ipdb; ipdb.set_trace()
            poly2_list = flattened_list[idx1 + 1:]
            # del poly2_list[idx1]
            for idx2, poly2 in enumerate(poly2_list):
                check = hausdorff_distance(poly1 , poly2, proximity_th)
                # check = polyline_merge_check_center(np.array(poly1), np.array(poly2), proximity_th)
                # print(d1, d2, check)
                if check:
                    G.add_edge(idx1, idx2 + idx1 + 1)
            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f'matching: {elapsed_time:.2f} s'
        bar.finish()
        # exit(0)

        C = [list(c) for c in nx.connected_components(G)]
        # import ipdb; ipdb.set_trace()

        # Merge polylines within each connected component
        bar = Bar('merging', max=len(C))
        merged_frame = []

        # flattened_index = [item for sublist in C for item in sublist]
        # not_match_list = [item for idx, item in enumerate(flattened_list) if idx not in flattened_index]

        # merged_frame += not_match_list
        k = 0
        import pickle as pkl
        
        for idxs2merge in C:
            k +=1 
            idxs2merge.sort()
            result = [[],[]]
            instance_set = [np.array(flattened_list[idx]) for idx in idxs2merge]
            ins[category].append(instance_set)
            merged_poly, left, right = merge_recursive(instance_set, result)
  
            merged_frame.append(merged_poly)
            bar.next()
            elapsed_time = time.time() - start_time
            bar.message = f'merging: {elapsed_time:.2f} s'
        # import ipdb; ipdb.set_trace()
        merged_global[category].append(merged_frame)
        # merged_global[category].append(flattened_list)
        bar.finish()
    if save:
        with open(name, 'wb') as f:
            pkl.dump(ins, f)
    return merged_global