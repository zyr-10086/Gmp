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
from . import Stitch_lib as slb


import pandas as pd

import pickle as pkl
from shapely.geometry import LineString, Polygon
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from progress.bar import Bar
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict


def fit_1d(vec):
    # 提取输入点集的 x 和 y 坐标
    pts_x = vec[:, 0]
    pts_y = vec[:, 1]
    
    # 使用 numpy.polyfit 进行一阶线性拟合
    m, b = np.polyfit(pts_x, pts_y, 1)
    
    # 根据拟合结果计算新的 y 坐标
    fitted_y = m * pts_x + b
    
    # 重新组合新的 x 和 y 坐标成新的点集 new_vec
    new_vec = np.column_stack((pts_x, fitted_y))
    
    return new_vec
def remove_bends_and_keep_longest(polyline):
    if len(polyline) < 3:
        return polyline  # 如果 polyline 少于3个点，则无法检测弯折
    
    segments = []
    current_segment = [polyline[0]]
    
    for i in range(1, len(polyline) - 1):
        delta_1 = polyline[i] - polyline[i - 1]
        delta_2 = polyline[i + 1] - polyline[i]
        
        if np.linalg.norm(delta_1) == 0 or np.linalg.norm(delta_2) == 0:
            continue
        
        direction_1 = delta_1 / np.linalg.norm(delta_1)
        direction_2 = delta_2 / np.linalg.norm(delta_2)
        
        # 检测反向弯折
        if np.dot(direction_1, direction_2) > -0.99:
            current_segment.append(polyline[i])
        else:
            current_segment.append(polyline[i])
            segments.append(np.array(current_segment))
            current_segment = [polyline[i]]  # 开始新的一段
    
    current_segment.append(polyline[-1])
    segments.append(np.array(current_segment))
    
    # 找到最长的段落
    max_length_segment = max(segments, key=lambda s: np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))
    
    return max_length_segment

def fit_2d_rotate(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    
    # 主成分方向
    principal_direction = pca.components_[0]
    
    # 计算旋转角度
    angle = np.arctan2(principal_direction[1], principal_direction[0])
    
    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # 旋转数据
    rotated_points = np.dot(points, rotation_matrix.T)
    
    # 使用一维拟合方法
    x = rotated_points[:, 0]
    y = rotated_points[:, 1]
    m, b = np.polyfit(x, y, 1)
    
    # 计算新的 y 坐标
    fitted_y_rotated = m * x + b
    
    # 反旋转回原始坐标系
    new_points = np.dot(np.column_stack((x, fitted_y_rotated)), rotation_matrix)
    
    return remove_bends_and_keep_longest(new_points)

def get_range(pkl_name):

    x = [[-2277.267190777856, -2236.732312743579],
         [-2289.1948418856778, -2227.818107782266],
         [-2276.730960326724, -2228.549954904361],
         [-2273.612641894883, -2230.6970198560184],
         [-2278.152935931118, -2226.9286666458834],
         [-2278.503080469362, -2228.017604020914],
         [-2278.370734007204, -2227.0679722911937],
         [-2278.60379043922, -2228.0464981927635],
         [-2275.597047700434, -2232.87592293603],
         [-2275.5987290265966, -2230.7211669590492],
         [-2277.6710042038126, -2227.26966198856],
         [-2277.6076610467067, -2228.9739792339296],
         [-2271.9602029508433, -2235.839374799022],
         [-2241.0879754448133, -2178.442427333226],
         [-2239.4964875301725, -2192.0026011678256],
         [-2228.495244280423, -2178.284996675112]]
    y = [[-752.050837038146, -522.3228417937371],
         [-762.2027449767028, -512.6842424596452],
         [-563.1302765976382, -528.8487883822589],
         [-545.9059551536509, -529.2596514290487],
         [-579.8825298218386, -562.8074974886914],
         [-597.3078047641304, -579.6338499325618],
         [-656.335149821637, -600.2464130146694],
         [-698.2847119191482, -672.7414543972795],
         [-753.393813184462, -706.4474252498134],
         [-550.3667948462707, -529.0375057060023],
         [-649.0118006618776, -551.2333025146362],
         [-723.8195925477045, -649.4154296566635],
         [-753.9045797813873, -724.8609376854617],
         [-815.9402829810654, -596.3372464915825],
         [-770.8908060581186, -709.9387400806884],
         [-747.4708692772189, -600.2650290690565]]

    if(pkl_name[0] == 'l' or pkl_name == 'global'):
        range_x = x[0]
        range_y = y[0]
    else:
        index = int(pkl_name[1])
        range_x = x[index + 1]
        range_y = y[index + 1]
    range_x[0] -= 20
    range_x[1] += 20
    range_y[0] -= 20
    range_y[1] += 20
    return range_x, range_y

def point_position(p1, p2, p3):
    # 计算直线方程的系数
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0] * p1[1] - p1[0] * p2[1]
    
    # 计算点到直线的有向距离
    distance = (a * p3[0] + b * p3[1] + c) / np.sqrt(a**2 + b**2)
    
    # 判断点的位置
    return distance

def calculate_angle(p1, p2, p3, p4):
    # 计算向量 v1 和 v2
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    
    # 计算向量的模长
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 计算点积
    dot_product = np.dot(v1, v2)
    
    # 计算夹角（弧度）
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    
    # 将夹角转换为度数
    angle_deg = np.degrees(angle_rad)
    
    # 将角度限制在0-90度之间
    return min(abs(angle_deg), 180 - abs(angle_deg))

def process_point(dp, ps, center, dist_th):
    # if not ps:
    #     return False, center

    dists = [math.dist(dp, p) for p in ps]
    if min(dists) > dist_th:
        return False, center
    else:
        index = dists.index(min(dists))
        p2 = ps[index]
        center_point = (p2 + dp) / 2
        return True, np.vstack([center, center_point]) if center.size != 0 else center_point.reshape(1, 2)

def reorder_polyline(points):
    # Calculate the difference between the last and first points to determine the orientation
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    
    if abs(dy) > abs(dx):
        # Nearly vertical, sort by y-coordinate (descending)
        points = sorted(points, key=lambda p: p[1], reverse=True)
    else:
        # Nearly horizontal, sort by x-coordinate (ascending)
        points = sorted(points, key=lambda p: p[0])

    return np.array(points)

def merge_close_polylines(polylines, threshold):
    """合并首尾距离在阈值内的 polylines"""
    merged = True
    
    while merged:
        merged = False
        i = 0
        
        while i < len(polylines):
            polyline1 = polylines[i]
            found_merge = False
            
            for j in range(i + 1, len(polylines)):
                polyline2 = polylines[j]
                
                # 计算 polyline1 的尾点与 polyline2 的首点之间的距离
                dist1 = math.dist(polyline1[-1], polyline2[0])
                # 计算 polyline2 的尾点与 polyline1 的首点之间的距离
                dist2 = math.dist(polyline2[-1], polyline1[0])
                
                if dist1 <= threshold and dist2 <= threshold:
                    # 如果两个距离都小于阈值，选择距离较短的方向合并
                    if dist1 <= dist2:
                        merged_polyline = np.vstack([polyline1, polyline2])
                    else:
                        merged_polyline = np.vstack([polyline2, polyline1])
                elif dist1 <= threshold:
                    # 合并 polyline1 的尾点与 polyline2 的首点
                    merged_polyline = np.vstack([polyline1, polyline2])
                elif dist2 <= threshold:
                    # 合并 polyline2 的尾点与 polyline1 的首点
                    merged_polyline = np.vstack([polyline2, polyline1])
                else:
                    continue  # 如果都不满足阈值条件，继续下一个比较
                
                # 找到可合并的 polyline，更新列表并标记合并
                polylines.pop(j)
                polylines[i] = merged_polyline
                found_merge = True
                merged = True
                break  # 跳出内层循环，重新开始合并检查
            
            if not found_merge:
                i += 1  # 如果没有找到可以合并的，继续下一个 polyline
    
    return polylines

def get_path(predecessors, start, end):
    path = []
    current_node = end
    while current_node != start and current_node != -9999:
        path.append(current_node)
        current_node = predecessors[start, current_node]
    if current_node == -9999:
        return []  # 表示从 start 到 end 不可达
    path.append(start)
    path.reverse()
    return path

def get_center_straight(data, dist_th, curve_th, save = False, name = 'pkls/center_v2/straight_center.pkl'):
    dividers = data[0][0]
    boundaries = data[1][0]
    centers_1, centers_2, roads = [], [], []


    for divider in dividers:
        center_1, center_2, road = np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        for d_index in range(len(divider) - 1):
            dp = divider[d_index]
            ps_1, ps_2 = [], []

            for boundary in boundaries:
                boundary = slb.uniform_resample_by_scale(boundary, 0.1)
                p2, _, _ = slb.get_p2p3p4_in(dp, boundary)
                direction = point_position(dp, divider[d_index + 1], p2)
                if calculate_angle(dp, divider[d_index + 1], dp, p2) > 60 and math.dist(dp,p2) > 1.5 and math.dist(dp,p2) < 6:
                    (ps_1 if direction > 0 else ps_2).append(p2)          

            flag_1, flag_2 = False, False    

            if ps_1:
                flag_1, center_1 = process_point(dp, ps_1, center_1, dist_th)
            if ps_2:
                flag_2, center_2 = process_point(dp, ps_2, center_2, dist_th)
            
            road = np.vstack([road,dp])

            if not(flag_1 and flag_2):
                if len(road) != 0:
                    if len(road) > 1:
                        roads.append(road)
                    road = np.array([]).reshape(0, 2)
                if len(center_1) != 0:
                    if len(center_1) > 1:
                        centers_1.append(center_1)
                    center_1 = np.array([]).reshape(0, 2)
                if len(center_2) != 0:
                    if len(center_2) > 1:
                        centers_2.append(center_2[::-1])
                    center_2 = np.array([]).reshape(0, 2)
        
        if len(road) > 1:
            roads.append(road)
        if len(center_1) > 1:
            centers_1.append(center_1)
        if len(center_2) > 1:
            centers_2.append(center_2[::-1])

    st_lines = centers_1 + centers_2
    road_list, road_graph_line, road_graph_dist, line_s, line_e = construct_road_network(st_lines, boundaries)

    # start = line_s[2] # i = 2
    # end = line_e[20]  # j = len(line_s) + 20
    
    # start_index = len(line_s) + 15
    # end_index = len(line_s) + 18

    # graph = csr_matrix(road_graph_dist)
    # dist_matrix, predecessors = shortest_path(csgraph=graph, directed=True, return_predecessors=True)
    
    # path = get_path(predecessors, start_index, end_index)

    # # import ipdb;ipdb.set_trace()
    # path_list = []

    # for i in range(len(path) - 1):
    #     i_start = path[i]
    #     i_end = path[i + 1]
    #     path_list.append(road_graph_line[i_start][i_end])


    data[2].append(road_list)
    data.append([])
    # data[3].append(path_list)
    
    if save:
        with open(name, 'wb') as f:
            pkl.dump(data, f)


    return road_graph_line, road_graph_dist


# 提取每条 polyline 的起点和终点# 提取每条 polyline 的起点和终点
def get_start_end_points(polylines):
    start_points = []
    end_points = []
    for polyline in polylines:
        start_points.append(polyline[0])  # 起点
        end_points.append(polyline[-1])   # 终点
    return np.array(start_points), np.array(end_points)

# 计算两个点集之间的距离矩阵（仅计算起点和终点之间的距离）
def calculate_start_end_distance_matrix(start_points, end_points):

    matrix_start_start = np.zeros((len(start_points), len(start_points)))
    matrix_end_end = np.zeros((len(end_points), len(end_points)))
    matrix_start_end = np.zeros((len(start_points), len(end_points)))
    matrix_end_start = distance_matrix(end_points, start_points)

    for i in range(len(start_points)):
        matrix_start_end[i][i] = math.dist(start_points[i], end_points[i])
        matrix_end_start[i][i] = 0

    result = np.block([[matrix_start_start, matrix_start_end], [matrix_end_start, matrix_end_end]])
    result[result == 0] = np.inf

    return result

def polyline_length(polyline):
    # 计算相邻点的差值
    differences = np.diff(polyline, axis=0)
    # 计算每个差值的欧几里得长度
    lengths = np.linalg.norm(differences, axis=1)
    # 返回长度总和
    return np.sum(lengths)
def smooth_transition(A, B, num_points=50):
    """
    生成平滑连接线，使用参数化的方法进行插值
    A: Polyline A 的坐标点 np.array 格式，例如 [[0, 0], [2, 0]]
    B: Polyline B 的坐标点 np.array 格式，例如 [[5, 1], [7, 2]]
    num_points: 平滑曲线生成的点数
    """
    # 起点和终点
    start_point = A[-1]
    end_point = B[0]

    # 控制点
    control_points = np.array([A[-2], start_point, end_point, B[1]])
    x = control_points[:, 0]
    y = control_points[:, 1]

    # 计算控制点的累积弧长（作为参数 t）
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    t = np.concatenate(([0], np.cumsum(distances)))

    # 对 t, x 和 t, y 进行样条插值
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    # 使用均匀分布的 t 值生成插值点
    t_new = np.linspace(t[0], t[-1], num_points)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    
    # 组合平滑曲线坐标
    smooth_curve = np.vstack((x_new, y_new)).T

    start_index = np.argmin(np.linalg.norm(smooth_curve - start_point, axis=1))
    end_index = np.argmin(np.linalg.norm(smooth_curve - end_point, axis=1)) + 1  # 包含 B 起点
    smooth_curve_segment = smooth_curve[start_index:end_index]
    return smooth_curve_segment

def smooth_transition_v2(A, B, num_points=50, s = 1):
    """
    生成平滑连接线，使用 B-spline 进行插值
    A: Polyline A 的坐标点 np.array 格式，例如 [[0, 0], [2, 0]]
    B: Polyline B 的坐标点 np.array 格式，例如 [[5, 1], [7, 2]]
    num_points: 平滑曲线生成的点数
    degree: B-spline 的阶数，控制平滑程度
    """
    # 起点和终点
    start_point = A[-1]
    end_point = B[0]

    # 控制点，包括起点和终点
    control_points = np.array([A[-2], start_point, end_point, B[1]])
    x = control_points[:, 0]
    y = control_points[:, 1]

    tck, u = splprep([x, y], s=s)
    x_new, y_new = splev(np.linspace(0, 1, num_points), tck)
    smooth_curve = np.vstack((x_new, y_new)).T

    start_index = np.argmin(np.linalg.norm(smooth_curve - start_point, axis=1))
    end_index = np.argmin(np.linalg.norm(smooth_curve - end_point, axis=1)) + 1  # 包含 B 起点
    smooth_curve_segment = smooth_curve[start_index:end_index]

    return smooth_curve_segment

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path

# 构建遵循方向性规则的最短路网
def construct_road_network(polylines, boundaries):
    # 提取起点和终点
    start_points, end_points = get_start_end_points(polylines)
    
    # 计算从终点到起点的距离矩阵
    dist_matrix = calculate_start_end_distance_matrix(start_points, end_points)
    
    # 构建最小生成树（遵循方向性）
    mst = minimum_spanning_tree(dist_matrix).toarray()
    num_roads = len(polylines)
    straight_matrix = dist_matrix[:num_roads,num_roads:].copy()
    straight_matrix[straight_matrix == np.inf] = 0
    mst[:num_roads,num_roads:] = straight_matrix

    # 保存生成树结果形成的路网为list，方便处理
    num_points = 2 * num_roads
    mst_edges = {(i, j): mst[i, j] for i in range(num_points) for j in range(num_points) if mst[i,j]}
    road_network = []
    road_network_polylines = []
    road_network_matrix = defaultdict(lambda: defaultdict(list))

    for (i, j), dist in mst_edges.items():
        road_network.append((i, j))
        start = start_points[i] if i < num_roads else end_points[i - num_roads]
        end = start_points[j] if j < num_roads else end_points[j - num_roads]
        new_line = np.array([start, end])
        flag = True

        for bound in boundaries:
            bound_line = LineString(bound)
            is_intersect = bound_line.intersects(LineString(new_line))
            if is_intersect:
                flag = False
                break

        if flag:
            road_network_polylines.append(new_line)
            road_network_matrix[i][j] = new_line
        else:
            # 曲线拟合
            A = polylines[i] if i < num_roads else polylines[i - num_roads]
            B = polylines[j] if j < num_roads else polylines[j - num_roads]
            curve = smooth_transition_v2(A,B,s=50)
            flag = True

            for bound in boundaries:
                bound_line = LineString(bound)
                is_intersect = bound_line.intersects(LineString(curve))
                if is_intersect:
                    flag = False
                    break      
                            
            if flag:
                road_network_polylines.append(curve)
                road_network_matrix[i][j] = curve
    
    # import ipdb;ipdb.set_trace()
    #添加边
    for i in range(num_points):
        for j in range(num_points):
            if i != j and dist_matrix[i][j] < np.inf:
                # Check if adding this edge creates a shorter path in the network
                if (i, j) not in road_network:
                    # import ipdb;ipdb.set_trace()
                    # Check current shortest path in the MST
                    dist_mst = shortest_path(csr_matrix(mst), directed=True, indices=i)[j]
                    
                    # If the direct connection is significantly shorter, add it
                    if dist_matrix[i][j] < dist_mst/3 and dist_matrix[i][j] < 30:

                        start = start_points[i] if i < num_roads else end_points[i - num_roads]
                        end = start_points[j] if j < num_roads else end_points[j - num_roads]
                        new_line = LineString(np.array([start, end]))
                        flag = True

                        for bound in boundaries:
                            bound_line = LineString(bound)
                            is_intersect = bound_line.intersects(new_line)
                            if is_intersect:
                                flag = False
                                break

                        if flag:
                            mst[i,j] = dist_matrix[i][j]
                            road_network_polylines.append(np.array([start, end]))
                            road_network.append((i, j)) 
                            road_network_matrix[i][j] = np.array([start, end])
                        else:

                            # 曲线拟合
                            A = polylines[i] if i < num_roads else polylines[i - num_roads]
                            B = polylines[j] if j < num_roads else polylines[j - num_roads]
                            curve = smooth_transition(A,B)
                            flag = True

                            for bound in boundaries:
                                bound_line = LineString(bound)
                                is_intersect = bound_line.intersects(LineString(curve))
                                if is_intersect:
                                    flag = False
                                    break      
                                            
                            if flag:
                                length = polyline_length(curve)
                                if length < dist_mst/3:
                                    road_network.append((i, j)) 
                                    road_network_polylines.append(curve)
                                    mst[i,j] = polyline_length(curve)
                                    road_network_matrix[i][j] = curve

    #对网络整体还要有个滤除
    for i in range(num_points):
        for j in range(num_points):
            if i != j and dist_matrix[i][j] < np.inf:
                # Check if adding this edge creates a shorter path in the network
                tmp_mst = mst.copy()
                tmp_mst[i][j] = np.inf
                if (i, j) in road_network:
                    # import ipdb;ipdb.set_trace()
                    # Check current shortest path in the MST
                    dist_mst = shortest_path(csr_matrix(tmp_mst), directed=True, indices=i)[j]
                    if dist_mst < dist_matrix[i][j] * 3:
                        road_network.remove((i, j))
                        # 移除与 road_network_matrix[i][j] 相同的数组
                        road_network_polylines = [polyline for polyline in road_network_polylines\
                                                   if not np.array_equal(polyline, road_network_matrix[i][j])]
                        road_network_matrix[i][j] = []

                        mst[i][j] = np.inf
    #转换回polyline
    # road_network_polylines = []
    # for i, j in road_network:
    #     start = start_points[i] if i < num_roads else end_points[i - num_roads]
    #     end = start_points[j] if j < num_roads else end_points[j - num_roads]
    #     road_network_polylines.append(np.array([start, end]))

    # 将最小生成树转换为 polyline 连接格式
    # road_network_polylines = []
    # for i in range(len(mst)):
    #     for j in range(len(mst[i])):
    #         if mst[i, j] > 0:  # 如果存在连接
    #             # 从 end_points[i] 到 start_points[j] 的 polyline
    #             line = np.array([end_points[i], start_points[j]])
    #             road_network_polylines.append(line)
    
    return road_network_polylines, road_network_matrix, mst, start_points, end_points


def found_road(point, roads):
    for index, road in enumerate(roads):
        if np.array_equal(point, road[0]):
            return 'start', index
        elif np.array_equal(point, road[-1]):
            return 'end', index
    return None, None

def create_shape(points):
    if len(points) == 2:
        # 如果只有两个点，生成线段
        return LineString(points)
    elif len(points) > 2:
        # 如果有三个或更多点，生成凸包作为凸多边形
        hull = ConvexHull(points)
        convex_points = points[hull.vertices]
        return Polygon(convex_points)

# 对路口进行筛选，保留合理的路口
def filter_intersections(junctions):
    # junctions 是包含多个路口的列表，每个路口为 numpy array
    filtered_junctions = junctions.copy()  # 复制一份进行操作

    i = 0
    while i < len(filtered_junctions):
        shape_i = create_shape(filtered_junctions[i])
        j = i + 1
        while j < len(filtered_junctions):
            shape_j = create_shape(filtered_junctions[j])
            if shape_i.intersects(shape_j):
                # 如果两个路口相交，比较面积或长度
                if shape_i.area < shape_j.area:
                    filtered_junctions.pop(j)  # 删除较大的
                else:
                    filtered_junctions.pop(i)  # 删除较大的
                    i -= 1  # 重新检查当前 i
                    break  # 重新处理当前 i
            else:
                j += 1
        i += 1

    return filtered_junctions


def get_inter_lines(inters, roads, left, right):
    lines = []
    
    inters = filter_intersections(inters)
    # import ipdb; ipdb.set_trace()

    for inter in inters:
        inputs = []
        outputs = []
        for p in inter:
            flag, index = found_road(p, roads)
            if flag =='start' :
                outputs.append(left[index][0])
                inputs.append(right[index][-1])
                # inputs.append(roads[index][0])
                # outputs.append(roads[index][0])
            elif flag == 'end':
                inputs.append(left[index][-1])
                outputs.append(right[index][0])
                # inputs.append(roads[index][-1])
                # outputs.append(roads[index][-1])

        # import ipdb;ipdb.set_trace()

        for input in inputs:
            for output in outputs:
                line = np.array([input, output])
                lines.append(line)
        # break

    return lines

def get_intersection_v1(roads, curve_th):
    intersections = []

    for c_i in range(len(roads)):
        c_1 = roads[c_i]
        points_pares = []
        min_dists = []

        for c_j in range(len(roads)):
            if c_i == c_j:
                continue  # 跳过与自己比较的情况

            c_2 = roads[c_j]
            
            points_pare = np.array([]).reshape(0, 2)
            min_dist = float('inf')

            for i1 in range(-1, 1):  # 注意 range 的范围，-1 到 0
                for i2 in range(-1, 1):  # 同样注意 range 的范围
                    cp1 = c_1[i1]
                    cp2 = c_2[i2]
                    dist = math.dist(cp1, cp2)
                    if dist < min_dist:
                        min_dist = dist
                        points_pare = np.array([cp1, cp2])
            points_pares.append(points_pare)
            min_dists.append(min_dist)

        if min_dists:
            if min(min_dists) < 1:
                index = min_dists.index(min(min_dists))
                intersections.append(points_pares[index])
            elif min(min_dists) > curve_th:
                min_index = min_dists.index(min(min_dists))
                temp_lst = min_dists[:]
                temp_lst[min_index] = float('inf')
                second_min_index = temp_lst.index(min(temp_lst))
                intersections.append(points_pares[min_index])
                # intersections.append(points_pares[second_min_index])
            else:
                for p_index in range(len(points_pares)):
                    if min_dists[p_index] < curve_th:
                        intersections.append(points_pares[p_index])
    
    return intersections
    
def get_intersection(roads, curve_th):
    intersections = []

    for c_i in range(len(roads)):
        c_1 = roads[c_i]
        intersections_ci = [[], []]

        for i1 in [-1, 0]:
            cp1 = c_1[i1]
            points_pares = []
            min_dists = []

            for c_j in range(len(roads)):
                if c_i == c_j:
                    continue  # 跳过与自己比较的情况
                c_2 = roads[c_j]

                min_dist = float('inf')
                closest_point = None

                for i2 in [-1, 0]:
                    cp2 = c_2[i2]
                    dist = math.dist(cp1, cp2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = cp2

                points_pares.append(closest_point)
                min_dists.append(min_dist)

            if min_dists:
                if min(min_dists) < 1:
                    index = min_dists.index(min(min_dists))
                    intersections_ci[i1 + 1].append(points_pares[index])
                    break
                elif min(min_dists) > curve_th:
                    min_index = min_dists.index(min(min_dists))
                    temp_lst = min_dists[:]
                    temp_lst[min_index] = float('inf')
                    second_min_index = temp_lst.index(min(temp_lst))
                    intersections_ci[i1 + 1].append(points_pares[min_index])
                    # intersections_ci[i1 + 1].append(points_pares[second_min_index])
                else:
                    for p_index in range(len(points_pares)):
                        if min_dists[p_index] < curve_th:
                            intersections_ci[i1 + 1].append(points_pares[p_index])

        # 去除重复端点
        inter1 = intersections_ci[0]
        inter2 = intersections_ci[1]

        for pare_point in inter1:
            if any(np.array_equal(pare_point, p) for p in inter2):
                dist_1 = math.dist(pare_point, c_1[-1])
                dist_2 = math.dist(pare_point, c_1[0])
                intersections.append(np.array([c_1[-1], pare_point]) if dist_1 < dist_2 else np.array([pare_point, c_1[0]]))
            else:
                intersections.append(np.array([c_1[-1], pare_point]))

        for pare_point in inter2:
            if not any(np.array_equal(pare_point, p) for p in inter1):
                intersections.append(np.array([c_1[0], pare_point]))

    # return intersections
    return merge_connected_arrays(intersections)


def merge_connected_arrays(arr_list):
    # 用于存储最终的连通数组组，使用 tuple 以支持集合操作
    connected_groups = []

    for arr in arr_list:
        # 将数组转换为 tuple 进行集合操作
        points = [tuple(point) for point in arr]
        merged = False

        dist = math.dist(points[0], points[1])
        # if dist > 2:
        for group in connected_groups:
            if set(group).intersection(points):
                # 如果有交集，将当前数组的元素合并到该组
                group.update(points)
                merged = True
                break

        if not merged:
            # 如果没有找到可合并的组，创建一个新的连通组
            connected_groups.append(set(points))

    # 将最终的连通组转换回 numpy 数组并排序
    return [np.array(sorted(group, key=lambda x: (x[0], x[1]))) for group in connected_groups]
   