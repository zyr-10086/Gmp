import Stitch_lib as slb
import numpy as np
import pandas as pd
import math
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.patches as patches


poly1 = np.array([[29.50753769,  3.42424242],
                [29.1959799 ,  3.42424242],
                [26.3919598 , -5.62626263],
                [29.50753769, -7.56565657]])
poly2 = np.array([[29.50753769,  0.19191919],
                [26.08040201, -5.94949495],
                [29.50753769, -7.24242424]])
poly3 = np.array([[-30.0,          6.65656566],
                [ 18.60301508,   6.97979798],
                [ 23.5879397,   5.68686869],
                [ 29.50753769,   4.39393939]])
poly4 = np.array([[-30.0,          6.65656566],
                [ 16.11055276,   6.97979798],
                [ 21.40703518,   6.33333333],
                [ 29.81909548,   4.39393939]])
# 合并两个 polyline 的点集

poly1 = slb.uniform_resample_by_scale(poly1, 0.1)
poly2 = slb.uniform_resample_by_scale(poly2, 0.1)

points = np.vstack((poly1, poly2))

# 步骤 2: 提取轮廓 (使用 Alpha Shapes)
def alpha_shape(points, alpha):
    if len(points) < 4:
        return points

    tri = Delaunay(points)
    triangles = points[tri.vertices]
    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = a * b * c / (4.0 * area)
    filtered = triangles[circum_r < 1.0 / alpha]
    return np.unique(filtered.reshape(-1, 2), axis=0)

# 计算 alpha shape
alpha = 1.5
contour_points = alpha_shape(points, alpha)

# 步骤 3: 通过最近邻排序生成 polyline
from scipy.spatial import distance_matrix

def order_points(points):
    dist_matrix = distance_matrix(points, points)
    order = [0]
    while len(order) < len(points):
        last_index = order[-1]
        next_index = np.argmin(dist_matrix[last_index])
        dist_matrix[:, last_index] = np.inf  # 确保不会重复选择
        order.append(next_index)
    return points[order]

polyline = order_points(contour_points)

# 可视化 polyline
plt.plot(polyline[:, 0], polyline[:, 1], 'b-', label='Fused Polyline')
plt.scatter(points[:, 0], points[:, 1], color='red', label='Original Points')
plt.legend()
plt.show()