import numpy as np
import matplotlib.pyplot as plt

# 生成随机 polyline
def generate_random_polyline(num_points, center=(0, 0), scale=10):
    angles = np.sort(np.random.rand(num_points) * 2 * np.pi)
    radii = scale * np.random.rand(num_points)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return list(zip(x, y))

# 计算有向面积以判断顺时针或逆时针# 计算有向面积以判断顺时针或逆时针
# 计算重心
# 计算有向面积以判断顺时针或逆时针
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
    # poly1.append(poly1[0])
    # poly2.append(poly2[0])

    area1 = calculate_signed_area(poly1)
    area2 = calculate_signed_area(poly2)
    
    # 如果方向不一致，则反转其中一条 polyline
    if area1 * area2 < 0:
        polyline2.reverse()
    
    return polyline1, polyline2
# 可视化函数
def plot_polylines_with_arrows(polyline1, polyline2, title):
    plt.figure(figsize=(8, 8))
    
    # 画 polyline1
    x1, y1 = zip(*polyline1)
    plt.plot(x1, y1, 'b-', marker='o', label='Polyline 1')
    for i in range(len(polyline1) - 1):
        plt.arrow(x1[i], y1[i], x1[i + 1] - x1[i], y1[i + 1] - y1[i], head_width=0.5, head_length=0.6, fc='blue', ec='blue')
    
    # 画 polyline2
    x2, y2 = zip(*polyline2)
    plt.plot(x2, y2, 'r-', marker='o', label='Polyline 2')
    for i in range(len(polyline2) - 1):
        plt.arrow(x2[i], y2[i], x2[i + 1] - x2[i], y2[i + 1] - y2[i], head_width=0.5, head_length=0.6, fc='red', ec='red')
    
    plt.legend()
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

# 生成随机 polyline
np.random.seed(42)
polyline1 = generate_random_polyline(num_points=8, center=(0, 0), scale=10)
polyline2 = generate_random_polyline(num_points=8, center=(15, 15), scale=10)
polyline1.reverse()
# 调整方向前的可视化
plot_polylines_with_arrows(polyline1, polyline2, "Before Aligning Directions")

# 调整方向
aligned_polyline1, aligned_polyline2 = ensure_same_direction(polyline1, polyline2)

# 调整方向后的可视化
plot_polylines_with_arrows(aligned_polyline1, aligned_polyline2, "After Aligning Directions")
