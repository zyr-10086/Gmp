import numpy as np
import matplotlib.pyplot as plt

# 生成不均匀采样的 polyline 点集
def generate_polyline():
    key_points = np.array([
        [0, 0],
        [3, 5],
        [7, 8],
        [10, 4],
        [13, 10]
    ])
    
    t_values = np.sort(np.random.rand(10))
    segment_lengths = np.linalg.norm(np.diff(key_points, axis=0), axis=1)
    cumulative_lengths = np.cumsum(segment_lengths)
    total_length = cumulative_lengths[-1]
    
    points = []
    for t in t_values:
        current_length = t * total_length
        segment_index = np.searchsorted(cumulative_lengths, current_length)
        previous_length = 0 if segment_index == 0 else cumulative_lengths[segment_index - 1]
        segment_ratio = (current_length - previous_length) / segment_lengths[segment_index]
        start_point = key_points[segment_index]
        end_point = key_points[segment_index + 1]
        point = start_point + segment_ratio * (end_point - start_point)
        points.append(point)
    
    return np.array(points)

# 根据指定比例尺进行均匀采样
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

# 绘制原始和重采样后的 polyline
def plot_polylines(original_points, resampled_points):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制原始 polyline
    axs[0].plot(original_points[:, 0], original_points[:, 1], '-o', label='Original polyline')
    axs[0].scatter(original_points[:, 0], original_points[:, 1], color='red')
    axs[0].set_title('Original Polyline')
    axs[0].set_aspect('equal')

    # 绘制重采样后的 polyline
    axs[1].plot(resampled_points[:, 0], resampled_points[:, 1], '-o', label='Resampled polyline')
    axs[1].scatter(resampled_points[:, 0], resampled_points[:, 1], color='blue')
    axs[1].set_title('Resampled Polyline')
    axs[1].set_aspect('equal')

    plt.tight_layout()
    plt.show()

# 主程序
original_points = generate_polyline()  # 生成原始的polyline
scale = 1.5  # 设定比例尺，两个点之间的目标距离
resampled_points = uniform_resample_by_scale(original_points, scale)  # 根据比例尺进行重采样

# 绘制结果
plot_polylines(original_points, resampled_points)
