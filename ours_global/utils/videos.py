import cv2
import os
import numpy as np
import re
from tqdm import tqdm

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

def make_video(images_folder, output_video_path, fps=0.01):
    images = [img for img in os.listdir(images_folder) if img.endswith(".png")]
    images.sort(key=natural_sort_key)  # 根据数字排序图片

    # 读取第一张图片以确定帧的大小
    frame = cv2.imread(os.path.join(images_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码器和输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in tqdm(images):
        img_path = os.path.join(images_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)  # 将帧写入视频
    # 添加一个额外的空帧
    extra_frame = np.zeros_like(frame)
    for _ in range(int(fps)):  # 添加与帧率相等数量的空帧
        video.write(extra_frame)

    video.release()

# 使用函数
images_folder = 'maps/video'  # 图片文件夹路径
output_video_path = 'output_video.mp4'  # 输出视频路径
make_video(images_folder, output_video_path, fps=5)
