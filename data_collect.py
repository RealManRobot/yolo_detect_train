"""
采集图片数据用来训练yolov8
"""

import pyrealsense2 as rs
import cv2
import os
import time
import numpy as np
import re

# 创建保存图像的文件夹
color_output_folder = 'images\images'
os.makedirs(color_output_folder, exist_ok=True)

# 获取当前文件夹中已有图片的最大序号
existing_images = [f for f in os.listdir(color_output_folder) if f.lower().endswith('.png')]
max_number = 0
pattern = re.compile(r'^(\d+)\.png$')
for img_name in existing_images:
    match = pattern.match(img_name)
    if match:
        number = int(match.group(1))
        if number > max_number:
            max_number = number
start_index = max_number  # 下一个图片序号从max_number开始+1


# 相机配置
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 创建对齐对象
align_to = rs.stream.color
align = rs.align(align_to)

print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Start Detectionqq >>>>>>>>>>>>>>>>>>>>>>>>>>>>')

start_time = time.time()
frame_count = 0

while True:

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 生成文件名
    frame_number = start_index + frame_count + 1
    filename = f'{frame_number:04d}.png'
    color_file_path = os.path.join(color_output_folder, filename)

    cv2.imshow('Color Image', color_image)

    c = cv2.waitKey(1) & 0xFF

    if c == ord('s'):
        cv2.imwrite(color_file_path, color_image)
        frame_count += 1
        print(f'frame_count:{frame_count}')

    elif c == ord('q'):
        break


# 关闭窗口并停止相机
cv2.destroyAllWindows()
pipeline.stop()

print(f'Total frames captured: {frame_count}')
