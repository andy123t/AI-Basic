# -*- coding: utf-8 -*-
# 2024-04-21 19:12:31
# Author: Andy

import dataset
from matplotlib import pyplot as plt
import cv2
import os

# Create a directory for the image files
os.makedirs('images', exist_ok=True)

n = 100
xs,ys = dataset.get_beans(n)

w = 0.1

for i in range(n):
    x = xs[i]
    y = ys[i]
    # a=x^2
    # b=-2*x*y
    # c=y^2
    # 斜率k=2aw+b
    k = 2*(x**2)*w + (-2*x*y)
    alpha = 0.1
    w = w - alpha*k
    y_pre = w*xs
    
    plt.clf() #清空窗口
    # Create a figure
    fig, ax = plt.subplots() 
    plt.xlim(0,1)
    plt.ylim(0,1.2)
    plt.scatter(xs,ys)
    
    ax.plot(xs, y_pre)
    #plt.pause(0.01) #暂停0.01s
    plt.savefig(os.path.join('images','frame_{:04d}.png'.format(i)),dpi=300)
    
    # Close the figure to free up memory
    plt.close(fig)


# image and video path
image_folder = 'images'
video_name = 'video.avi'

# image to frame
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

