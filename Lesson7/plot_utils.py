import numpy as np
import matplotlib.pyplot as plt

def show_scatter(xs, y):
    x = xs[:, 0]
    z = xs[:, 1]
    fig = plt.figure()
    # 直接使用 projection='3d' 创建 3D 坐标轴
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, z, y)
    plt.show()

def show_surface(x, z, forward_propgation):
    x = np.arange(np.min(x), np.max(x), 0.1)
    z = np.arange(np.min(z), np.max(z), 0.1)
    x, z = np.meshgrid(x, z)
    y = forward_propgation(x, z)
    fig = plt.figure()
    # 直接使用 projection='3d' 创建 3D 坐标轴
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()

def show_scatter_surface(xs, y, forward_propgation):
    x = xs[:, 0]
    z = xs[:, 1]
    fig = plt.figure()
    # 直接使用 projection='3d' 创建 3D 坐标轴
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, z, y)

    x = np.arange(np.min(x), np.max(x), 0.01)
    z = np.arange(np.min(z), np.max(z), 0.01)
    x, z = np.meshgrid(x, z)
    y = forward_propgation(x, z)

    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()

