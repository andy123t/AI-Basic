import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def show_scatter_curve(X, Y, pres):
    plt.scatter(X, Y)
    plt.plot(X, pres)
    plt.show()


def show_scatter(X, Y):
    if X.ndim > 1:
        show_3d_scatter(X, Y)
    else:
        plt.scatter(X, Y)
        plt.show()


def show_3d_scatter(X, Y):
    x = X[:, 0]
    z = X[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, Y)
    plt.show()


def show_surface(x, z, forward_propagation):
    x = np.arange(np.min(x), np.max(x), 0.1)
    z = np.arange(np.min(z), np.max(z), 0.1)
    x, z = np.meshgrid(x, z)
    X = np.column_stack((x.flatten(), z.flatten()))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y = forward_propagation(X_tensor).detach().numpy()
    y = y.reshape(x.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()


def show_scatter_surface(X, Y, forward_propagation):
    if isinstance(forward_propagation, nn.Module):
        show_scatter_surface_with_model(X, Y, forward_propagation)
        return
    x = X[:, 0]
    z = X[:, 1]
    y = Y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, y)

    x = np.arange(np.min(x), np.max(x), 0.1)
    z = np.arange(np.min(z), np.max(z), 0.1)
    x, z = np.meshgrid(x, z)

    X = np.column_stack((x.flatten(), z.flatten()))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    r = forward_propagation(X_tensor)
    # 直接使用 r 作为 y，因为 r 已经是 numpy.ndarray 类型
    y = r
    y = y.reshape(x.shape)
    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()


def show_scatter_surface_with_model(X, Y, model):
    x = X[:, 0]
    z = X[:, 1]
    y = Y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, y)

    x = np.arange(np.min(x), np.max(x), 0.1)
    z = np.arange(np.min(z), np.max(z), 0.1)
    x, z = np.meshgrid(x, z)

    X = np.column_stack((x.flatten(), z.flatten()))
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y = model(X_tensor).detach().numpy()
    y = y.reshape(x.shape)
    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()


def pre(X, Y, model):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model(X_tensor)

