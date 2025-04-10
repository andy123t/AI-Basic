# AI-Basic

B 站课程：[小白也能听懂的人工智能原理](https://www.bilibili.com/cheese/play/ep6911) 笔记

这个付费课程由 UP 主 “Ele实验室” 制作，专门为小白讲解人工智能原理。我试看了几集，讲解清晰生动，内容通俗易懂，果断以 76.8 元购入，感觉非常值。

学习的过程中，我在原有代码的基础上做了一些修改和优化。

**主要更改内容**：

- 3D 绘图：调用 `fig.add_subplot()`，使用 `projection='3d'` 参数创建 3D 坐标轴，不用导入 Axes3D 类。
- 不用 Keras 框架，使用 PyTorch 框架，重写程序及其相关函数模块 `plot_utils`。

