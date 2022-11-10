[SPCBPT: Subspace-based Probabilistic Connections for Bidirectional Path Tracing]([SPCBPT (ssufujia.github.io)](https://ssufujia.github.io/SPCBPT/)) 的一个OptiX 7实现。

### 环境

* OptiX 7.5.0
* Cuda 11.7
* Visual Studio 2019  
* Cmake 3.24.2

### 构建方法

* 打开cmake gui
* 指定src文件夹为源代码路径
* 创建build目录
* 点击"Configure"并选择VS2019 x64平台
* 点击“Finish”以确认配置
* 之后可能会因为找不到OPTIX的安装路径而报错，此时将OptiX_INSTALL_DIR设为安装OPTIX7的路径，比如C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.5.0，并再次Configure
* 点击“generate”生成项目并点击"open Project"打开项目
* 右键点击optixPathTracer并将其设为启动项，以Release形式编译运行

### 与论文版本的代码的差异

由于种种原因，目前该实现的实现细节并非完全与论文中的一致，在某些地方有细微的差别，包括：

* t=1，即光子路径直接连接摄像头的策略并非被实现，因为一般而言这一策略都过于低效。
* 光子路的跨帧重用、环境光照和透明材质目前仍未完成，将在未来补全。
* 子空间分类时并非考虑入射方向因素，为了提供更好的分类表现，在目前的场景中我们需要把分类决策树的精度都给到位置和法线上。
* 子空间采样矩阵并非从均匀矩阵中训练而成，而是从一个初始的根据子空间之间的路径的贡献值积分来构建的矩阵上再进一步迭代训练，这能够加快训练的速度.
* 简单起见，追踪训练集使用的算法为单向路径追踪+NEE，而非论文中的BDPT。
* 目前的算法表现在某些地方的亮噪点要比我的论文版本的代码稍微多一点点，我会在未来找到这一问题的解答并改正。