

## nvcr: nvidia-container-runtime

- runtime的含义：
runtime这个词是因为它提供了一个容器运行时环境。
简单来说:
-   容器是离散的,它离开运行时就没有意义
-   运行时提供了一个环境,使容器能运行起来
更具体点:
-   运行时提供内核支持、网络栈、 rootfs(文件系统)等基础设施,使容器能运行
-   容器依赖运行时,运行时为容器提供运行所需的一切
-   不同的运行时有不同的功能,为不同类型的容器提供不同的运行能力
因此,NVIDIA称其产品为nvidia-container-runtime,的确恰当:
-   它提供了一个能使容器访问GPU的运行时环境
-   让依赖GPU的容器(如深度学习容器)得以运行
-   为GPU容器提供了独立的网络栈、rootfs、GPU驱动等
总的来说:
运行时(runtime)这个词突出表明它提供了一个环境,在这个环境中,容器才得以运行。
而nvidia-container-runtime具体提供的是一个容器可访问主机GPU的运行时环境。



## ddp可用的容器
- `docker run --gpus all -itd --name='torch1.13' --ipc=host  -v /home/shenw/LLM:/LLM -p 10024:22 -p 10025:10025 -p 10026:10026 nvcr.io/nvidia/pytorch:22.07-py3`
- 关键参数<mark style="background: #FF5582A6;"> --ipc=host</mark> 要指定(Inter-Process Communication ,进程间通信)命名空间。用于在容器间或容器与宿主机之间共享 IPC 资源。
- 可选： --ipc=host --ulimit memlock=-1 --ulimit stack=67108864


## 资源
- nvcr: [pytorch环境镜像](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags)
- NGC: https://catalog.ngc.nvidia.com/containers


