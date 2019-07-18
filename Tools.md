## Fast Training
1. [APEX](https://github.com/NVIDIA/apex)

   > A PyTorch Extension from NVIDIA: Tools for easy mixed precision (Automatic Mixed Precision, and FP16) and distributed training (DistributedDataParallel, Sync BN) in Pytorch

2. [Horovod](https://github.com/uber/horovod#running-horovod)

    > Horovod is a distributed training framework for TensorFlow, Keras, and PyTorch. The goal of Horovod is to make distributed Deep Learning fast and easy to use. Use baidu's RingAllReduce.
    
    > While installing MPI and NCCL itself may seem like an extra hassle, it only needs to be done once by the team dealing with infrastructure, while everyone else in the company who builds the models can enjoy the simplicity of training them at scale.
    
3. [BytePS](https://github.com/bytedance/byteps)
   > BytePS is a high performance and general distributed training framework. It supports TensorFlow, Keras, PyTorch, and MXNet, and can run on either TCP or RDMA network.
   
   > BytePS outperforms existing open-sourced distributed training frameworks by a large margin. For example, on a popular public cloud and with the same number of GPUs, BytePS can double the training speed (see below), compared with Horovod+NCCL.

## Fast Layers
1. [Pytorch-extension](https://github.com/sniklaus/pytorch-extension)
    > This is an example of a CUDA extension for PyTorch which uses CuPy to compute the Hadamard product of two tensors.

    > For a more advanced PyTorch extension that uses CuPy as well, please see: https://github.com/szagoruyko/pyinn

## Viz
1. Netron: [Github](https://github.com/lutzroeder/netron), [Brower](https://lutzroeder.github.io/netron/)

   > Netron is a interactive viewer for neural network, deep learning and machine learning models. You can use it without writing any code.
   > The input is model files like `.pth`, `.onnx` and so on.
   
2. [HiddenLayer](https://github.com/waleedka/hiddenlayer):

    > A lightweight library for neural network graphs and training metrics for PyTorch, Tensorflow, and Keras.

## Profile NN
1. [TorchStat](https://github.com/Swall0w/torchstat):
   
   > A lightweight neural network analyzer based on PyTorch: **#parameters, FLOPs, #MAC, Memory usage**.
   
2. [Pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter):

   > Count the FLOPs of your PyTorch model.

3. [torchsummary](https://github.com/sksq96/pytorch-summary)
   > Keras style model.summary() in PyTorch. It is useful to check output size of each layer.
   
4. [Pytorch-Memory-Utils](https://github.com/Oldpan/Pytorch-Memory-Utils)
   > A tool can help you to detect your GPU memory during training with Pytorch.
   
5. [torchprof](https://github.com/awwong1/torchprof)
   > All metrics are derived using the PyTorch autograd profiler. It will give you **Self CPU total | CPU total | CUDA total**.
