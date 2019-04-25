# DNNAC
All about acceleration and compression of Deep Neural Networks

---------------------------------

### Quantization

* XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks

  > A classic paper for binary neural network saying all weights and activation are binarized.
  > 
  > Implementation: [MXNet](https://github.com/hpi-xnor/BMXNet-v2), [Pytorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch), [Torch](https://github.com/allenai/XNOR-Net) (origin)
  
* DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
  > Full stack quantization for weights, activation and gradient.
  > 
  > Implementation: [Tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net)
  
* Deep Learning with Low Precision by Half-wave Gaussian Quantization
  > Try to improve expersiveness of quantized activation function. 
  >
  > Implementation: [Caffe](https://github.com/zhaoweicai/hwgq) (origin)
  
* Quantizing deep convolutional networks for efficient inference: A whitepaper 
  > Non-official technical report of quantization from Google. You can find a lot of technical details about quantization in this paper. 
  
* A Survey on Methods and Theories of Quantized Neural Networks
  > Nice survey on quantization (up to Dec. 2018)
  
### Pruning

* Learning both Weights and Connections for Efficient Neural Networks
  > A very simple way to introduce arbitrary sparisity. 
  
* Learning Structured Sparsity in Deep Neural Networks
  > An united way to introduce structured sparsity.
  >
  > Implementation: [Caffe](https://github.com/wenwei202/caffe/tree/scnn)

### Others
* Benchmark Analysis of Representative Deep Neural Network Architectures [IEEE Access, University of Milano-Bicocca]
  > This work presents an in-depth analysis of the majority of the deep neural networks (DNNs) proposed in the state of the art for image recognition in terms of GFLOPs, #weights, Top-1 accuacy and so on.
  
* Net2Net : Accelerating Learning via Knowledge Transfer
  > An interesting way to change the architecture of models while keeping output the same
  > 
  > Implementation: [TF](https://github.com/paengs/Net2Net), [Pytorch](https://github.com/erogol/Net2Net)


### Embedded System

* [EMDL](https://github.com/EMDL/awesome-emdl): Embedded and mobile deep learning research notes
  > Embedded and mobile deep learning research notes on Github
