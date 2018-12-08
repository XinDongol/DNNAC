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
  
### Pruning

* Learning both Weights and Connections for Efficient Neural Networks
  > A very simple way to introduce arbitrary sparisity. 
  
* Learning Structured Sparsity in Deep Neural Networks
  > An united way to introduce structured sparsity.
  >
  > Implementation: [Caffe](https://github.com/wenwei202/caffe/tree/scnn)

### Others

* Net2Net : Accelerating Learning via Knowledge Transfer
  > An interesting way to change the architecture of models while keeping output the same
  > 
  > Implementation: [TF](https://github.com/paengs/Net2Net), [Pytorch](https://github.com/erogol/Net2Net)
