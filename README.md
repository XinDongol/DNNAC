# DNNAC
All about acceleration and compression of Deep Neural Networks

---------------------------------

### Quantization
#### General

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

* Data-Free Quantization through Weight Equalization and Bias Correction
  > Implementation: [Pytorch](https://github.com/ANSHUMAN87/Bias-Correction)
  
* Additive Noise Annealing and Approximation Properties of Quantized Neural Networks
  > Implementation: [Pytorch](https://github.com/spallanzanimatteo/QuantLab)
* Towards Learning of Filter-Level Heterogeneous Compression of Convolutional Neural Networks
  > find optimal bit-width with NAS
  >
  > Implementation: [Pytorch](https://github.com/yochaiz/Slimmable)
* Progressive Stochastic Binarization of Deep Networks
  > Use power-of-2
  >
  > Implementation: [TF](https://github.com/JGU-VC/progressive_stochastic_binarization)
* Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks
  > how to find the optimal threshold
  >
  > Implementation: [TF](https://github.com/Xilinx/graffitist)
* FAT: Fast Adjustable Threshold for Uniform Neural Network Quantization (Winning Solution on LPIRC-II)
  > Implementation: [TF](https://github.com/agoncharenko1992/FAT-fast-adjustable-threshold)
* Proximal Mean-field for Neural Network Quantization
  > Implementation: [Pytorch](https://github.com/tajanthan/pmf)
* A Survey on Methods and Theories of Quantized Neural Networks
  > Nice survey on quantization (up to Dec. 2018)

##### Binary
* Balanced Binary Neural Networks with Gated Residual
* IR-Net: Forward and Backward Information Retention for Highly Accurate Binary Neural Networks
#### Application-oriented
##### NLP
* Differentiable Product Quantization for Embedding Compression
  > compress the embedding table with end-to-end learned KD codes via differentiable product quantization (DPQ)
  >
  > Implementation: [TF](https://github.com/chentingpc/dpq_embedding_compression)
##### Adversarial
* Model Compression with Adversarial Robustness: A Unified Optimization Framework
  > This paper studies model compression through a different lens: could we compress models without hurting their robustness to adversarial attacks, in addition to maintaining accuracy?
  >
  > Implementation: [Pytorch](https://github.com/shupenggui/ATMC)
### Pruning

* Learning both Weights and Connections for Efficient Neural Networks
  > A very simple way to introduce arbitrary sparisity. 
  
* Learning Structured Sparsity in Deep Neural Networks
  > An united way to introduce structured sparsity.
  >
  > Implementation: [Caffe](https://github.com/wenwei202/caffe/tree/scnn)
  
### Neural Architecture Search (NAS)
* Resource
  1. [automl.org](https://www.automl.org/automl/literature-on-neural-architecture-search/)
* Partial Channel Connections for Memory-Efficient Differentiable Architecture Search
  > Our approach is memory efficient:(i) batch-size is increased to further accelerate the search on CIFAR10, (ii) directly search on ImageNet.
  > Searched on ImageNet, we achieved currently one of, if not only, the best performance on ImageNet (24.2%/7.3%) under the mobile setting!
  > The search process in CIFAR10 only requires 0.1 GPU-days, i.e., ~3 hours on one Nvidia 1080ti.(1.5 hours on one Tesla V100)
  > Implementation: [PyTorch](https://github.com/yuhuixu1993/PC-DARTS) (origin)

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


### Tools

#### Research
* [slimmable_networks](https://github.com/JiahuiYu/slimmable_networks)
  > An open source framework for slimmable training on tasks of ImageNet classification and COCO detection, which has enabled numerous projects.
* [distiller](https://github.com/NervanaSystems/distiller)
  > a Python package for neural network compression research
* [QPyTorch](https://github.com/Tiiiger/QPyTorch)
  > QPyTorch is a low-precision arithmetic simulation package in PyTorch. It is designed to support researches on low-precision machine learning, especially for researches in low-precision training.
* [Graffitist](https://github.com/Xilinx/graffitist)
  > Graffitist is a flexible and scalable framework built on top of TensorFlow to process low-level graph descriptions of deep neural networks (DNNs) for accurate and efficient inference on fixed-point hardware. It comprises of a (growing) library of transforms to apply various neural network compression techniques such as quantization, pruning, and compression. Each transform consists of unique pattern matching and manipulation algorithms that when run sequentially produce an optimized output graph.
#### Industry
* [dabnn](https://github.com/JDAI-CV/dabnn)
  > dabnn is an accelerated binary neural networks inference framework for mobile platform