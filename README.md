# Unsupervised-Capsule-Network
Deep Capsule networks with a few variations. Originally proposed by Tieleman &amp; Hinton (http://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf)

### Description
Given an input image, a convolutional neural network (encoder) predicts a set of structure variables corresponding to transformation matrices (total number ```T``` is pre-specified). Each matrix has 7 values of interest -- scale (x,y), translation (x,y), shear, rotation and intensity. A separate set of ```T``` weight matrices (NxN, N=10) are also learnt and they denote the entities or parts. These entities do not get an input, so they can be thought of as persistent memory units. Given the transformation matrix for each entity, the entity can be placed into the global image space by a bunch of matrix computations. The final image is produced by combining contributions (aggregator) from all the entities. This is the decoder (entities+transformation matrices+aggregator). The entire model can then be back-propagated end-to-end via gradient descent.  

For example, here's the model on MNIST test (not ran until convergence). The total number of entities in the decoder was set to be 6 and 20 in two different experiments. As seen below, the encoder does the necessary deformations to use only the available number of entities. ```It discovers strokes!```

#### 6 Entities in decoder (left is the original image, middle columns are the 6 transformed entities per image, right is the reconstructd image)
Note: not ran until convergence

![alt-text](https://github.com/mrkulk/Unsupervised-Capsule-Network/blob/master/capsule_6.png "6 Entities in decoder") 

#### 20 Entities in decoder (left is the original image, middle columns are the 20 transformed entities per image, right is the reconstructd image)

![alt-text](https://github.com/mrkulk/Unsupervised-Capsule-Network/blob/master/capsule_20.png "20 Entities in decoder")

### Requirements
Other than normal torch dependencies (GPU required), please install the stn module from here: http://gitxiv.com/posts/5WTXTLuEA4Hd8W84G/spatial-transformer-networks

### Usage
1. Start training the model: qlua main.lua --plot
2. Execute eval.lua to dump results on test set
3. For further analysis (classification), see the ipython notebook which takes the features dumped by eval.lua

### Known Issues
The intensity computation needs to be fixed as it does not saturate to near 1. That is why reconstructions systematically have lower intensity. This is probably due to the log computations in the intensity module. 
