# Unsupervised-Capsule-Network
Deep Capsule networks with a few variations. Originally proposed by Tieleman &amp; Hinton (http://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf)

### Description
Given an input image, a convolutional neural network predicts a set of structure variables corresponding to transformation matrices (total number ```T``` is pre-specified). Each matrix has 7 values of interest -- scale (x,y), translation (x,y), shear, rotation and intensity. A separate set of ```T``` weight matrices (NxN, N=10) are also learnt and they denote the entitiy or parts. These entities do not get an input, so they can be thought of as persistent memory units. Given the transformation matrix for each entity, the entity can be placed into the global image space by a bunch of matrix computations. The final image is produced by combining contributions from all the entities. The entire model can then be back-propagated end-to-end via gradient descent.  

### Requirements
Other than normal torch dependencies (GPU required), please install the stn module from here: http://gitxiv.com/posts/5WTXTLuEA4Hd8W84G/spatial-transformer-networks
### Usage
1. Start training the model: qlua main.lua --plot
2. Execute eval.lua to dump results on test set
3. For further analysis (classification), see the ipython notebook which takes the features dumped by eval.lua
