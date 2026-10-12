# STTGC-Net: Spatial-Temporal Transformer with Graph Convolution for Skeleton-Based Action Recognition
## Abstract
Skeleton data plays an important role in human action recognition due to the compact and distinct information of human poses provided by the skeleton data. Skeleton-based action recognition is gaining interest due to the availability of Kinect cameras at a reasonable price. 
With the growing popularity of Geometric deep learning, Graph Convolutional networks (GCN) are extensively used for processing skeleton data, due to their ability to model data topology. 
It has been found that Spatial-Temporal Graph Convolutional Networks (ST-GCN) are efficient in learning both spatial and temporal dependencies on non-Euclidean data, such as skeleton graphs.
However, the state-of-the-art ST-GCN models lack flexibility in feature extraction and do not explicitly consider the high-order Spatio-Temporal significance of the spatial connection topology and intensity of the joints. 
They lack an attention module that can help us learn together when and where to concentrate on a certain action in the action sequence. The transformer-based methods can effectively capture long-distance dependencies. 
However, using just a traditional transformer approach overlooks the graph structure in the data, which can provide valuable information about the interrelationships among the joint points during actions. 
To address this problem, we propose an architecture named Spatial-Temporal Transformer Network with Graph Convolution (STTGC-Net), that can flexibly capture local and global contexts. 
Spatial-Temporal Graph Convolutional Networks and Spatial-Temporal Transformer Attention Modules are sequentially fused to create the proposed framework. 
The ST-GCNs capture temporal dynamics, hierarchy, and local topological information at several levels, and the transformer attention module displays the correlations between joint pairs in global topology via dynamic attention, which resolves the mentioned constraints of the GCN and transformer. 
We validate our model by conducting tests on the NTU 60 and NTU 120 datasets, which achieve state-of-the-art performance.
