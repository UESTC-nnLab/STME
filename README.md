# SMFNet
The Pytorch implementation of the our paper of UESTC-nnLab [**A Simple and Effective Multi-Scale Time Fusion Network for Moving Infrared Dim-small Target Detectio**]
![outline](./readme/method.png)

## Abstract
The detection of infrared dim-small targets has become a challenging and hot topic in recent years. Due to the
extremely small pixels and low intensity of infrared dim target, the existing methods based on single frame seldom considers
the temporal relationship between frames, thus resulting in poor performance in detecting infrared moving objects. Recently,
several works based on temporal-domain information modeling are proposed to solve the problem of infrared dim-small tar-get detection, mainly relies on improved Conv-LSTM or self-attention to aggregate features from adjacent frames. However, unlike general objects, infrared dim and small target detection suffers from an overwhelming amount of background informa-tion, directly employing inter-frame attention mechanisms or Conv-LSTM would introduce excessive interfering information
and computational load. To promote infrared dim-small target detection, this paper explores a method of 2D-3D spatio-temporal
features fusion, which is prevailing in video domain. Simple but effective, we propose a 2D-3D Fusion Network based on Spatio-
Temporal Feature Fusion principles. It consists of two parallel feature extraction branches, one is designed to extract temporal
features across the entire video clip and the other is designed to isolate spatial features from a selected key frame. Following
this, a self-refine module comes into play, adaptively adjusting the extracted temporal features and convolutional feature fusion techniques complemented with channel-wise self-attention are adopted to fuse spatio-temporal features at different scales, enhancing deep feature representation. Extensive experiments validate this method simple yet effective on various infrared datasets including IRDST and ITDST. Without any bells and whistles, our method reduces FLOPs by approximately 65.98\%,
enables faster inference speeds, and achieves higher accuracy in comparison to current SOTA.
## Datasets
- You can download them directly from the website: [ITSDT-15K](https://www.scidb.cn/en/detail?dataSetId=de971a1898774dc5921b68793817916e&dataSetType=journal), [IRDST](https://xzbai.buaa.edu.cn/datasets.html).
## Results

- PR curve on DAUB and IRDST datasets.
- We provide the results on [ITSDT-15K](./readme/ITDST_results) and [IRDST](./readme/IRDST_results), and you can plot them using Python.

<img src="/readme/ITSDT_PR.jpg" width="500px">
<img src="/readme/IRDST_PR.jpg" width="500px">