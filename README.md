# CVPR19 Incremental Learning
Thi repository is for the paper "Learning a Unified Classifier Incrementally via Rebalancing".

[[Papers]](http://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html) [[Project Page]](https://hshustc.github.io/CVPR19_Incremental_Learning/)

# Instructions
1. Dependencies
	- Python 3.6 (Anaconda3 Recommended)
	- Pytorch 0.4.0
2. Getting Started 
	- the data for CIFAR100 and ImageNet are put in `cifar100-class-incremental/data` and `imagenet-class-incremental/data`, or you can make soft links to the directories which include the corresponding data
	- see `cifar100-class-incremental/run.sh` for the experiments on CIFAR100
	- see `imagenet-class-incremental/run.sh` for the experiments on ImageNet-Subset
	- see `imagenet-class-incremental/run_all.sh` for the experiments on ImageNet-Full

# Citation
Please cite the following paper if you find this useful in your research:
```
@InProceedings{Hou_2019_CVPR,
author = {Hou, Saihui and Pan, Xinyu and Change Loy, Chen and Wang, Zilei and Lin, Dahua},
title = {Learning a Unified Classifier Incrementally via Rebalancing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```