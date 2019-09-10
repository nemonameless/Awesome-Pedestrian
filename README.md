# Awesome-Pedestrian
Collection of online resources about pedestrian.

## Datasets

1. Pedestrian Detection
     - [CityPersons](https://bitbucket.org/shanshanzhang/citypersons/src/default/)
     - [WiderPerson](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/) 13,382 images 
     - [Caltech]()
     - [KITTI]()
     - [ETH]()
     - [INRIA]()
     - [KAIST multispectral datase]()
        
2. Pedestrian Attribute

## Challenges

- [WIDER Face & Person Challenge](https://competitions.codalab.org/competitions/20132)
- [MOTChallenge:Detection in Crowded Scenes](https://motchallenge.net/workshops/bmtt2019/detection.html)

## Evaluation Metric
1. Detection(Crowd Detection)
   - mAP
   - MR (Reasonable)	
   - MR (Reasonable_small)	
   - MR (Reasonable_occ=heavy)	
   - MR (All)

## Opensource Projects

- [MMDetection](https://github.com/open-mmlab/mmdetection)

## Online Resources

- [Pedestrian-Attribute-Recognition-Paper-List](https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List)

## Papers & Documents
## Detection 
### Crowd/Occlusion
#### CVPR 2019
- [Pedestrian Detection With Autoregressive Network Phases - CVPR - 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Brazil_Pedestrian_Detection_With_Autoregressive_Network_Phases_CVPR_2019_paper.pdf)[[code](https://github.com/garrickbrazil/AR-Ped)]
- [High-Level Semantic Feature Detection: A New Perspective for Pedestrian Detection - CVPR - 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Point_in_Box_Out_Beyond_Counting_Persons_in_Crowds_CVPR_2019_paper.pdf) [[code](https://github.com/liuwei16/CSP)]
- [Adaptive NMS: Refining Pedestrian Detection in a Crowd - CVPR - 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf)

#### ECCV 2018
- [Occlusion-aware R-CNN:
Detecting Pedestrians in a Crowd](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)
- [Bi-box Regression for Pedestrian Detection and
Occlusion Estimation](http://openaccess.thecvf.com/content_ECCV_2018/papers/CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper.pdf)
- [Graininess-Aware Deep Feature Learning for
Pedestrian Detection](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chunze_Lin_Graininess-Aware_Deep_Feature_ECCV_2018_paper.pdf)
- [Small-scale Pedestrian Detection Based on
Topological Line Localization and Temporal
Feature Aggregation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.pdf)
- [Learning Efficient Single-stage Pedestrian
Detectors by Asymptotic Localization Fitting](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Liu_Learning_Efficient_Single-stage_ECCV_2018_paper.pdf)[[code](https://github.com/VideoObjectSearch/ALFNet)]

#### CVPR 2018
- [Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors](http://openaccess.thecvf.com/content_cvpr_2018/papers/Noh_Improving_Occlusion_and_CVPR_2018_paper.pdf)
- [WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.pdf)
- [Occluded Pedestrian Detection Through Guided Attention in CNNs](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Occluded_Pedestrian_Detection_CVPR_2018_paper.pdf)
- [Repulsion Loss: Detecting Pedestrians in a Crowd](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)

#### ICCV 2017
- [Multi-label Learning of Part Detectors for Heavily Occluded Pedestrian Detection](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Multi-Label_Learning_of_ICCV_2017_paper.pdf)
- [Illuminating Pedestrians via Simultaneous Detection & Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Brazil_Illuminating_Pedestrians_via_ICCV_2017_paper.pdf)

#### CVPR 2017
- [CityPersons: A Diverse Dataset for Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)
- [Learning Cross-Modal Deep Representations for Robust Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Learning_Cross-Modal_Deep_CVPR_2017_paper.pdf)
- [What Can Help Pedestrian Detection?](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf)
- [Self-learning Scene-specific Pedestrian Detectors
using a Progressive Latent Model](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ye_Self-Learning_Scene-Specific_Pedestrian_CVPR_2017_paper.pdf)
-[Expecting the Unexpected:
Training Detectors for Unusual Pedestrians with Adversarial Imposters](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Expecting_the_Unexpected_CVPR_2017_paper.pdf)

#### Arxiv-19
- [Pedestrian Detection in Thermal Images using Saliency Maps - CVPR Workshop](https://arxiv.org/abs/1904.06859v1)
- [SSA-CNN: Semantic Self-Attention CNN for Pedestrian Detection](https://arxiv.org/abs/1902.09080v3)
- [Distant Pedestrian Detection in the Wild using Single Shot Detector with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1905.12759v1)
- [Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pedestrian Detection](https://arxiv.org/abs/1902.05291v1)
- [The Cross-Modality Disparity Problem in Multispectral Pedestrian Detection](https://arxiv.org/abs/1901.02645v1)
- [GFD-SSD: Gated Fusion Double SSD for Multispectral Pedestrian Detection](https://arxiv.org/abs/1903.06999)
- [WIDER Face and Pedestrian Challenge 2018: Methods and Results](https://arxiv.org/abs/1902.06854)
- [Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pedestrian Detection](https://arxiv.org/abs/1902.05291)



#### Arxiv-18
- [Convolutional Neural Networks for Aerial Multi-Label PedestrianDetection](https://arxiv.org/abs/1807.05983v1)
- [Part-Level Convolutional Neural Networks for Pedestrian Detection Using Saliency and Boundary Box Alignment](https://arxiv.org/abs/1810.00689v1)
- [Pedestrian Detection with Autoregressive Network Phases](https://arxiv.org/abs/1812.00440v1)
- [Part-Level Convolutional Neural Networks for Pedestrian Detection Using Saliency and Boundary Box Alignment - ICASSP](https://arxiv.org/abs/1810.00689v1) [[code](https://github.com/iyyun/Part-CNN)]
- [Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation - BMVC 2018](https://arxiv.org/abs/1808.04818v1) [[code](https://github.com/Li-Chengyang/MSDS-RCNN)]
- [Disparity Sliding Window: Object Proposals from Disparity Images - IROS 2018](https://arxiv.org/abs/1805.06830v2)[[code](https://github.com/julimueller/disparity-sliding-window)]
- [An FPGA-Accelerated Design for Deep Learning Pedestrian Detection in Self-Driving Vehicles](https://arxiv.org/abs/1809.05879)
- [Real-time Pedestrian Detection Approach with an Efficient Data Communication Bandwidth Strategy](https://arxiv.org/abs/1808.09023)
- [SAM-RCNN: Scale-Aware Multi-Resolution Multi-Channel Pedestrian Detection](https://arxiv.org/abs/1808.02246)
- [A Content-Based Late Fusion Approach Applied to Pedestrian Detection](https://arxiv.org/abs/1806.03361)
- [Fused Deep Neural Networks for Efficient Pedestrian Detection](https://arxiv.org/abs/1805.08688)
- [PCN: Part and Context Information for Pedestrian Detection with CNN - BMVC 2017](https://arxiv.org/abs/1804.04483)
- [Exploring Multi-Branch and High-Level Semantic Networks for Improving Pedestrian Detection](https://arxiv.org/abs/1804.00872)
- [Illumination-aware Faster R-CNN for Robust Multispectral Pedestrian Detection - PR](https://arxiv.org/abs/1803.05347)
- [Fusion of Multispectral Data Through Illumination-aware Deep Neural Networks for Pedestrian Detection](https://arxiv.org/abs/1802.09972)
- [Aggregated Channels Network for Real-Time Pedestrian Detection](https://arxiv.org/abs/1801.00476)



#### Arxiv-17
- [Scene-Specific Pedestrian Detection Based on Parallel Vision](https://arxiv.org/abs/1712.08745)
- [Too Far to See? Not Really! --- Pedestrian Detection with Scale-aware Localization Policy](https://arxiv.org/abs/1709.00235)
- [Rotational Rectification Network: Enabling Pedestrian Detection for Mobile Vision](https://arxiv.org/abs/1706.08917)
- [MixedPeds: Pedestrian Detection in Unannotated Videos using Synthetically Generated Human-agents for Training](https://arxiv.org/abs/1707.09100)
- [Comparing Apples and Oranges: Off-Road Pedestrian Detection on the NREC Agricultural Person-Detection Dataset](https://arxiv.org/abs/1707.07169)




#### Arxiv-16
- [A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection](https://arxiv.org/abs/1607.07155)



## Trajectory
#### CVPR 2019
- [SR-LSTM: State Refinement for LSTM Towards Pedestrian Trajectory Prediction](https://arxiv.org/abs/1801.00868)
#### CVPR 2018
- [Encoding Crowd Interaction With Deep Neural Network for Pedestrian Trajectory Prediction](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)
#### CVPR 2017
- [Forecasting Interactive Dynamics of Pedestrians with Fictitious Play](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ma_Forecasting_Interactive_Dynamics_CVPR_2017_paper.pdf)


## Counting
#### CVPR 2019
- [Point in, Box out: Beyond Counting Persons in Crowds](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Point_in_Box_Out_Beyond_Counting_Persons_in_Crowds_CVPR_2019_paper.pdf)
#### Arxiv-19
- [Dynamic Region Division for Adaptive Learning Pedestrian Counting - ICME 2019](https://arxiv.org/abs/1908.03978)

## Attribute/Analysis
#### Arxiv-19
- [Detector-in-Detector: Multi-Level Analysis for Human-Parts ACCV 2018](https://arxiv.org/abs/1902.07017)
- [Attribute Aware Pooling for Pedestrian Attribute Recognition - IJCAI 2019](https://arxiv.org/abs/1907.11837)

#### CVPR 2017
- [HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_HydraPlus-Net_Attentive_Deep_ICCV_2017_paper.pdf)
