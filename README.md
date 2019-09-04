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
##### CVPR 2019
- [Pedestrian Detection With Autoregressive Network Phases - CVPR - 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Brazil_Pedestrian_Detection_With_Autoregressive_Network_Phases_CVPR_2019_paper.pdf)[[code](https://github.com/garrickbrazil/AR-Ped)]
- [High-Level Semantic Feature Detection: A New Perspective for Pedestrian Detection - CVPR - 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Point_in_Box_Out_Beyond_Counting_Persons_in_Crowds_CVPR_2019_paper.pdf) [[code](https://github.com/liuwei16/CSP)]
- [Adaptive NMS: Refining Pedestrian Detection in a Crowd - CVPR - 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf)

##### ECCV 2018
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

##### CVPR 2018
- [Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors](http://openaccess.thecvf.com/content_cvpr_2018/papers/Noh_Improving_Occlusion_and_CVPR_2018_paper.pdf)
- [WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.pdf)
- [Occluded Pedestrian Detection Through Guided Attention in CNNs](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Occluded_Pedestrian_Detection_CVPR_2018_paper.pdf)
- [Repulsion Loss: Detecting Pedestrians in a Crowd](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)

##### ICCV 2017
- [Multi-label Learning of Part Detectors
for Heavily Occluded Pedestrian Detection](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Multi-Label_Learning_of_ICCV_2017_paper.pdf)
- [Illuminating Pedestrians via Simultaneous Detection & Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Brazil_Illuminating_Pedestrians_via_ICCV_2017_paper.pdf)  

##### CVPR 2017
- [CityPersons: A Diverse Dataset for Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)
- [Learning Cross-Modal Deep Representations for Robust Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Learning_Cross-Modal_Deep_CVPR_2017_paper.pdf)
- [What Can Help Pedestrian Detection?](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf)
- [Self-learning Scene-specific Pedestrian Detectors
using a Progressive Latent Model](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ye_Self-Learning_Scene-Specific_Pedestrian_CVPR_2017_paper.pdf)
-[Expecting the Unexpected:
Training Detectors for Unusual Pedestrians with Adversarial Imposters](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Expecting_the_Unexpected_CVPR_2017_paper.pdf)

##### arxiv-19
- [Pedestrian Detection in Thermal Images using Saliency Maps - cvpr workshop](https://arxiv.org/abs/1904.06859v1)
##### arxiv-18
- [Part-Level Convolutional Neural Networks for Pedestrian Detection Using Saliency and Boundary Box Alignment - ICASSP](https://arxiv.org/abs/1810.00689v1) [[code](https://github.com/iyyun/Part-CNN)]
- [Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation - BMVC 2018](https://arxiv.org/abs/1808.04818v1) [[code](https://github.com/Li-Chengyang/MSDS-RCNN)]
- [Disparity Sliding Window: Object Proposals from Disparity Images - IROS 2018](https://arxiv.org/abs/1805.06830v2)[[code](https://github.com/julimueller/disparity-sliding-window)]
##### arxiv-17

##### arxiv-16
- [A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection](https://arxiv.org/abs/1607.07155)



## Trajectory
##### CVPR 2019
- [SR-LSTM: State Refinement for LSTM Towards Pedestrian Trajectory Prediction](https://arxiv.org/abs/1801.00868)
##### CVPR 2018
- [Encoding Crowd Interaction With Deep Neural Network for Pedestrian Trajectory Prediction](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)
##### CVPR 2017
- [Forecasting Interactive Dynamics of Pedestrians with Fictitious Play](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ma_Forecasting_Interactive_Dynamics_CVPR_2017_paper.pdf)


## Counting
##### CVPR 2019
- [Point in, Box out: Beyond Counting Persons in Crowds](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Point_in_Box_Out_Beyond_Counting_Persons_in_Crowds_CVPR_2019_paper.pdf)


## Attribute/Analysis
##### CVPR 2017
- [HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_HydraPlus-Net_Attentive_Deep_ICCV_2017_paper.pdf)
