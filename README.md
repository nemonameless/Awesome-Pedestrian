# Awesome-Pedestrian
Collection of online resources about pedestrian.

## Datasets/SOTA

1. Pedestrian Detection
     - [CityPersons (2.975k)](https://bitbucket.org/shanshanzhang/citypersons/src/default/)
     - [Caltech (42.782k)](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
     - [CrowdHuman (15k)](https://www.objects365.org/workshop2019.html)
     - [KITTI (3.712k)](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
     - [COCOPersons (64.115k)](http://cocodataset.org)
     - [OCHuman (4.731k)](http://www.liruilong.cn/projects/pose2seg/index.html)
     - [WiderPerson (13.382k)](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/)
     - [INRIA](http://pascal.inrialpes.fr/data/human/)
     - [NICTA](https://research.csiro.au/data61/automap-datasets-and-code/)
     - [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
     - [CUHK Occlusion Dataset](http://mmlab.ie.cuhk.edu.hk/datasets/cuhk_occlusion/index.html)
     - [CUHK Square Dataset](http://mmlab.ie.cuhk.edu.hk/datasets/cuhk_square/index.html)
     - [BIWI Walking Pedestrians dataset](http://www.vision.ee.ethz.ch/en/datasets/)
     - [Central Pedestrian Crossing Sequences](http://www.vision.ee.ethz.ch/en/datasets/)
     - [TUD](http://tahiti.mis.informatik.tu-darmstadt.de/oldmis/People/cwojek/tud-brussels-data-set/index.html)
     - [KAIST multispectral datase]()

2. Some SOTAs
    - [Caltech](https://www.paperswithcode.com/sota/pedestrian-detection-on-caltech)        
    - [CityPersons](https://www.paperswithcode.com/sota/pedestrian-detection-on-citypersons)

3. Pedestrian Attribute


## Challenges/Competitions
- [WIDER Face & Person Challenge](https://competitions.codalab.org/competitions/20132)
- [MOTChallenge:Detection in Crowded Scenes](https://motchallenge.net/workshops/bmtt2019/detection.html)
- [CrowdHuman](https://www.objects365.org/workshop2019.html)

## Evaluation Metric
1. Detection(Crowd Detection)
   - mAP
   - MR (Reasonable)	
   - MR (Reasonable_small)	
   - MR (Reasonable_occ=heavy)	
   - MR (All)

## Opensource Projects

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [CSP CVPR19: pytorch](https://github.com/lw396285v/CSP-pedestrian-detection-in-pytorch) [keras](https://github.com/liuwei16/CSP)
- [Bi-box_Regression ECCV18](https://github.com/rainofmine/Bi-box_Regression)
- [ALFNet ECCV18](https://github.com/liuwei16/ALFNet)
- [Repulsion_Loss CVPR18](https://github.com/rainofmine/Repulsion_Loss)
- [SDS-RCNN ICCV17](https://github.com/garrickbrazil/SDS-RCNN)

## Online Resources
- [Pedestrian-Detection](https://github.com/xingkongliang/Pedestrian-Detection)
- [awesome-pedestrian-detection](https://github.com/anjali-chadha/awesome-pedestrian-detection)
- [Pedestrian-Attribute-Recognition-Paper-List](https://github.com/wangxiao5791509/Pedestrian-Attribute-Recognition-Paper-List)
- [human_papernotes](https://github.com/DuinoDu/human_papernotes)


## Papers & Documents
### Crowd/Occlusion
#### CVPR 2019
- [Pedestrian Detection With Autoregressive Network Phases](http://openaccess.thecvf.com/content_CVPR_2019/papers/Brazil_Pedestrian_Detection_With_Autoregressive_Network_Phases_CVPR_2019_paper.pdf)[[code](https://github.com/garrickbrazil/AR-Ped)]
- [High-Level Semantic Feature Detection: A New Perspective for Pedestrian Detection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.pdf) [[code](https://github.com/liuwei16/CSP)]
- [Adaptive NMS: Refining Pedestrian Detection in a Crowd](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf)

#### ECCV 2018
- [Occlusion-aware R-CNN:Detecting Pedestrians in a Crowd](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)
- [Bi-box Regression for Pedestrian Detection and Occlusion Estimation](http://openaccess.thecvf.com/content_ECCV_2018/papers/CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper.pdf)
- [Graininess-Aware Deep Feature Learning for Pedestrian Detection](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chunze_Lin_Graininess-Aware_Deep_Feature_ECCV_2018_paper.pdf)
- [Small-scale Pedestrian Detection Based on Topological Line Localization and Temporal Feature Aggregation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.pdf)
- [Learning Efficient Single-stage Pedestrian Detectors by Asymptotic Localization Fitting](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Liu_Learning_Efficient_Single-stage_ECCV_2018_paper.pdf)[[code](https://github.com/VideoObjectSearch/ALFNet)]

#### CVPR 2018
- [Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors](http://openaccess.thecvf.com/content_cvpr_2018/papers/Noh_Improving_Occlusion_and_CVPR_2018_paper.pdf)
- [Occluded Pedestrian Detection Through Guided Attention in CNNs](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Occluded_Pedestrian_Detection_CVPR_2018_paper.pdf)
- [Repulsion Loss: Detecting Pedestrians in a Crowd](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)
- [WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chavdarova_WILDTRACK_A_Multi-Camera_CVPR_2018_paper.pdf)


#### ICCV 2017
- [Multi-label Learning of Part Detectors for Heavily Occluded Pedestrian Detection](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Multi-Label_Learning_of_ICCV_2017_paper.pdf)
- [Illuminating Pedestrians via Simultaneous Detection & Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Brazil_Illuminating_Pedestrians_via_ICCV_2017_paper.pdf)

#### CVPR 2017
- [CityPersons: A Diverse Dataset for Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)
- [Learning Cross-Modal Deep Representations for Robust Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Learning_Cross-Modal_Deep_CVPR_2017_paper.pdf)
- [What Can Help Pedestrian Detection?](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf)
- [Self-learning Scene-specific Pedestrian Detectors using a Progressive Latent Model](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ye_Self-Learning_Scene-Specific_Pedestrian_CVPR_2017_paper.pdf)
- [Expecting the Unexpected:Training Detectors for Unusual Pedestrians with Adversarial Imposters](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Expecting_the_Unexpected_CVPR_2017_paper.pdf)

#### ECCV-2016
- [Is Faster R-CNN Doing Well for Pedestrian Detection?](http://arxiv.org/abs/1607.07032v2)[code](https://github.com/zhangliliang/RPN_BF/tree/RPN-pedestrian)

#### CVPR-2016
- [Semantic Channels for Fast Pedestrian Detection](http://openaccess.thecvf.com/content_cvpr_2016/papers/Costea_Semantic_Channels_for_CVPR_2016_paper.pdf)
- [How Far are We from Solving Pedestrian Detection?](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_How_Far_Are_CVPR_2016_paper.pdf)
- [Pedestrian Detection Inspired by Appearance Constancy and Shape Symmetry](http://openaccess.thecvf.com/content_cvpr_2016/papers/Cao_Pedestrian_Detection_Inspired_CVPR_2016_paper.pdf)


#### Arxiv-19
- [Pedestrian Detection in Thermal Images using Saliency Maps - CVPR Workshop](https://arxiv.org/abs/1904.06859v1)
- [SSA-CNN: Semantic Self-Attention CNN for Pedestrian Detection](https://arxiv.org/abs/1902.09080v3)
- [Distant Pedestrian Detection in the Wild using Single Shot Detector with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1905.12759v1)
- [Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pedestrian Detection](https://arxiv.org/abs/1902.05291v1)
- [The Cross-Modality Disparity Problem in Multispectral Pedestrian Detection](https://arxiv.org/abs/1901.02645v1)
- [GFD-SSD: Gated Fusion Double SSD for Multispectral Pedestrian Detection](https://arxiv.org/abs/1903.06999)
- [WIDER Face and Pedestrian Challenge 2018: Methods and Results](https://arxiv.org/abs/1902.06854)
- [Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pedestrian Detection](https://arxiv.org/abs/1902.05291)
- [FPN++: A Simple Baseline for Pedestrian Detection - ICME 2019](https://ieeexplore.ieee.org/document/8784794)
##### IEEE Access
- [Learning Pixel-Level and Instance-Level Context-Aware Features for Pedestrian Detection in Crowds](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8763938)
- [Deep Feature Fusion by Competitive Attention for Pedestrian Detection](https://ieeexplore.ieee.org/document/8629899)
- [See Extensively While Focusing on the Core Area for Pedestrian Detection](https://ieeexplore.ieee.org/document/8651292)
- [Single Shot Multibox Detector With Kalman Filter for Online Pedestrian Detection in Video](https://ieeexplore.ieee.org/document/8631151)


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
- [ZoomNet: Deep Aggregation Learning for High-Performance Small Pedestrian Detection - ACML 2018](https://www.semanticscholar.org/paper/ZoomNet%3A-Deep-Aggregation-Learning-for-Small-Shang-Ai/35c473bae9d146072625cc3d452c8f6b84c8cc47)

#### Arxiv-17
- [Scene-Specific Pedestrian Detection Based on Parallel Vision](https://arxiv.org/abs/1712.08745)
- [Too Far to See? Not Really! --- Pedestrian Detection with Scale-aware Localization Policy - TIM 2017](https://arxiv.org/abs/1709.00235)
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
- [Detector-in-Detector: Multi-Level Analysis for Human-Parts - ACCV 2018](https://arxiv.org/abs/1902.07017)
- [Attribute Aware Pooling for Pedestrian Attribute Recognition - IJCAI 2019](https://arxiv.org/abs/1907.11837)
- [Pedestrian Attribute Recognition: A Survey](https://arxiv.org/abs/1901.07474v1)

#### CVPR 2017
- [HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_HydraPlus-Net_Attentive_Deep_ICCV_2017_paper.pdf)
