# [AAA: Adaptive Aggregation of Arbitrary Online Trackers with Theoretical Performance Guarantee](https://arxiv.org/abs/2009.09237)

![Figure 1](assets/Fig1.png?raw=true "Score")

Heon Song, Daiki Suehiro, Seiichi Uchida

> For visual object tracking, it is difficult to realize an almighty online tracker due to the huge variations of target appearance depending on an image sequence. This paper proposes an online tracking method that adaptively aggregates arbitrary multiple online trackers. The performance of the proposed method is theoretically guaranteed to be comparable to that of the best tracker for any image sequence, although the best expert is unknown during tracking. The experimental study on the large variations of benchmark datasets and aggregated trackers demonstrates that the proposed method can achieve state-of-the-art performance.

## Experts

In this repository, we implemented or edited the following trackers to use as experts.  
**You can use the trackers with just a few lines of code.**

| Tracker   | Link                |
|-----------|---------------------|
| ATOM (CVPR 2019)      | [Paper](https://arxiv.org/abs/1811.07628) / [Original Repo](https://github.com/visionml/pytracking) |
| DaSiamRPN (ECCV 2018) | [Paper](https://arxiv.org/abs/1808.06048) / [Original Repo](https://github.com/foolwood/DaSiamRPN) |
| DiMP (ICCV 2019)      | [Paper](https://arxiv.org/abs/1904.07220) / [Original Repo](https://github.com/visionml/pytracking) |
| DROL (AAAI 2020)      | [Paper](https://arxiv.org/abs/1909.02959) / [Original Repo](https://github.com/shallowtoil/DROL) |
| GradNet (ICCV 2019)   | [Paper](https://arxiv.org/abs/1909.06800) / [Original Repo](https://github.com/LPXTT/GradNet-Tensorflow) |
| KYS (ECCV 2020)   | [Paper](https://arxiv.org/abs/2003.11014) / [Original Repo](https://github.com/visionml/pytracking) |
| MemDTC (TPAMI 2019)  | [Paper](https://arxiv.org/abs/1907.07613) / [Original Repo](https://github.com/skyoung/MemTrack) |
| MemTrack (ECCV 2018)  | [Paper](https://arxiv.org/abs/1803.07268) / [Original Repo](https://github.com/skyoung/MemTrack) |
| Ocean (ECCV 2020)     | [Paper](https://arxiv.org/abs/2006.10721) / [Original Repo](https://github.com/researchmm/TracKit) |
| PrDiMP (CVPR 2020)    | [Paper](https://arxiv.org/abs/2003.12565) / [Original Repo](https://github.com/visionml/pytracking) |
| RLS-RTMDNet (CVPR 2020)    | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Recursive_Least-Squares_Estimator-Aided_Online_Learning_for_Visual_Tracking_CVPR_2020_paper.html) / [Original Repo](https://github.com/Amgao/RLS-RTMDNet) |
| SiamBAN (CVPR 2020)   | [Paper](https://arxiv.org/abs/2003.06761) / [Original Repo](https://github.com/hqucv/siamban) |
| SiamCAR (CVPR 2020)   | [Paper](https://arxiv.org/abs/1911.07241) / [Original Repo](https://github.com/ohhhyeahhh/SiamCAR) |
| SiamDW (CVPR 2019)    | [Paper](https://arxiv.org/abs/1901.01660) / [Original Repo](https://github.com/researchmm/SiamDW) |
| SiamFC (ECCVW 2016)    | [Paper](https://arxiv.org/abs/1606.09549) / [Original Repo](https://github.com/got-10k/siamfc) |
| SiamFC++ (AAAI 2020)  | [Paper](https://arxiv.org/abs/1911.06188) / [Original Repo](https://github.com/MegviiDetection/video_analyst) |
| SiamMCF (ECCVW 2018)   | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_6) / [Original Repo](https://github.com/hmorimitsu/siam-mcf) |
| SiamR-CNN (CVPR 2020) | [Paper](https://arxiv.org/abs/1911.12836) / [Original Repo](https://github.com/VisualComputingInstitute/SiamR-CNN) |
| SiamRPN (CVPR 2018)   | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) / [Original Repo](https://github.com/huanglianghua/siamrpn-pytorch) |
| SiamRPN++ (CVPR 2019) | [Paper](https://arxiv.org/abs/1812.11703) / [Original Repo](https://github.com/STVIR/pysot) |
| SPM (CVPR 2019)      | [Paper](https://arxiv.org/abs/1904.04452) / [Original Repo](https://github.com/microsoft/SPM-Tracker) |
| Staple (CVPR 2016)   | [Paper](https://arxiv.org/abs/1512.01355) / [Original Repo](https://github.com/wwdguu/pyCFTrackers) |
| THOR (BMVC 2019)     | [Paper](https://arxiv.org/abs/1907.12920) / [Original Repo](https://github.com/xl-sr/THOR) |

<sup>For DaSiamRPN, RLS-RTMDNet and SPM, we've modified the code a little bit to be compatible with Python3 and Pytorch >= 1.3.</sup>

## Datasets

We evaluated the performance of the experts and AAA on the following datasets.

* [OTB2015](https://ieeexplore.ieee.org/document/7001050)[<http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html>]
* [NFS](https://arxiv.org/abs/1703.05884)[<http://ci2cv.net/nfs/index.html>]
* [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)[<https://uav123.org/>]
* [TColor128](https://ieeexplore.ieee.org/document/7277070)[<http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html>]
* [TrackingNet](https://arxiv.org/abs/1803.10794)[<https://tracking-net.org/>]
* [VOT2018](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_1)[<http://www.votchallenge.net/>]
* [LaSOT](https://arxiv.org/abs/1809.07845)[<https://cis.temple.edu/lasot/download.html>]
* [Got10K](https://arxiv.org/abs/1810.11981)[<http://got-10k.aitestunion.com/>]

<sup>VOT2018 is evaluated in unsupervised experiment as same as other datasets.</sup>

## Frameworks

The following frameworks were used to conveniently track videos and evaluate trackers.

* pytracking[<https://github.com/visionml/pytracking>] for tracking datasets.
* pysot-toolkit[<https://github.com/StrangerZhang/pysot-toolkit>] for evaluating trackers.

## Requirements

We strongly recommend using a virtual environment like Anaconda or Docker.  
The following is how to build the virtual environment for AAA using anaconda.

```sh
# clone this repository
git clone https://github.com/songheony/AAA-journal
cd AAA-journal

# create and activate anaconda environment
conda create -y -n [ENV_NAME] python=[PYTHON_VERSION>=3]
conda activate [ENV_NAME]

# install requirements
bash install_for_aaa.sh
```

## Tracking

If you want to apply AAA to your own project,  
simply make the following python script:

```python
from algorithms.aaa import AAA

img_paths = []  # list of image file paths
initial_bbox = [x, y, w, h]  # left x, top y, width, height of the initial target bbox
n_experts = 6  # the number of experts you are using

# define AAA
theta = 0.69  # you can tune this hyperparameter by running run_tuning.sh
algorithm = AAA(n_experts, mode="LOG_DIR", feature_threshold=theta)

# initialize AAA
algorith.initialize(img_paths[0], initial_bbox)

# track the target
for img_path in img_paths[1:]:
    experts_result = np.zeros((n_experts, 4))  # the matrix of experts' estimation

    # state is the prediction of target bbox.
    # if the frame is not anchor frame, offline is None. else offline will be offline tracking results.
    # weight is the weight of the experts.
    state, offline, weight = self.track(img_path, experts_result)  
```

In addition, trackers that we have implemented can be easily executed the following python script.

```python
from select_options import select_expert

img_paths = []  # list of image file paths
initial_bbox = [x, y, w, h]  # left x, top y, width, height of the initial target bbox

# define Expert
tracker_name = "ATOM"
tracker = select_expert(tracker_name)

# initialize Expert
tracker.initialize(img_paths[0], initial_bbox)

# track the target
for img_path in img_paths[1:]:
    # state is the prediction of target bbox.
    state = self.track(img_path)  
```

## Requirements for experts

* PyTorch 1.6.0
* Tensorflow 1.14.0
* CUDA 10.1
* GCC 8

First, metafiles including pretrained weights are need to be donloaded.  
And, the path of metafiles in ``path_config.py`` and ``local.py`` file must be edited.

In order to run experts, you need to install additional libraries.  
We offer install script to make it easy to run experts:

```sh
# Only for Ubuntu
sudo apt install -y libopenmpi-dev libgl1-mesa-glx ninja-build

# activate anaconda environment
conda activate [ENV_NAME]

# install requirements
bash install_for_experts.sh
```

## Reproduce our results

We provide scripts to reproduce all results, figures, and tables in our paper.  
In addition, we provide the following files in case you don't have time to run all the scripts yourself.  
[Experts tracking results](##)  
[AAA tuning results](##)  
[AAA tracking results](##)  
[HDT tracking results](##)  
[MCCT tracking results](##)  
[Baselines tracking results](##)  

```sh
# Run experts
# If you've downloaded Experts tracking results, you can skip this command
bash run_experts.sh

# Tune the hyperparameter
# If you've downloaded AAA tuning results, you can skip this command
bash run_tuning.sh

# Run AAA
# If you've download AAA tracking results, you can skip this command
bash run_algorithm.sh

# Run HDT
# If you've download HDT tracking results, you can skip this command
bash run_hdt.sh

# Run MCCT
# If you've download MCCT tracking results, you can skip this command
bash run_mcct.sh

# Run Max and Random
# If you've download Baselines tracking results, you can skip this command
bash run_baselines.sh

# Visualize figures and tables in our paper
python visualize_figure.py
```

The code is supposed to run algorithms after running experts for test.  
However, it is easy to modify the code to do both simultaneously.

## Citation

If you find AAA useful in your work, please cite our paper:  

```none
@article{song2020aaa,
  title={AAA: Adaptive Aggregation of Arbitrary Online Trackers with Theoretical Performance Guarantee},
  author={Song, Heon and Suehiro, Daiki and Uchida, Seiichi},
  journal={arXiv preprint arXiv:2009.09237},
  year={2020}
}
```

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
