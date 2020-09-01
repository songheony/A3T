# Adaptive Aggregation of Arbitrary Online Trackers <br/> with a Regret Bound

## Experts

* [ATOM](https://arxiv.org/abs/1811.07628)[<https://github.com/visionml/pytracking>]
* [DaSiamRPN](https://arxiv.org/abs/1808.06048)[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>]<sup>[1]</sup>
* [GradNet](https://arxiv.org/abs/1909.06800)[<https://github.com/LPXTT/GradNet-Tensorflow>]
* [MemTrack](https://arxiv.org/abs/1803.07268)[<https://github.com/skyoung/MemTrack>]
* [SiamDW](https://arxiv.org/abs/1901.01660)[<https://github.com/researchmm/SiamDW>]
* [SiamFC](https://arxiv.org/abs/1606.09549)[<https://github.com/got-10k/siamfc>]
* [SiamMCF](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_6)[<https://github.com/hmorimitsu/siam-mcf>]
* [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)[<https://github.com/huanglianghua/siamrpn-pytorch>]
* [SiamRPN++](https://arxiv.org/abs/1812.11703)[<https://github.com/STVIR/pysot>]
* [SPM](https://arxiv.org/abs/1904.04452)[<https://github.com/microsoft/SPM-Tracker>]
* [Staple](https://arxiv.org/abs/1512.01355)[<https://github.com/wwdguu/pyCFTrackers>]
* [THOR](https://arxiv.org/abs/1907.12920)[<https://github.com/xl-sr/THOR>]

[1] Since the original code of DaSiamRPN is for Python2, We've had to modify the code a little bit to be compatible with Python3.

## Datasets

* [OTB2015](https://ieeexplore.ieee.org/document/7001050)[<http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html>]
* [NFS](https://arxiv.org/abs/1703.05884)[<http://ci2cv.net/nfs/index.html>]
* [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)[<https://uav123.org/>]
* [TColor128](https://ieeexplore.ieee.org/document/7277070)[<http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html>]
* [VOT2018](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_1)[<http://www.votchallenge.net/>]<sup>[2]</sup>
* [LaSOT](https://arxiv.org/abs/1809.07845)[<https://cis.temple.edu/lasot/download.html>]
* [Got10K](https://arxiv.org/abs/1810.11981)[<http://got-10k.aitestunion.com/>]

[2] VOT2018 is evaluated in unsupervised experiment as same as other datasets.

## Frameworks

* pytracking[<https://github.com/visionml/pytracking>] for tracking datasets.
* pysot-toolkit[<https://github.com/StrangerZhang/pysot-toolkit>] for evaluating trackers.

## Requirements

```sh
conda create -n [ENV_NAME] python=[PYTHON_VERSION>=3.6]
conda install pytorch torchvision cudatoolkit=[CUDA_VERSION] -c pytorch
pip install python-igraph opencv-python opencv-contrib-python
```

## How to run

```sh
git clone https://github.com/songheony/AAA-journal
mkdir AAA-journal/external
cd AAA-journal/external
git clone [FRAMEWORK_GIT]
git clone [EXPERT_GIT]
conda activate [ENV_NAME]
bash run_experts.sh
bash run_tuning.sh
bash run_algorithm.sh
bash run_eval.sh
python visualize_figure.py
```

1. Clone this repository and make external directory.

2. Clone experts who you want to hire.<sup>[3]</sup>

3. Edit run_expert.sh file and run experts.

4. Edit run_tuning.sh file and tune hyperparamter theta.

5. Edit run_algorithm.sh file and run algorithm.<sup>[4]</sup>

6. Edit run_eval.sh and evaluate the trackers.

7. Edit visualize_figure.py and create figures used in our paper.

[3] Depending on the expert, you may need to install additional subparty libraries such as tensorflow.  
[4] The code is supposed to run algorithms after running experts for test. However, it is easy to modify the code to do both simultaneously.

## Reproduce our results

You can reproduce our results by using created environment and results.  
Download [AAA+Experts Tracking results](https://drive.google.com/file/d/1M4mk1zh4tp8vnCQ5gk4-pN-jvrkZSX1w/view?usp=sharing) and [Evaluation results](https://drive.google.com/file/d/1mNMy_4w7BchUF4skS9PK391pWA9KI2LC/view?usp=sharing).  
Then, run the followind commands.  

```sh
git clone https://github.com/songheony/AAA-journal
mkdir AAA-journal/external
cd AAA-journal/external

# clone frameworks
git clone https://github.com/visionml/pytracking
git clone https://github.com/StrangerZhang/pysot-toolkit

# clone experts
git clone https://github.com/songheony/DaSiamRPN
git clone https://github.com/LPXTT/GradNet-Tensorflow
git clone https://github.com/skyoung/MemTrack
git clone https://github.com/researchmm/SiamDW
git clone https://github.com/got-10k/siamfc
git clone https://github.com/hmorimitsu/siam-mcf
git clone https://github.com/huanglianghua/siamrpn-pytorch
git clone https://github.com/STVIR/pysot
git clone https://github.com/microsoft/SPM-Tracker
git clone https://github.com/wwdguu/pyCFTrackers
git clone https://github.com/xl-sr/THOR

# create anaconda environment
conda env create -f environment.yml

# run experts. if you download AAA+Experts Tracking results, you can skip this command
bash run_experts.sh

# tune the hyperparameter. if you download AAA+Experts Tracking results, you can skip this command
bash run_tuning.sh

# run AAA. if you download AAA+Experts Tracking results, you can skip this command
bash run_algorithm.sh

# run HDT. if you download AAA+Experts Tracking results, you can skip this command
bash run_hdt.sh

# run MCCT. if you download AAA+Experts Tracking results, you can skip this command
bash run_mcct.sh

# run Max and Random. if you download AAA+Experts Tracking results, you can skip this command
bash run_baselines.sh

# evaluate experts and AAA. if you download Evaluation results Tracking results, you can skip this command
bash run_eval.sh

# visualize results
python visualize_figure.py
```

## Simple using

If you want AAA to your own project, simply use the following code:

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

## Citation

If you find AAA useful in your work, please cite our paper:  

```none
@inproceedings{song2020adaptive,
title={Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound},
author={Song, Heon and Suehiro, Daiki and Uchida, Seiichi},
booktitle={The IEEE Winter Conference on Applications of Computer Vision},
pages={681--689},
year={2020}
}
```

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com
