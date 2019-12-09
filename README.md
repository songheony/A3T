# Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound

## Experts

* [ATOM](https://arxiv.org/abs/1811.07628)[<https://github.com/visionml/pytracking>]
* [DaSiamRPN](https://arxiv.org/abs/1808.06048)[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>]<sup>[1]</sup>
* [DiMP](https://arxiv.org/abs/1904.07220)[<https://github.com/visionml/pytracking>]
* [GradNet](https://arxiv.org/abs/1909.06800)[<https://github.com/LPXTT/GradNet-Tensorflow>]
* [MemTrack](https://arxiv.org/abs/1803.07268)[<https://github.com/skyoung/MemTrack>]
* [SiamDW](https://arxiv.org/abs/1901.01660)[<https://github.com/researchmm/SiamDW>]
* [SiamFC](https://arxiv.org/abs/1606.09549)[<https://github.com/huanglianghua/siamfc-pytorch>]
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
bash run.sh
python run_algorithm.py
```

1. Clone this repository and make external directory.

2. Clone experts who you want to hire.<sup>[3]</sup>

3. Run the experts.

4. Run algorithms what you want.<sup>[4]</sup>

5. Evaluate the trackers and the baselines.

[3] Depending on the expert, you may need to install additional subparty libraries such as tensorflow.
[4] The code is supposed to run algorithms after running experts for test. However, it is easy to modify the code to do both simultaneously.

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)
