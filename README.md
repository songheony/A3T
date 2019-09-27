# Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound

## Experts

* [ATOM](https://arxiv.org/abs/1811.07628)[<https://github.com/visionml/pytracking>]
* [DaSiamRPN](https://arxiv.org/abs/1808.06048)[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>]<sup>[1]</sup>
* [DiMP](https://arxiv.org/abs/1904.07220)[<https://github.com/visionml/pytracking>]
* [ECO](https://arxiv.org/abs/1611.09224)[<https://github.com/visionml/pytracking>]<sup>[2]</sup>
* [SiamDW](https://arxiv.org/abs/1901.01660)[<https://github.com/researchmm/SiamDW>]
* [SiamFC](https://arxiv.org/abs/1606.09549)[<https://github.com/huanglianghua/siamfc-pytorch>]
* [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)[<https://github.com/huanglianghua/siamrpn-pytorch>]
* [SiamRPN++](https://arxiv.org/abs/1812.11703)[<https://github.com/STVIR/pysot>]
* [Staple](https://arxiv.org/abs/1512.01355)[<https://github.com/wwdguu/pyCFTrackers>]
* [STRCF](https://arxiv.org/abs/1803.08679)[<https://github.com/wwdguu/pyCFTrackers>]<sup>[3]</sup>
* [TADT](https://arxiv.org/abs/1904.01772)[<https://github.com/ZikunZhou/TADT-python>]<sup>[3]</sup>

[1] Since the original code of DaSiamRPN is for Python2, We've had to modify the code a little bit to be compatible with Python3.  
[2] The author's new ECO is used.  
[3] We've found that there are errors on some dataset with SAMF, STRCF, and TADT.

## Datasets

* [OTB2015](https://ieeexplore.ieee.org/document/7001050)[<http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html>]
* [NFS](https://arxiv.org/abs/1703.05884)[<http://ci2cv.net/nfs/index.html>]
* [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)[<https://uav123.org/>]
* [TColor128](https://ieeexplore.ieee.org/document/7277070)[<http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html>]
* [VOT2018](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_1)[<http://www.votchallenge.net/>]<sup>[5]</sup>
* [LaSOT](https://arxiv.org/abs/1809.07845)[<https://cis.temple.edu/lasot/download.html>]

[5] VOT2018 is evaluated in unsupervised experiment as same as other datasets.

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

2. Clone experts who you want to hire.

3. Run the experts.

4. Run algorithms what you want.<sup>[6]</sup>

5. Evaluate the trackers and the baselines.

[6] The code is supposed to run algorithms after running experts for test. However, it is easy to modify the code to do both simultaneously.

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)
* Contact: songheony@gmail.com

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)
