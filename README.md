# Welcome to Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound 👋

## Experts

* ATOM[<https://github.com/visionml/pytracking>]
* BACF[<https://github.com/wwdguu/pyCFTrackers>]
* CSRDCF[<https://github.com/wwdguu/pyCFTrackers>]
* DaSiamRPN[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>][*]
* ECO[<https://github.com/StrangerZhang/pyECO>]
* ECO_new[<https://github.com/visionml/pytracking>]
* MDNet[<https://github.com/hyeonseobnam/py-MDNet.git>][**]
* SAMF[<https://github.com/wwdguu/pyCFTrackers>]
* SiamDW[<https://github.com/researchmm/SiamDW>]
* SiamFC[<https://github.com/huanglianghua/siamfc-pytorch>]
* SiamRPN[<https://github.com/huanglianghua/siamrpn-pytorch>]
* Staple[<https://github.com/wwdguu/pyCFTrackers>]
* STRCF[<https://github.com/wwdguu/pyCFTrackers>]
* TADT[<https://github.com/ZikunZhou/TADT-python>]
* Vital[<https://github.com/abnerwang/py-Vital.git>]

[*] Since original code of DaSiamRPN is for Python2, I had to modify the code a little bit to be compatible with Python3.
[**] The version of scipy should be under 1.1.0 because MDNet need to use imresize function.

## Datasets

* OTB2015[<http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html>]
* NFS[<http://ci2cv.net/nfs/index.html>]
* UAV123[<https://uav123.org/>]
* TColor128[<http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html>]
* VOT2018[<http://www.votchallenge.net/>][***]
* LaSOT[<https://cis.temple.edu/lasot/download.html>]

[***] VOT2018 is evaluated in unsupervised experiment as same as other datasets.

## Frameworks

* pytracking[<https://github.com/visionml/pytracking>] for tracking datasets.
* pysot-toolkit[<https://github.com/StrangerZhang/pysot-toolkit>] for evaluating trackers.

## Requirements

```sh
conda create -n [ENV_NAME] python=[PYTHON_VERSION>=3.6]
conda install pytorch torchvision cudatoolkit=[CUDA_VERSION] -c pytorch
pip install ortools opencv-python opencv-contrib-python
```

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)
