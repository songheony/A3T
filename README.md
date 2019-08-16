# Welcome to Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound 👋

## Framework

* pytracking[<https://github.com/visionml/pytracking>] for tracking datasets.
* pysot-toolkit[<https://github.com/StrangerZhang/pysot-toolkit>] for evaluating trackers.

## Experts

* ATOM[<https://github.com/visionml/pytracking>]
* BACF[<https://github.com/wwdguu/pyCFTrackers>]
* CSRDCF[<https://github.com/wwdguu/pyCFTrackers>]
* DaSiamRPN[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>][*]
* ECO[<https://github.com/StrangerZhang/pyECO>]
* ECO_new[<https://github.com/visionml/pytracking>]
* MDNet[<https://github.com/hyeonseobnam/py-MDNet.git>]
* SAMF[<https://github.com/wwdguu/pyCFTrackers>]
* SiamDW[<https://github.com/researchmm/SiamDW>]
* SiamFC[<https://github.com/huanglianghua/siamfc-pytorch>]
* SiamRPN[<https://github.com/huanglianghua/siamrpn-pytorch>]
* Staple[<https://github.com/wwdguu/pyCFTrackers>]
* STRCF[<https://github.com/wwdguu/pyCFTrackers>]
* TADT[<https://github.com/ZikunZhou/TADT-python>]
* Vital[<https://github.com/abnerwang/py-Vital.git>]

[*] Since original code of DaSiamRPN is for Python2, I had to modify the code a little bit to be compatible with Python3.

## Requirements

```sh
conda create -n [ENV_NAME] python=[PYTHON_VERSION>=3.7]
conda install pytorch torchvision cudatoolkit=[CUDA_VERSION>=1.0] -c pytorch
conda install matplotlib pywget shapely pandas
pip install ortools opencv-python opencv-contrib-python
git clone https://github.com/got-10k/toolkit.git external/toolkit
```

Optional (for experts) [**]:

```sh
conda install scipy cupy scikit-learn
pip install pyyaml easydict mxnet-cu100 yacs
sudo apt install ninja-build
```

[**] The version of scipy should be under 1.1.0 because MDNet need to use imresize function.

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)
