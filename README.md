# Welcome to Adaptive Aggregation of Arbitrary Online Trackers with a Regret Bound 👋

## Experts

* ATOM[<https://github.com/visionml/pytracking>]
* ECO[<https://github.com/StrangerZhang/pyECO>,<https://github.com/wwdguu/pyCFTrackers>]
* TADT[<https://github.com/ZikunZhou/TADT-python>]
* UDT[<https://github.com/594422814/UDT_pytorch>]
* Vital[<https://github.com/abnerwang/py-Vital.git>]
* MDNet[<https://github.com/hyeonseobnam/py-MDNet.git>]
* DaSiamRPN[<https://github.com/foolwood/DaSiamRPN>,<https://github.com/songheony/DaSiamRPN>][*]
* SiamDW[<https://github.com/researchmm/SiamDW>]
* SiamFC[<https://github.com/huanglianghua/siamfc-pytorch>]
* SiamRPN[<https://github.com/huanglianghua/siamrpn-pytorch>]

[*] Since original code of DaSiamRPN is for python 2.x, I should change the code a little bit for python 3.x

## Requirements

```sh
conda create -n [ENV_NAME] python=[PYTHON_VERSION>=3.7]
conda install pytorch torchvision cudatoolkit=[CUDA_VERSION>=1.0] -c pytorch
conda install matplotlib pywget shapely pandas
pip install ortools opencv-python opencv-contrib-python
```

Optional (for experts):
```sh
pip install yaml easydict
sudo apt install ninja-build
```

[*] Since the original code is for Python2, I've modified it very little to be compatible with Python3.

## Author

👤 **Heon Song**

* Github: [@songheony](https://github.com/songheony)

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)
