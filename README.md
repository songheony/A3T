# AAA: Adaptive Aggregation of Arbitrary Online Trackers <br/> with Theoretical Performance Guarantee

[AAA: Adaptive Aggregation of Arbitrary Online Trackers with Theoretical Performance Guarantee](https://arxiv.org/abs/2009.09237)

Heon Song, Daiki Suehiro, Seiichi Uchida

> For visual object tracking, it is difficult to realize an almighty online tracker due to the huge variations of target appearance depending on an image sequence. This paper proposes an online tracking method that adaptively aggregates arbitrary multiple online trackers. The performance of the proposed method is theoretically guaranteed to be comparable to that of the best tracker for any image sequence, although the best expert is unknown during tracking. The experimental study on the large variations of benchmark datasets and aggregated trackers demonstrates that the proposed method can achieve state-of-the-art performance.

![Alt text](assets/Table1.png?raw=true "Score")

## Experts

In this repository, we implemented or edited the following trackers to use as experts.  
**You can use the tracker with just a few lines of code.**

| Tracker   | Link                |
|-----------|---------------------|
| ATOM (CVPR 2019)      | [Paper](https://arxiv.org/abs/1811.07628) / [Original Repo](https://github.com/visionml/pytracking) |
| DaSiamRPN (ECCV 2018) | [Paper](https://arxiv.org/abs/1808.06048) / [Original Repo](https://github.com/foolwood/DaSiamRPN) |
| DiMP (ICCV 2019)      | [Paper](https://arxiv.org/abs/1904.07220) / [Original Repo](https://github.com/visionml/pytracking) |
| DROL (AAAI 2020)      | [Paper](https://arxiv.org/abs/1909.02959) / [Original Repo](https://github.com/shallowtoil/DROL) |
| GradNet (ICCV 2019)   | [Paper](https://arxiv.org/abs/1909.06800) / [Original Repo](https://github.com/LPXTT/GradNet-Tensorflow) |
| KYS (ECCV 2020)   | [Paper](https://arxiv.org/abs/2003.11014) / [Original Repo](https://github.com/visionml/pytracking) |
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
| SPM (CVPR 2019)       | [Paper](https://arxiv.org/abs/1904.04452) / [Original Repo](https://github.com/microsoft/SPM-Tracker) |
| Staple (CVPR 2016)   | [Paper](https://arxiv.org/abs/1512.01355) / [Original Repo](https://github.com/wwdguu/pyCFTrackers) |
| THOR (BMVC 2019)     | [Paper](https://arxiv.org/abs/1907.12920) / [Original Repo](https://github.com/xl-sr/THOR) |
| TRAS (ACCV 2020)     | [Paper](https://arxiv.org/abs/2007.04108) / [Original Repo](https://github.com/dontfollowmeimcrazy/vot-kd-rl) |

[1] Since the original code of DaSiamRPN is for Python2, We've had to modify the code a little bit to be compatible with Python3.

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

[2] VOT2018 is evaluated in unsupervised experiment as same as other datasets.

## Frameworks

The following frameworks were used to conveniently track videos and evaluate trackers.

* pytracking[<https://github.com/visionml/pytracking>] for tracking datasets.
* pysot-toolkit[<https://github.com/StrangerZhang/pysot-toolkit>] for evaluating trackers.

## Requirements

First, you need to download this repository and the frameworks.

```sh
# clone this repository
git clone https://github.com/songheony/AAA-journal

# make directory for external libraries
mkdir AAA-journal/external
cd AAA-journal/external

# clone frameworks
git clone https://github.com/visionml/pytracking
git clone https://github.com/StrangerZhang/pysot-toolkit

# install region
cd pysot-toolkit/pysot/utils/
python setup.py build_ext --inplace
```

After that, you need to install the following libraries.

* pytorch
* python-igraph
* opencv-python

We strongly recommend using a virtual environment like Anaconda or Docker.  
The following is how to build the virtual environment for AAA when using anaconda.

```sh
conda create -n [ENV_NAME] python=[PYTHON_VERSION>=3]
conda activate [ENV_NAME]
conda install pytorch torchvision cudatoolkit=[CUDA_VERSION] -c pytorch
pip install python-igraph opencv-python opencv-contrib-python
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

* PyTorch 1.4.0
* CUDA 10.0
* GCC 8

First, metafiles including pretrained weights are need to be donloaded.  
After modifing the path of metafiles in ``path_config.py`` and ``local.py`` file, run following commands to clone repositories.

```sh
cd external

# clone experts
git clone https://github.com/songheony/DaSiamRPN
git clone https://github.com/LPXTT/GradNet-Tensorflow
git clone https://github.com/skyoung/MemTrack
git clone https://github.com/researchmm/TracKit
git clone https://github.com/hqucv/siamban
git clone https://github.com/ohhhyeahhh/SiamCAR
git clone https://github.com/researchmm/SiamDW
git clone https://github.com/got-10k/siamfc
git clone https://github.com/MegviiDetection/video_analyst
git clone https://github.com/hmorimitsu/siam-mcf
git clone https://github.com/VisualComputingInstitute/SiamR-CNN
git clone https://github.com/huanglianghua/siamrpn-pytorch
git clone https://github.com/STVIR/pysot
git clone https://github.com/microsoft/SPM-Tracker
git clone https://github.com/wwdguu/pyCFTrackers
git clone https://github.com/xl-sr/THOR
git clone https://github.com/dontfollowmeimcrazy/vot-kd-rl

cd ../
```

In order to run experts, you need to install additional libraries.  
We offer three options to make it easy to run experts:

### Install pre-built Anaconda environment

```sh
conda create -f environment.yml
conda activate aaa
```

### Install pre-built Docker environment

```sh
docker pull songheony/aaa
docker run --gpus all -it -v "${PWD}:/workspace" --ipc=host songheony/aaa bash
```

### Install python libraries manually

```sh
# For mpi4py
sudo apt install libopenmpi-dev

# Change cupy-cuda100 and mxnet-cu100 to proper CUDA version.
pip install tensorflow-gpu==1.14 matplotlib pandas tqdm cython visdom scikit-image tikzplotlib pycocotools lvis jpeg4py pyyaml yacs colorama tensorboard future optuna shapely scipy easydict tensorboardX mpi4py==2.0.0 gaft hyperopt ray==0.6.3 requests pillow msgpack msgpack_numpy tabulate xmltodict zmq annoy wget protobuf cupy-cuda100 mxnet-cu100 h5py pyzmq numba ipdb loguru scikit-learn spatial-correlation-sampler

pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
```

After installing the libraries, some libraries need to be compiled manually.

```sh
cd external

# edit network path of ATOM, DiMP, PrDiMP, KYS
cd pytracking
cp ../../local.py pytracking/evaluation/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
cd ../

# For ATOM
cd pytracking
git submodule update --init  
apt-get install ninja-build
cd ../

# For DROL
cd DROL
python setup.py build_ext --inplace
cd ../

# For RLS-RTMDNet
cd RLS-RTMDNet/modules/roi_align
python setup.py build_ext --inplace
cd ../../../

# For SiamBAN
cd siamban
python setup.py build_ext --inplace
cd ../

# For SiamR-CNN
cd SiamR-CNN
git clone https://github.com/pvoigtlaender/got10k-toolkit.git
git clone https://github.com/tensorpack/tensorpack.git
cd tensorpack
git checkout d24a9230d50b1dea1712a4c2765a11876f1e193c
cd ../../

# For SiamRPN++
cd pysot
python setup.py build_ext --inplace
cd ../

# For SPM
cd SPM-Tracker
bash compile.sh
cd ../

# For Staple
cd pyCFTrackers/lib/pysot/utils
python setup.py build_ext --inplace
cd ../../../../
cd pyCFTrackers/lib/eco/features/
python setup.py build_ext --inplace
cd ../../../../

# For THOR
cd THOR
bash benchmark/make_toolkits.sh
cd ../

cd ../
```

## Reproduce our results

<!-- You can reproduce our results by using created environment and results.  
If you don't want to run experts or AAA, you can download [AAA+Experts Tracking results](https://drive.google.com/file/d/1Vw8KuF-4_1Dc7XHa6lAyHxjKve-UlE5B/view?usp=sharing) and [Evaluation results](https://drive.google.com/file/d/1nqQk8fZIef1hFIRM_RW425ti6stKwxJA/view?usp=sharing). Moreover, you can download figures in our paper from [here](https://drive.google.com/file/d/12O2saVFQD9e01GuTkKHWi79ohqjQR3eQ/view?usp=sharing).  
Or, if you want to reproduce our results by yourself, run the following commands.   -->

```sh
# run experts. if you've downloaded Experts Tracking results, you can skip this command
bash run_experts.sh

# tune the hyperparameter. if you've downloaded AAA Tuning results, you can skip this command
bash run_tuning.sh

# run AAA. if you've download AAA Tracking results, you can skip this command
bash run_algorithm.sh

# run HDT. if you've download HDT Tracking results, you can skip this command
bash run_hdt.sh

# run MCCT. if you've download MCCT Tracking results, you can skip this command
bash run_mcct.sh

# run Max and Random. if you've download Baselines Tracking results, you can skip this command
bash run_baselines.sh

# evaluate experts and AAA. if you've download Evaluation results, you can skip this command
bash run_eval.sh

# visualize results
python visualize_figure.py
```

The code is supposed to run algorithms after running experts for test. However, it is easy to modify the code to do both simultaneously.

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
