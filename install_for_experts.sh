# install libraries
conda install -y tensorflow-gpu==1.14
pip install matplotlib pandas tqdm visdom scikit-image tikzplotlib pycocotools lvis jpeg4py pyyaml yacs colorama tb-nightly future optuna shapely scipy easydict tensorboardX mpi4py==2.0.0 gaft hyperopt ray==0.6.3 requests pillow msgpack msgpack_numpy tabulate xmltodict zmq annoy wget protobuf cupy-cuda101 mxnet-cu101 h5py pyzmq numba ipdb loguru scikit-learn got10k

# make directory for external libraries
cd external

# clone experts
# ATOM
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension
# DaSiamRPN
git clone https://github.com/songheony/DaSiamRPN
# DROL
git clone https://github.com/shallowtoil/DROL
# GradNet
git clone https://github.com/LPXTT/GradNet-Tensorflow
# MemDTC
git clone https://github.com/skyoung/MemDTC
# MemTrack
git clone https://github.com/skyoung/MemTrack
# Ocean
git clone https://github.com/researchmm/TracKit
# ROAM
git clone https://github.com/skyoung/ROAM
# RPT
git clone https://github.com/songheony/RPT
# SiamBAN
git clone https://github.com/hqucv/siamban
# SiamCAR
git clone https://github.com/ohhhyeahhh/SiamCAR
# SiamDW
git clone https://github.com/researchmm/SiamDW
# SiamFC
git clone https://github.com/got-10k/siamfc
# SiamFC++
git clone https://github.com/MegviiDetection/video_analyst
# SiamMCF
git clone https://github.com/hmorimitsu/siam-mcf
# SiamR-CNN
git clone https://github.com/VisualComputingInstitute/SiamR-CNN
# SiamRPN
git clone https://github.com/huanglianghua/siamrpn-pytorch
# SiamRPN++
git clone https://github.com/STVIR/pysot
# SPM
git clone https://github.com/songheony/SPM-Tracker
# Staple
git clone https://github.com/wwdguu/pyCFTrackers
# THOR
git clone https://github.com/xl-sr/THOR

# edit network path of ATOM, DiMP, PrDiMP, KYS
cd pytracking
cp ../../local.py pytracking/evaluation/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
cd ../

# For ATOM
cd pytracking
git submodule update --init
cd ../
cd Pytorch-Correlation-extension
pip install -e .
cd ../

# For DROL
cd DROL
python setup.py build_ext --inplace
cd ../

# For RLS-RTMDNet
cd RLS-RTMDNet/modules/roi_align
python setup.py build_ext --inplace
cd ../../../

# For RPT
cd RPT
python setup.py build_ext --inplace
cd siamreppoints
python ./setup.py build_ext --inplace
cd ../../

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