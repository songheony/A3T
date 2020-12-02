# install libraries
conda install -y pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install python-igraph opencv-python opencv-contrib-python Cython

# make directory for external libraries
mkdir external
cd external

# clone frameworks
git clone https://github.com/songheony/pytracking.git
git clone https://github.com/StrangerZhang/pysot-toolkit

# install region
cd pysot-toolkit/pysot/utils/
python setup.py build_ext --inplace