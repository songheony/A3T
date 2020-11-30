FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8
RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update

# ==================================================================
# tools
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \
    git clone --depth 10 https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install

# ==================================================================
# python
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    python -m pip --no-cache-dir install --upgrade \
        setuptools \
        && \
    python -m pip --no-cache-dir install --upgrade \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm

# ==================================================================
# pytorch
# ------------------------------------------------------------------

RUN python -m pip --no-cache-dir install --upgrade \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    python -m pip --no-cache-dir install --upgrade \
        torch==1.7.0+cu101 \
        torchvision==0.8.1+cu101 -f \
        https://download.pytorch.org/whl/torch_stable.html

# ==================================================================
# tensorflow
# ------------------------------------------------------------------

RUN python -m pip --no-cache-dir install --upgrade \
        tensorflow-gpu==1.15

# ==================================================================
# AAA
# ------------------------------------------------------------------

RUN python -m pip --no-cache-dir install --upgrade \
        python-igraph \
        opencv-python \
        opencv-contrib-python

# ==================================================================
# experts
# ------------------------------------------------------------------

RUN pip uninstall -y enum34

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libopenmpi-dev

RUN python -m pip --no-cache-dir install --upgrade \
        matplotlib \
        pandas \
        tqdm \
        cython \
        visdom \
        scikit-image \
        tikzplotlib \
        pycocotools \
        lvis \
        jpeg4py \
        pyyaml \
        yacs \
        colorama \
        tensorboard \
        future \
        optuna \
        shapely \
        scipy \
        easydict \
        tensorboardX \
        mpi4py==2.0.0 \
        gaft \
        hyperopt \
        ray==0.6.3 \
        requests \
        pillow \
        msgpack \
        msgpack_numpy \
        tabulate \
        xmltodict \
        zmq \
        annoy \
        wget \
        protobuf \
        cupy-cuda101 \
        mxnet-cu101 \
        h5py \
        pyzmq \
        git+https://github.com/got-10k/toolkit.git@master \
        git+https://github.com/tensorpack/tensorpack.git

# ==================================================================
# fix error
# ------------------------------------------------------------------

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgl1-mesa-glx ninja-build

RUN python -m pip --no-cache-dir install --upgrade numba

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

EXPOSE 6006
