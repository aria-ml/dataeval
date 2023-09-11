# CUDA+cuDNN docker from nVidia - https://hub.docker.com/r/nvidia/cuda/
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

USER root

# Install git, wget and dependencies for pyright (nodejs),
# add-apt-repository (software-properties-common), opencv (libgl1),
# sphinx (graphviz) and system level python packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gnupg2 \
        git \
        wget \
        nodejs \
        software-properties-common \
        libgl1 \
        graphviz \
        python3-pip \
        python3-venv \
        python-is-python3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set the local timezone for tzdata
RUN tz=`(wget -qO - http://geoip.ubuntu.com/lookup | sed -n -e 's/.*<TimeZone>\(.*\)<\/TimeZone>.*/\1/p')` \
 && ln -fs /usr/share/zoneinfo/$tz /etc/localtime

# Add the deadsnakes repository and install all flavors of python and supporting libraries
RUN add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3.8-venv \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        python3.9-venv \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.10-venv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create and switch to daml user
ARG HOMEDIR=/home/daml
RUN useradd -rm -d ${HOMEDIR} -s /bin/bash -G sudo -u 1000 daml
USER daml
WORKDIR ${HOMEDIR}

# Install tox, pyright and nvidia python index to base image
ENV PATH ${HOMEDIR}/.local/bin:$PATH
RUN pip install tox nvidia-pyindex

ARG PIPDOCS="${HOMEDIR}/.venv-docs/bin/python -I -m pip" \
    PIP38="${HOMEDIR}/.venv-py38/bin/python -I -m pip" \
    PIP39="${HOMEDIR}/.venv-py39/bin/python -I -m pip" \
    PIP310="${HOMEDIR}/.venv-py310/bin/python -I -m pip"

# Create virtual environments
RUN (/usr/bin/python3.8 -m venv ${HOMEDIR}/.venv-docs \
   & /usr/bin/python3.8 -m venv ${HOMEDIR}/.venv-py38 \
   & /usr/bin/python3.9 -m venv ${HOMEDIR}/.venv-py39 \
   & /usr/bin/python3.10 -m venv ${HOMEDIR}/.venv-py310 \
   & wait)

# Upgrade pip
RUN (${PIPDOCS} install --upgrade pip setuptools wheel \
   & ${PIP38} install --upgrade pip setuptools wheel \
   & ${PIP39} install --upgrade pip setuptools wheel \
   & ${PIP310} install --upgrade pip setuptools wheel \
   & wait)

ARG DOCS_DEPS="pydantic==1.10.12 sphinx==7.1.2 sphinx-immaterial==0.11.4"
ARG TEST_DEPS="pytest pyright coverage flake8 flake8-pyproject black[jupyter] isort"
ARG CUDA_DEPS="opencv-contrib-python-headless==4.8.0.76 nvidia-cudnn-cu11==8.6.0.163 tensorrt==8.6.1.post1"

# Set system locale for docs
ENV LC_ALL=C.UTF-8

# Install dependencies
RUN (${PIPDOCS} install ${DOCS_DEPS} \
   & ${PIP38} install ${TEST_DEPS} ${CUDA_DEPS} \
   & ${PIP39} install ${TEST_DEPS} ${CUDA_DEPS} \
   & ${PIP310} install ${TEST_DEPS} ${CUDA_DEPS} \
   & wait)

# Enable GPG signing
ENV GPG_TTY=$(tty)
