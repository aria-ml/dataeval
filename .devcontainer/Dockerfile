# CUDA+cuDNN docker from nVidia - https://hub.docker.com/r/nvidia/cuda/
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

USER root

# Install git, wget and dependencies for pyright (nodejs),
# add-apt-repository (software-properties-common), opencv (libgl1),
# sphinx (graphviz, pandoc) and system level python packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gnupg2 \
        git \
        wget \
        nodejs \
        software-properties-common \
        libgl1 \
        graphviz \
        pandoc \
        python3-pip \
        python3-venv \
        python-is-python3 \
        parallel \
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
RUN addgroup --gid 1000 daml
RUN addgroup --gid 1001 docker
RUN adduser --ingroup daml --ingroup docker --ingroup sudo --uid 1000 --disabled-password daml
USER daml

WORKDIR ${HOMEDIR}

# Enable system python packages to be found
ENV PATH ${HOMEDIR}/.local/bin:$PATH

# Set locale settings
ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8

# Enable GPG signing
ENV GPG_TTY=$(tty)

RUN pip install poetry tox pyright nvidia-pyindex
RUN pyright

ARG PIP38="${HOMEDIR}/.venv-py38/bin/python -I -m pip" \
    PIP39="${HOMEDIR}/.venv-py39/bin/python -I -m pip" \
    PIP310="${HOMEDIR}/.venv-py310/bin/python -I -m pip"

# Create virtual environments
RUN (/usr/bin/python3.8 -m venv ${HOMEDIR}/.venv-py38 \
   & /usr/bin/python3.9 -m venv ${HOMEDIR}/.venv-py39 \
   & /usr/bin/python3.10 -m venv ${HOMEDIR}/.venv-py310 \
   & wait)

# Upgrade pip
RUN (${PIP38} install --upgrade pip setuptools wheel \
   & ${PIP39} install --upgrade pip setuptools wheel \
   & ${PIP310} install --upgrade pip setuptools wheel \
   & wait)

# Prepare the dev environment
COPY --chown=daml:daml requirements.txt ./
RUN (${PIP38} install -r requirements.txt \
   & ${PIP39} install -r requirements.txt \
   & ${PIP310} install -r requirements.txt \
   & wait)

# cuda deps in requirements.txt
# opencv-contrib-python-headless==4.8.0.76
# nvidia-cudnn-cu11==8.6.0.163
