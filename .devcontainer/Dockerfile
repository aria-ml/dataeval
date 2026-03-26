# syntax=docker/dockerfile:1.4

ARG USER="dataeval"
ARG UID="1000"

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG UID
ARG USER

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        clang \
        curl \
        git \
        gnupg2 \
        graphviz \
        libgl1 \
        openssh-server \
        parallel \
        sudo

# Dev container tools expect non-root users to be able to sudo in a
# non-interactive context, so allow the user to use passwordless sudo.
RUN useradd -m -u ${UID} -s /bin/bash ${USER} -G sudo
RUN echo "${USER} ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/user-nopasswd

RUN addgroup --gid 1001 docker
RUN usermod -a -G docker ${USER}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

USER ${USER}
WORKDIR /${USER}

ENV LANGUAGE=en
ENV LC_ALL=C.UTF-8
ENV LANG=en_US.UTF-8
