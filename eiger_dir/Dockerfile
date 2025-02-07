FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y curl vim
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash

RUN apt-get install -y libxcb-* libx11-xcb1 libxkbcommon-x11-0 x11-utils libdbus-1-dev libglib2.0-0 libgl1-mesa-glx
# RUN apt-get update && apt-get install -y \
    #    \
      
