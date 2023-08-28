FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel as build
WORKDIR /workspace

# change chell to bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
SHELL=/bin/bash

# install basic dependencies
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y curl sudo htop vim
RUN pip3 install -U git+https://github.com/huggingface/peft.git

# install axolotl and chaiverse
RUN git clone https://github.com/OpenAccess-AI-Collective/axolotl
RUN cd axolotl && \
    pip3 install -e .[flash-attn]
RUN git clone https://github.com/chai-research/chaiverse
RUN cd chaiverse && \
    pip3 install -e .

# install jupyter notebook
WORKDIR /workspace/chaiverse
RUN pip install jupyterlab
RUN pip install ipywidgets

# start script to allow for runpod connection
ADD start.sh /
RUN chmod +x /start.sh
CMD [ "/start.sh" ]
