# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/pytorch:24.06-py3

ENV DEBIAN_FRONTEND=noninteractive

# metainformation
LABEL org.opencontainers.image.version = "2.0"
LABEL org.opencontainers.image.authors = "contact@eleuther.ai"
LABEL org.opencontainers.image.source = "https://www.github.com/eleutherai/gpt-neox"
LABEL org.opencontainers.image.licenses = " Apache-2.0"
LABEL org.opencontainers.image.base.name="nvcr.io/nvidia/pytorch:24.06-py3"

#### System package (uses default Python 3 version in Ubuntu 20.04)
RUN apt-get update -y && \
    apt-get install -y \
    python3-pip sudo pdsh \
    htop tmux zstd software-properties-common \
    nfs-common pdsh cmake htop iftop iotop ssh \
    iputils-ping net-tools libcupti-dev libmlx4-1 infiniband-diags ibutils \
    rdmacm-utils perftest rdma-core && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    python -m pip install --upgrade pip && \
    python -m pip install gpustat

### SSH
RUN mkdir /var/run/sshd && \
    # Prevent user being kicked off after login
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    # FIX SUDO BUG: https://github.com/sudo-project/sudo/issues/42
    echo "Set disable_coredump false" >> /etc/sudo.conf

# Expose SSH port
EXPOSE 22

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

#### User account
RUN useradd --create-home --uid 1000 --shell /bin/bash mchorse && \
    usermod -aG sudo mchorse && \
    echo "mchorse ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## SSH config and bashrc
RUN mkdir -p /home/mchorse/.ssh /job && \
    echo 'Host *' > /home/mchorse/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/mchorse/.ssh/config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/home/mchorse/.local/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /home/mchorse/.bashrc

#### Python packages
COPY requirements/* ./
RUN python -m pip install --no-cache-dir -r requirements.txt && pip install -r requirements-onebitadam.txt
RUN python -m pip install -r requirements-wandb.txt
RUN python -m pip install protobuf==3.20.*

COPY megatron/fused_kernels/ /megatron/fused_kernels
WORKDIR /megatron/fused_kernels
RUN python setup.py install

SHELL ["/bin/bash", "-c"]

RUN DS_BUILD_FUSED_LAMB=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_STOCHASTIC_TRANSFORMER=1  DS_BUILD_UTILS=1 \
    TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX" \
    python -m pip install git+https://github.com/microsoft/DeepSpeed.git@v0.14.4

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

#### SWITCH TO mchorse USER
USER mchorse
WORKDIR /home/mchorse
