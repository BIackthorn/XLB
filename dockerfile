FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

#--- [BEGIN] SSHD

RUN apt-get update
RUN apt-get -y install openssh-server
RUN systemctl enable ssh
RUN mkdir /var/run/sshd
# Add host keys
RUN cd /etc/ssh/ && ssh-keygen -A -N ''

# Config SSH Daemon
#RUN  sed -i "s/#PasswordAuthentication.*/PasswordAuthentication no/g" /etc/ssh/sshd_config \
#  && sed -i "s/#PermitRootLogin.*/PermitRootLogin no/g" /etc/ssh/sshd_config \
#  && sed -i "s/#AuthorizedKeysFile/AuthorizedKeysFile/g" /etc/ssh/sshd_config

RUN  sed -i "s/#PermitRootLogin.*/PermitRootLogin no/g" /etc/ssh/sshd_config \
  && sed -i "s/#AuthorizedKeysFile/AuthorizedKeysFile/g" /etc/ssh/sshd_config

# Unlock non-password USER to enable SSH login
# RUN passwd -u ${USER}
# RUN usermod -p '*' ${USER}

# Set up user's public and private keys
# ENV SSHDIR ${USER_HOME}/.ssh
# RUN mkdir -p ${SSHDIR}

# Default ssh config file that skips (yes/no) question when first login to the host
RUN echo "StrictHostKeyChecking no" > ${SSHDIR}/config
# This file can be overwritten by the following  step if ssh/ directory has config file

RUN apt-get install iproute2 -y


RUN apt-get update && apt-get install -y openssh-server

EXPOSE 22





RUN apt-get update && \
    apt-get install -y gnupg2 build-essential apt-utils python3 git python3-pip libxrender1 gdb wget libssl-dev software-properties-common python3-dev vim libgl1


RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
CMD /bin/bash -c 'cat /message.txt;cp /etc/group2 /etc/group; cp /etc/sudoers2 /etc/sudoers; service ssh start ; echo "${TARGET_DOCKER_USER}	ALL=(ALL:ALL) ALL" >> /etc/sudoers; su - ${TARGET_DOCKER_USER}; /bin/bash'

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pyevtk numpy pillow scipy jupyter jupyter-server matplotlib pandas scikit-fmm
RUN python3 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3 -m pip install jmp warp-lang tqdm pyvista trimesh Rtree orbax-checkpoint termcolor
RUN python3 -m pip install git+https://github.com/loliverhennigh/PhantomGaze.git
ENV PYTHONPATH=.
ENV XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_enable_async_all_reduce=true"
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV NCCL_NVLS_ENABLE=0
ENV CUDA_MODULE_LOADING=EAGER
