#docker build -t clion/remote-cuda-env:1.0 -f Dockerfile .
#docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 --name clion_remote_env clion/remote-cuda-env:1.0

FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update \
  && apt-get install -y ssh \
    build-essential \
    gcc \
    g++ \
    gdb \
    clang \
    cmake \
    rsync \
    tar \
    python \
    openssh-server \
    sudo \
    && apt-get clean

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test

RUN usermod -aG sudo test

RUN service ssh start

RUN  echo 'test:test' | chpasswd

EXPOSE 22

CMD ["/usr/sbin/sshd","-D"]



