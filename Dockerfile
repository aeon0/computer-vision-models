FROM tensorflow/tensorflow:2.4.2-gpu

# Install some usefull stuff
RUN apt-get update --fix-missing
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y git nano pkg-config wget usbutils
RUN add-apt-repository universe
RUN apt update
RUN apt install -y graphviz

# Install Edge-TPU support
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y libedgetpu1-std edgetpu-compiler python3-pycoral

# install tinker, provide some default values for promt
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN apt-get install -y python3-tk

# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version

# Add current project, Note: When changing the APP_PATH also adapt in .vscode/tasks.json
ENV APP_PATH /home/computer-vision-models
ENV PYTHONPATH ${APP_PATH}:${PYTHONPATH}
RUN mkdir -p ${APP_PATH}
COPY . ${APP_PATH}
WORKDIR ${APP_PATH}

# Update conda environment and activate conda env
RUN conda env update environment.yml
RUN echo "source activate cvms" > ~/.bashrc
ENV PATH /root/miniconda3/envs/cvms/bin:$PATH
