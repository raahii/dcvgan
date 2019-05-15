FROM nvidia/cuda:9.0-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    cmake \
    libgtk2.0-dev \
    graphviz \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user
WORKDIR $HOME/dcvgan

# Install python with pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PYTHON_VERSION miniconda3-4.3.30
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

# Install dependencies
RUN conda install -y ffmpeg
ADD ./requirements.txt ./
RUN pip install -r ./requirements.txt
