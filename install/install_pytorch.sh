#!/bin/bash

#  Uninstall packages that may break rpc
if ! pip uninstall ipykernel jupyter jupyter-client; then
    echo "Failed to uninstall required packages using pip, they do not exist."
fi

# Install required libraries using pip
if ! pip install -r requirements.txt; then
    echo "Failed to install requirements using pip, cannot continuing without them."
    exit 1
fi

# Install PyTorch using conda
if ! conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia; then
    echo "Failed to install Tensorflow using conda, aborting."
    exit 1
fi

if ! pip install stable-baselines3[extra]
	echo "Failed to install stable baselines using pip, aborting."
	exit 1
fi

pip install git+https://github.com/DLR-RM/stable-baselines3@feat/gymnasium-support
pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@feat/gymnasium-support

echo "Installation complete."

