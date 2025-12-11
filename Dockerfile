FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements/env files
COPY environment.yml .

# Install Conda (Miniconda)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create environment
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "dbme", "/bin/bash", "-c"]

# Copy source code
COPY . .

# Default command
CMD ["conda", "run", "--no-capture-output", "-n", "dbme", "python", "-m", "src.main"]
