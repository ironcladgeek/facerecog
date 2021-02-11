FROM nvidia/cuda:10.1-base-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install --no-install-recommends -y \
    gcc python3-dev libxml2-dev libxslt1-dev zlib1g-dev g++ wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "facecup", "/bin/bash", "-c"]

# The code to run when container is started:
COPY . .
RUN ls -a
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "facecup", "python", "run.py"]
