FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 


RUN conda --version

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

RUN ls

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "facecup", "/bin/bash", "-c"]

# Make sure the environment is activated:
RUN echo "Make sure pymongo is installed:"
RUN python -c "import pymongo"

# The code to run when container is started:
COPY . .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "facecup", "python", "run.py"]
