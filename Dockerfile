FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN apt-get update && apt-get install --no-install-recommends -y \
    gcc python3-dev python3-pip python-pip libxml2-dev libxslt1-dev \
    zlib1g-dev g++ libprotobuf-dev protobuf-compiler cmake \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Installing dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir setuptools==51.1.1
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Code to run when container is started
COPY . .
RUN ls -a
CMD ["run.py"]
ENTRYPOINT ["python3"]
