FROM nvidia/cuda:10.1-base-ubuntu18.04


RUN apt-get update && apt-get -y install gcc python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++ libprotobuf-dev protobuf-compiler cmake 

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Create the environment:
COPY requirements.txt .
RUN python3 -m pip install scikit-build
RUN python3 -m pip install -r requirements.txt

# The code to run when container is started:
COPY . .
RUN ls -a
ENTRYPOINT ["python", "run.py"]
