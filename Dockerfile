FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
# RUN apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && apt-get install -y python3.7
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
RUN pip3 install pandas