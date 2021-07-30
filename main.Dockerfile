FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN export TMPDIR='/var/tmp'
# RUN apt-get update && apt-get install -y python3.7
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt