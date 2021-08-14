FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-opencv && apt-get install -y time
RUN apt install -y python3-pip
RUN TMPDIR='/var/tmp/' pip3 install -r requirements.txt