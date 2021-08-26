FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
COPY packages.txt packages.txt
RUN apt-get update
RUN xargs -a packages.txt apt-get install -y
COPY requirements.txt requirements.txt
RUN TMPDIR='/var/tmp/' pip3 install -r requirements.txt --cache-dir='/var/tmp/' --build '/var/tmp/' 