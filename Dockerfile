FROM python:3.7
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt