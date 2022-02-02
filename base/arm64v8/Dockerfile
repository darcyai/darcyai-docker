FROM python:3.7-slim-buster

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils libssl-dev \
    && apt-get install -y --no-install-recommends cmake \
    && apt-get install -y --no-install-recommends curl gnupg ca-certificates zlib1g-dev libjpeg-dev \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 libxrender1 libglib2.0-bin \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install imutils==0.5.4 pillow==8.3.2 opencv-python==4.5.4.60
