ARG BASE_IMAGE_TAG=dev
ARG BASE_IMAGE_SUFFIX=x86
FROM edgeworx/darcy-ai-base-$BASE_IMAGE_SUFFIX:$BASE_IMAGE_TAG

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update

RUN apt-get install libedgetpu1-std python3-pycoral -y

RUN pip3 install numpy==1.21.0
