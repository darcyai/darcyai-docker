FROM debian:buster

RUN apt-get update
RUN apt-get install curl gnupg ca-certificates zlib1g-dev libjpeg-dev -y
RUN apt-get install python3 python3-pip -y
RUN apt-get install -y apt-utils libssl-dev
RUN apt-get install -y cmake

RUN pip3 install imutils==0.5.4 pillow==8.3.2 opencv-python==4.5.3.56
