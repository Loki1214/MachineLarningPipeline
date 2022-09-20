FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

USER root

RUN apt update
RUN apt install -y python3 python3-pip
RUN apt install -y curl
RUN apt install -y mysql-client libmysqlclient-dev
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision
RUN pip3 install pandas boto3 mysqlclient
RUN pip3 install setproctitle

WORKDIR /home
COPY data/torch                  /home/data/torch
COPY work/registerMNIST.py       /home
COPY work/model_definition.py    /home
COPY work/custom_dataset.py      /home
COPY work/train_NeuralNetwork.py /home
COPY work/imageClassifier.py     /home
COPY train_NeuralNetwork.sh      /home
RUN mkdir -p /home/data
RUN mkdir -p /home/trainedDNN

COPY entry-point.sh /
ENTRYPOINT ["/entry-point.sh"]
CMD ["/home/train_NeuralNetwork.sh"]