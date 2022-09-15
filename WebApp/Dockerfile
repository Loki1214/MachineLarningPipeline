FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt update
RUN apt install -y python3 python3-pip
RUN apt install -y libmysqlclient-dev
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision
RUN pip3 install flask boto3 mysqlclient

WORKDIR /home
COPY app/templates/          /home/templates
COPY app/main.py             /home
COPY app/model_definition.py /home
COPY app/model_weights.pth   /home
COPY app/imageClassifier.py  /home
RUN mkdir -p /home/static
CMD ["python3", "main.py"]