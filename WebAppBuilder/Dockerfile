FROM docker

RUN apk --update-cache add python3 py3-pip

RUN pip3 install --upgrade pip
RUN pip3 install flask
RUN pip3 install setproctitle

WORKDIR /home
COPY webAppFiles        /home/webAppFiles
COPY webApp_builder.py  /home
COPY build_new_image.sh /home
RUN mkdir -p /home/trainedDNN

CMD ["python3", "webApp_builder.py"]