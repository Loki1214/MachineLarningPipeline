#!/bin/sh

timestump=$(date "+%Y%m%d")
docker build . -t localhost:5050/mywebapp:${timestump}
docker push localhost:5050/mywebapp:${timestump}
echo "docker image build with tag registry:5000/mywebapp:${timestump}"
docker images registry:5000/mywebapp | grep ${timestump}