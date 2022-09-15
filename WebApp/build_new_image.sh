#!/bin/sh

timestump=$(date "+%Y%m%d")
docker build --no-cache . -t localhost:5050/mywebapp:${timestump} -t localhost:5050/mywebapp:latest
docker push -a localhost:5050/mywebapp
echo "docker image build with tag registry:5000/mywebapp:${timestump}"
docker images registry:5000/mywebapp