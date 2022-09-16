#!/bin/sh

timestump=$(date "+%Y%m%d")
docker build --no-cache . -t localhost:5050/mywebapp:${timestump} -t localhost:5050/mywebapp:latest
docker push -a localhost:5050/mywebapp
echo "docker image build with tag localhost:5050/mywebapp:${timestump}"
docker images localhost:5050/mywebapp
docker image rm -f $(docker images localhost:5050/mywebapp | grep none | awk '{print $3}')