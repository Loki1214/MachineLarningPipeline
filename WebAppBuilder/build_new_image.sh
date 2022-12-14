#!/bin/bash

LOG_OUT=log/stdout.log
LOG_ERR=log/stderr.log
exec 1> >(
	while read -r l; do echo "[$(date +"%Y-%m-%d %H:%M:%S")] $l"; done \
		| tee -a $LOG_OUT
	)
exec 2> >(
	while read -r l; do echo "[$(date +"%Y-%m-%d %H:%M:%S")] $l"; done \
	| tee -a $LOG_ERR 1>&2
	)

baseDir=`dirname $0`
modelDir=$1
context=${baseDir}/webAppFiles

if [ ! -e ${modelDir}/model_weights.pth  ] \
	|| [ ! -e ${modelDir}/model_definition.py  ] \
	|| [ ! -e ${modelDir}/model_definition.py  ]; then
	"Files required to build webapp image do not exist in ${modelDir}."
	exit 404
fi
cp --force ${modelDir}/* ${context}/app

Registry=localhost:5050
timestump=$(date "+%Y%m%d")
docker build --no-cache ${context} -t ${Registry}/mywebapp:${timestump} -t ${Registry}/mywebapp:latest
docker push -a ${Registry}/mywebapp
build_result=$?

echo "docker image built with tag ${Registry}/mywebapp:${timestump}"
docker image rm -f $(docker images ${Registry}/mywebapp | grep none | awk '{print $3}') 2>/dev/null
docker images ${Registry}/mywebapp

exit $build_result
