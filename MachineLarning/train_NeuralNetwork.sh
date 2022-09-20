#!/bin/bash

LOG_OUT=log/stdout.log
LOG_ERR=log/stderr.log
# exec 1> >(
# 	while read -r l; do echo "[$(date +"%Y-%m-%d %H:%M:%S")] $l"; done \
# 		| tee -a $LOG_OUT
# 	)
# exec 2> >(
# 	while read -r l; do echo "[$(date +"%Y-%m-%d %H:%M:%S")] $l"; done \
# 	| tee -a $LOG_ERR 1>&2
# 	)

NewDataNum=2;
IntervalSec=10;


LOCK_FILE=/tmp/$(basename $0 .sh).lock
trap "rm -f ${LOCK_FILE}" SIGHUP SIGINT SIGQUIT SIGTERM

while ! ln -s $$ $LOCK_FILE; do
	lockDate=`stat -c %Y $LOCK_FILE`
	now=$(date +%s)
	ls -l $LOCK_FILE
	echo $now, $lockDate, $((now - lockDate))
	if [ $((now - lockDate)) -gt 3600 ]; then
		rm -f ${LOCK_FILE}
		continue;
	fi

    echo "The script \"${0}\" is already running.";
    exit 129;
done

function train_NeuralNetwork() {
	echo "Running command: python3 train_NeuralNetwork.py";
	python3 train_NeuralNetwork.py;
	cp model_definition.py model_weights.pth imageClassifier.py trainedDNN/;
	echo "Send request for building a new image.";
	curl -X POST -F  imageClassifier.py=@trainedDNN/imageClassifier.py  \
				 -F model_definition.py=@trainedDNN/model_definition.py \
				 -F   model_weights.pth=@trainedDNN/model_weights.pth   \
			http://builder:5000
}

while true; do
	echo "Running script ${0}"
	MNIST=$(mysql --host=${DATABASE} --port=3306 -u${MYSQL_USER} -p${MYSQL_PASSWORD} --database=${MYSQL_DATABASE} \
		-B -N -e "SELECT COUNT(*) FROM MNIST" 2>/dev/null)
	echo "MNIST = ${MNIST}"
	if [ ${MNIST:-0} -lt 70000 ]; then
		echo "Running command: python3 registerMNIST.py";
		python3 registerMNIST.py;
		train_NeuralNetwork;
	fi

	Nunused=$(mysql --host=${DATABASE} --port=3306 -u${MYSQL_USER} -p${MYSQL_PASSWORD} --database=${MYSQL_DATABASE} \
		-B -N -e "SELECT COUNT(is_used=false or NULL) FROM uploaded" 2>/dev/null)
	echo "Nunused = ${Nunused}"
	if [ ${Nunused:-0} -gt $((NewDataNum-1)) ]; then
		train_NeuralNetwork;
	fi

	echo 'END OF THE SCRIPT'
	echo >> $LOG_OUT
	echo >> $LOG_ERR
	sleep $IntervalSec;
done
rm -f $LOCK_FILE