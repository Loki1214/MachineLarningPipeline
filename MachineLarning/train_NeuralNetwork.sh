#!/bin/sh

Nunused=$(mysql -uroot --port=3306 -p${MYSQL_ROOT_PASSWORD} --database=DigitImages \
	-B -N -e "SELECT COUNT(is_used=false or NULL) FROM uploaded" 2>/dev/null)

if [ $Nunused -gt 1000 ]; then
	python3 train_NeuralNetwork.py
	cp model_definition.py model_weights.pth imageClassifier.py trainedDNN/
fi