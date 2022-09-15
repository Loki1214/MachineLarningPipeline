#!/bin/sh

python3 train_NeuralNetwork.py
cp model_definition.py model_weights.pth imageClassifier.py trainedDNN/
