#!/bin/sh

python3 work/train_NeuralNetwork.py
cp work/model_definition.py work/model_weights.pth work/imageClassifier.py trainedDNN/
