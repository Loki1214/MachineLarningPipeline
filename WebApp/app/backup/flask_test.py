from flask import Flask, jsonify
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['Get'])
def predict():
    return 'URL preserved for the prediction.'

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})

# >>> Prepare input images
