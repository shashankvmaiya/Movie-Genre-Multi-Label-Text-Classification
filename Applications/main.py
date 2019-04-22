from flask import Flask, render_template, request, url_for, jsonify
from keras.applications.inception_v3 import preprocess_input
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import base64
import io
import os
import sys
from werkzeug.utils import secure_filename


# from keras.models import load_model
# import matplotlib.pyplot as plt

app = Flask(__name__)

print('About to load the model')
MODEL_PATH = 'models/keras_mnist.h5'
model = load_model(MODEL_PATH)
print('Model loaded')


def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    # basepath = os.path.dirname(__file__)
    # file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    file_path = 'models/'+secure_filename(f.filename)
    print('Attempting to save the file at path: ', file_path)
    f.save(file_path)
    print('Saved image at path: ', file_path)
    return file_path


@app.route("/")
def index():
    return render_template("index.html")


# @app.route('/predict', methods=['POST'])
# @app.route("/", methods=['POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print('----------Entered predict---------')
    # basepath = os.path.dirname(__file__)
    # model_path = os.path.join(basepath, 'keras_mnist.h5')
    # print('Model Path: ', model_path)
    # model = load_model('models/keras_mnist_2.h5')
    # print('Model loaded')
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        # data = request.get_json()['data']
        # data = base64.b64decode(data)
        # img_data = io.BytesIO(data)
        img = image.load_img(file_path, target_size=(28, 28), color_mode="grayscale")
        x = (255-image.img_to_array(img))/255.0
        x = np.expand_dims(x, axis=0)
        print('Image Post-processing complete')
        predictions = model.predict(x)
        predictions = predictions.reshape(10)
        print('Predictions completed, llrs = ', predictions)
        digit_prob_order = np.argsort(-predictions)
        thresh = 0.99
        if predictions[digit_prob_order[0]] > thresh:
            top_pred_str = ''
        elif predictions[digit_prob_order[0]] + predictions[digit_prob_order[1]] > thresh:
            top_pred_str = '  (Prob = {0:.2f}, Other Predictions: {1} with Prob = {2:.2f})'.format(predictions[digit_prob_order[0]], digit_prob_order[1], predictions[digit_prob_order[1]])
        else:
            top_pred_str = '  (Prob = {0:.2f}, Other Predictions: [{1}, {3}] with Prob = [{2:.2f}, {4:.2f}])'.format(predictions[digit_prob_order[0]], digit_prob_order[1], predictions[digit_prob_order[1]], digit_prob_order[2], predictions[digit_prob_order[2]])
        digit_str = str(np.argmax(predictions))
        output_str = digit_str+top_pred_str
        print('Predicted digit = ', output_str)
        return output_str
    # return render_template('results.html', prediction=digit)
    # return jsonify({"prediction": data})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
