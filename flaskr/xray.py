# Copyright [2023] [Hoesu Chun]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os


from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, current_app
)
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename


import matplotlib.cm as cm
import tensorflow as tf
import numpy as np


from . import classifier


bp = Blueprint('xray', __name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    """
    Checks if the image file has a valid extension
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/', methods=['GET', 'POST'])
def index():
    """
    View Function for the main page

    File's validity is verified first. If the uploaded file is verified,
    the file is saved in the server and the model makes a prediction. Grad-CAM
    image is generated and saved in the server as well.
    """
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file error')
            return redirect(url_for("xray.redirect_to_diagnosis", _anchor="diagnosis"))
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for("xray.redirect_to_diagnosis", _anchor="diagnosis"))
        # If the file is valid
        if file and allowed_file(file.filename):
            # Check the name's validity
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            # Save the file
            file.save(file_path)
            # Make Prediction
            pred, prob = predict_image(file_path)
            # Save the grad-cam image
            save_gradcam(file_path, model)
            # Render results
            return render_template('result.html', anchor="diagnosis",
                                   image1=filePathHtml('images', filename),
                                   pred=pred, prob=prob,
                                   image2=filePathHtml('gradcam', filename))
    return render_template('index.html')


@bp.route('/')
def redirect_to_diagnosis():
    return render_template('index.html')


def filePathHtml(folder, filename):
    filepath = os.path.join('static', folder, filename)
    return filepath


# Defined Parameters for the models
IMAGE_SIZE = 224
LABELS = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
LAST_LAYER = "top_activation"
# Load model
model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'flaskr', 'static', 'EfficientNet.h5'))
def predict_image(image_path):
    """
    Makes a prediction of the image file

    Args:
        image_path: string path of the image file

    Returns:
        Prediction: String name of the class
        Probability: Formatted string of the probability
    """
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)
    # verify if the image is X-ray
    if classifier.isXray(image) is False:
        flash("The uploaded image does not seem to be X-ray.")
    prediction = model.predict(image)
    prediction = [p for p in prediction]
    return LABELS[np.argmax(prediction[0])], "{:.2f}%".format(np.max(prediction[0])*100)


def make_gradcam_heatmap(image_path, model, pred_index=None):
    """
    Makes a gradcam heatmap

    Args:
        image_path: string path of the image file
        model: tensorflow model object
        pred_index: (optional) Index of the target class. Defaults to None.
        If none, heatmap is generated for the class with the largest probability.

    Returns:
        The created heatmap
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(LAST_LAYER).output, model.output]
    )

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_gradcam(image_path, model, pred_index=None, alpha=1.):
    """
    Makes and saves the gradcam heatmap

    Args:
        image_path: string path of the image file
        model: tensorflow model object
        pred_index: (optional) Index of the target class. Defaults to None.
        If none, heatmap is generated for the class with the largest probability.
        alpha: (optional) Visibility of the heatmap. Defaults to 1.
    """
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)

    heatmap = make_gradcam_heatmap(image_path, model, pred_index)
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    filename = image_path.split(os.path.sep)[-1]

    save_path = os.path.join(current_app.config['GRADCAM_FOLDER'], filename)
    superimposed_img.save(save_path)

