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

import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'flaskr', 'static', 'classifier.h5'))
def isXray(image):
    """
    Verifies if the given image is of X-ray

    Args:
        image: image file

    Returns:
        Boolean if the image is of X-ray
    """
    prediction = model.predict(image)
    prediction = [p for p in prediction]
    prediction = np.argmax(prediction[0])
    if prediction == 0:
        return True
    else:
        return False

