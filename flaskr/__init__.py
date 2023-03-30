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

from flask import Flask


def create_app(test_config=None):
    """
    Create Flask App

    This is a Flask App-Factory function. App-factory allows for flexibility for
    the entire flask app.

    One example code to deploy a production server is as below.

    Example:

    ```
    from waitress import serve
    import os

    from flaskr.__init__ import create_app

    port = int(os.environ.get('PORT', 5000))
    serve(create_app(), host='0.0.0.0', port=port)
    ```
    """
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    # folders for images
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'flaskr', 'static', 'images')
    GRADCAM_FOLDER = os.path.join(os.getcwd(), 'flaskr', 'static', 'gradcam')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

    # registering blueprints
    from . import xray
    app.register_blueprint(xray.bp)
    app.add_url_rule('/', endpoint='index')

    return app

