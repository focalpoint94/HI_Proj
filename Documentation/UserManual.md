# Local Deployment

1. Have python (programming language) installed.
2. Execute '`pip install -r requirements.txt`' inside the main HI_PROJ directory to install necessary dependencies.
3. Start the web application by executing '`python server.py`' in the main HI_PROJ directory. 
4. Alternatively, one may build a docker image and then deploy a container locally in order to start the web application.
    a. Execute `docker build -t hi_proj:latest .` in the main directory.
    b. Execute `docker run -p 5000:5000 hi_proj:latest` in the main directory.
    c. Go to `localhost:5000` in your web browser to examine the web application locally.
5. In order to use the website, the user must have a chest x-ray image that can be uploaded. The upload button is at the near bottom of the website, while before talks about our project introduction, implementation, evaluation metrics, and example analysis. Download the [image1]() and [image2]() for a sample chest x-ray image that you can upload to the website to.

# Production on Heroku

The web application is deployed on https://hiflaskr.herokuapp.com. It is the same as the local version.
