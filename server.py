import os


from waitress import serve


from flaskr.__init__ import create_app


port = int(os.environ.get('PORT', 5000))
serve(create_app(), host='0.0.0.0', port=port)

