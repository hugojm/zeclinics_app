
from flask import Flask
app = Flask(__name__)

posts = []
@app.route("/")
def index():
    return "{} posts".format(len(posts))
