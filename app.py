from flask import Flask
import topictagging
app = Flask(__name__)

@app.route("/")
def home():
    return "Topic Tagging!"

@app.route("/tag/<fileName>")
def nishant(fileName):
    outputTag = topictagging.main(fileName)
    return outputTag