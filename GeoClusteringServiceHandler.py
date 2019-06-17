from flask import Flask

app = Flask(__name__)

HOST = "127.0.0.1"
PORT = 17000

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/hello")
def hello():
    return "Hello, World"

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)