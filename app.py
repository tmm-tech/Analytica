from flask import Flask
from prediction import predict_value

app = Flask(__name__)


@app.route('/')
def home():
    print(__name__)
    return '<h1>Analytica</h1>'


@app.route('/prediction/<stock>', methods=['POST'])
def predict(stock):

    result = predict_value(stock)

    return result


if __name__ == '__main__':
    app.run()
