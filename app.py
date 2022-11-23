from flask import Flask
from prediction import predict_value

app = Flask(__name__)



@app.route('/', methods=['POST', 'GET'])
def home():
    print(__name__)
    return 'Analytica'


@app.route('/prediction/<stock>', methods=['POST', 'GET'])
def predict(stock):
    try:
        result = predict_value(stock)
        return str(result)
    except EOFError as e:
        print("as: " + str(e))


if __name__ == '__main__':
    app.run()
