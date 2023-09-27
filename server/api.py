from flask import Flask, request
from flask_cors import CORS

from db import add_user, authenticate
#from DNN import  check_pattern

app = Flask(__name__)
CORS(app)

@app.route('/register', methods=['POST'])
def register():
    body = request.json
    email, password = body['email'], body['password']
    add_user(email, password)
    return {
        'status': 'success',
    }

@app.route('/login', methods=['POST'])
def login():
    body = request.json
    email, password = body['email'], body['password']
    status = authenticate(email, password)
    return {
        'status': status,
    }

@app.route('/recover', methods=['POST'])
def recover():
    body = request.json
    email = body['email']
    pattern= body['pattern']
    #status= check_pattern(pattern)
    status = "Genuine"

    return {
        'status': status,
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0')
