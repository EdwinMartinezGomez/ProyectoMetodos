import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from flask import Flask, jsonify
from model.FixedPoint import fixedPoint

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Hola, Mundo!"})

@app.route('/puntofijo', methods=['GET'])
def fixed_point():
    aux = fixedPoint('np.sin(x)', '(1)**(1/2)', 0.5, 0.0001, 1000, 1)
    return jsonify({"message": aux})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)