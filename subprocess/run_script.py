# server.py

from flask import Flask, request, jsonify
from flask_cors import CORS  
import subprocess
import json
app = Flask(__name__)
CORS(app)

@app.route('/train', methods=['POST'])
def run_script():
    try:

        result = subprocess.check_output(['python', 'train.py'], text=True)
        result_data = json.loads(result)
        if 'accuracy' in result_data:
            accuracy = result_data["accuracy"]
        else:
            accuracy = None
        return jsonify({'success': True, 'result': accuracy})
    except Exception as e:
        return jsonify({'success': False, 'result': str(e)})


if __name__ == '__main__':
    app.run(debug=False)
