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

        result = subprocess.check_output(['python', 'test.py'], text=True)
        result_data = json.loads(result)
        if 'result' in result_data:
            result_value = result_data["result"]
        else:
            result_value = None
        return jsonify({'success': True, 'result': result_value})
    except Exception as e:
        return jsonify({'success': False, 'result': str(e)})


if __name__ == '__main__':
    app.run(debug=False)
