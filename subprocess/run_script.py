# server.py

from flask import Flask, request, jsonify
import subprocess
import json
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def run_script():
    try:
        # Run the Python script using subprocess and capture the output
        print('------------1-----------------')
        result = subprocess.check_output(['python', 'train.py'], text=True)
        
        # Parse the JSON result
        print(result,'------------2-----------------')
        result_data = json.loads(result)
        print('result-data-----',result_data)
        if 'accuracy' in result_data:
            accuracy = result_data["accuracy"]
            print('accuracy got it ')
        else:
            accuracy = None
            print('------------3-----------------')
        # Return the result as JSON
        print('accuracy')
        return jsonify({'success': True, 'result': accuracy})
    except Exception as e:
        return jsonify({'success': False, 'result': str(e)})


if __name__ == '__main__':
    app.run(debug=False)
    #run_script()