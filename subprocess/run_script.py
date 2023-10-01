# server.py

from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run_script', methods=['POST'])
def run_script():
    try:
        # Run the Python script using subprocess and capture the output
        result = subprocess.check_output(['python', 'train.py'], text=True)
        
        # Parse the JSON result
        result_data = json.loads(result)
        
        if 'accuracy' in result_data:
            accuracy = result_data['accuracy']
        else:
            accuracy = None

        # Return the result as JSON
        return jsonify({'success': True, 'accuracy': accuracy})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
