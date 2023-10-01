document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('typingForm');
    const resultElement = document.getElementById('result');

    form.addEventListener('submit', function (event) {
        event.preventDefault();

        // Get the input data from the form
        const formData = new FormData(form);
        const inputData = {};
        formData.forEach((value, key) => {
            inputData[key] = value;
        });

        // Send the input data to the Python script
        fetch('run_python_script.py', {
            method: 'POST',
            body: JSON.stringify(inputData),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            // Display the result on the HTML page
            if (data.result === 0) {
                resultElement.textContent = "Fraud person";
            } else {
                resultElement.textContent = "Original person";
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
