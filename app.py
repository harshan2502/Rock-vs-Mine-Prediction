from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained machine learning model (make sure you have the model file in the same folder)
model = pickle.load(open('rock_mine_model.pkl', 'rb'))

# Home page route (this will show the form to enter data)
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read the textarea input as a single string
        raw_input = request.form['features']
        
        # Split the string by commas and remove extra spaces
        input_data = [float(x.strip()) for x in raw_input.split(',') if x.strip()]
        
        # Check if exactly 60 values are entered
        if len(input_data) != 60:
            raise ValueError("Please enter exactly 60 values.")

        # Convert the input data into a numpy array (required by the model)
        final_features = np.array(input_data).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(final_features)[0]
        result = "Mine" if prediction == 'M' else "Rock"

        return render_template('index.html', prediction_text=f'The object is predicted to be a: {result}')
    
    except Exception as e:
        # Handle any errors (like missing inputs) and show them to the user
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Start the Flask app (this will run the server and allow users to access it in a browser)
if __name__ == "__main__":
    app.run(debug=True)
