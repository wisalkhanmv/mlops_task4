from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler used for training
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json['data']
    
    # Convert input data to numpy array and scale it
    input_data = scaler.transform([data])
    
    # Make predictions using the loaded model
    prediction = model.predict(input_data)
    
    # Map the predicted class index to class name
    class_names = ['setosa', 'versicolor', 'virginica']
    predicted_class = class_names[prediction[0]]
    
    # Return the prediction as JSON response
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
