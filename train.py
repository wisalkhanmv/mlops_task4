from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)  # Increase max_iter to 1000
model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)
