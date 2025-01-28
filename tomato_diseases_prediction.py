import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
path = kagglehub.dataset_download("kaustubhb999/tomatoleaf")
print("Path to dataset files:", path)

dataset_path = f"{path}/TomatoLeafDisease.csv"
data = pd.read_csv(dataset_path)

print(data.head())

X = data.drop(columns=['disease'])  
y = data['disease'] 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
sample = np.array([[0.5, 0.2, 0.8, 0.6]]) 
prediction = model.predict(sample)
print(f"Predicted Disease: {prediction[0]}")
