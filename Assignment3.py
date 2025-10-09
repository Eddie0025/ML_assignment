import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


# Q1: K-Fold Cross Validation (USA House Price Dataset)

# Load the dataset (Assume 'USA_House_Price.csv' is available)
data = pd.read_csv('USA_House_Price.csv')

# Split into features (X) and target (y)
X = data.drop('price', axis=1)  # Assuming 'price' is the target column
y = data['price']

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Fold Cross Validation with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Variable to store R2 scores for each fold
r2_scores = []

# Cross-validation loop
for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Initialize Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute R2 score and store it
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Print average R2 score across all folds
print(f"Average R2 score from 5-fold cross-validation: {np.mean(r2_scores):.4f}")


# Q2: Validation Set with Gradient Descent

# Custom gradient descent for Linear Regression (as requested)
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Initialize weights (theta) to zeros
    for epoch in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (2/m) * X.T.dot(errors)
        theta -= learning_rate * gradient  # Update the weights
    return theta

# Prepare data (Assuming the USA House Price Dataset is used)
# Normalize the features using StandardScaler
X_scaled = scaler.fit_transform(X)

# Add a bias term (intercept) to the features (X) - by adding a column of ones
X_scaled_with_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_with_bias, y, test_size=0.2, random_state=42)

# Run gradient descent
theta = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)

# Predictions on the test set
y_pred = X_test.dot(theta)

# Compute R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 score with Gradient Descent: {r2:.4f}")


# Q3: Car Price Prediction Dataset

# Load the car price dataset (Assume 'imports-85.data' is available)
car_df = pd.read_csv('imports-85.data', names=[
    'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
    'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
    'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
    'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
])

# Data Preprocessing for Car Dataset
# Handle missing values (fill with mean or mode)
car_df = car_df.replace('?', np.nan)
car_df['normalized-losses'] = car_df['normalized-losses'].fillna(car_df['normalized-losses'].mean())
car_df['num-of-doors'] = car_df['num-of-doors'].fillna(car_df['num-of-doors'].mode()[0])
car_df['horsepower'] = car_df['horsepower'].fillna(car_df['horsepower'].mean())
car_df['bore'] = car_df['bore'].fillna(car_df['bore'].mean())
car_df['stroke'] = car_df['stroke'].fillna(car_df['stroke'].mean())

# Convert categorical columns to numerical using LabelEncoder
label_encoders = {}
categorical_columns = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']

for column in categorical_columns:
    le = LabelEncoder()
    car_df[column] = le.fit_transform(car_df[column])
    label_encoders[column] = le

# Scale the features using StandardScaler
features = car_df.drop('price', axis=1)
target = car_df['price']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA (Principal Component Analysis)
pca = PCA(n_components=0.95)  # Keep 95% of variance
features_pca = pca.fit_transform(features_scaled)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Compute R2 score
r2 = r2_score(y_test, y_pred)
print(f"R2 score for Car Price Prediction Model with PCA: {r2:.4f}")

# --------------------------------------------------
# Name: Adityavir Singh Randhawa
# Roll No: 102483009
# Sub Group: 3C53
# --------------------------------------------------
