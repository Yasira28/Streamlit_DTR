import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

# Function to generate synthetic dataset
def generate_data():
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=(100,))
    return X, y

# Function to train and evaluate the model
def train_evaluate_model(X_train, X_test, y_train, y_test, max_depth, min_samples_split):
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    return model, train_rmse, test_rmse

# Streamlit app
def main():
    st.title('Decision Tree Regression')

    # Sidebar for hyperparameters
    st.sidebar.header('Model Hyperparameters')
    max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=20, value=5)
    min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=100, value=2)

    # Generate dataset
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate model
    model, train_rmse, test_rmse = train_evaluate_model(X_train, X_test, y_train, y_test, max_depth, min_samples_split)

    # Print underfitting or overfitting
    if train_rmse < test_rmse:
        st.write("The model is possibly overfitting.")
    else:
        st.write("The model is possibly underfitting.")

    # Display metrics
    st.subheader('Model Performance')
    st.write(f'Training RMSE: {train_rmse:.4f}')
    st.write(f'Test RMSE: {test_rmse:.4f}')

    # Plotting root node and leaf node
    st.subheader('Decision Tree Visualization')
    fig, ax = plt.subplots(figsize=(50, 45))
    plot_tree(model, ax=ax, filled=True, feature_names=['X'], impurity=False, precision=2)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
