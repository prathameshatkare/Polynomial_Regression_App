import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def main():
    st.title("Salary Prediction App")
    st.write("This app predicts salary based on position level using polynomial regression")
    
    # Load data
    @st.cache_data
    def load_data():
        data = pd.read_csv('https://raw.githubusercontent.com/yash240990/Python/master/Position_Salaries.csv')
        return data
    
    data = load_data()
    
    # Display dataset
    st.subheader("Dataset")
    st.dataframe(data)
    
    # Prepare data
    x = data.Level.values.reshape(-1, 1)
    y = data.Salary.values
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    degree = st.sidebar.slider("Polynomial Degree", min_value=1, max_value=10, value=4, step=1)
    
    # Create polynomial features
    poly_reg = PolynomialFeatures(degree=degree)
    x_poly = poly_reg.fit_transform(x)
    
    # Train model
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    pred_values = lin_reg.predict(x_poly)
    
    # Calculate accuracy
    accuracy = r2_score(y, pred_values)
    
    # Display metrics
    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{accuracy:.4f}")
    with col2:
        st.metric("Accuracy", f"{int(accuracy*100)}%")
    
    # Plot results
    st.subheader(f"Polynomial Regression Results (Degree = {degree})")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual values
    ax.plot(x, y, color='red', label='Actual Salaries')
    ax.scatter(x, y, color='red')
    
    # Create a smooth curve for predicted values
    x_grid = np.arange(min(x)[0], max(x)[0] + 0.1, 0.1).reshape(-1, 1)
    x_grid_poly = poly_reg.fit_transform(x_grid)
    y_grid = lin_reg.predict(x_grid_poly)
    
    # Plot predicted values
    ax.plot(x_grid, y_grid, color='green', label='Predicted Salaries')
    ax.scatter(x, pred_values, color='green')
    
    ax.set_title(f'Actual vs Predicted Using Polynomial Regression (Degree = {degree})')
    ax.set_xlabel('Position Level')
    ax.set_ylabel('Salary')
    ax.legend()
    
    st.pyplot(fig)
    
    # Salary prediction section
    st.subheader("Predict Your Salary")
    user_level = st.slider("Select your position level", min_value=1, max_value=10, value=5, step=1)
    
    # Make prediction
    level_poly = poly_reg.fit_transform([[user_level]])
    predicted_salary = lin_reg.predict(level_poly)[0]
    
    st.success(f"Predicted Salary at Level {user_level}: ${int(predicted_salary):,}")
    
    # Compare different polynomial degrees
    if st.checkbox("Compare Different Polynomial Degrees"):
        st.subheader("Comparison of Different Polynomial Degrees")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot actual values
        ax.scatter(x, y, color='red', label='Actual Data')
        
        # Create smooth curves for different degrees
        x_grid = np.arange(min(x)[0], max(x)[0] + 0.1, 0.1).reshape(-1, 1)
        
        degrees = [1, 2, 3, 4]
        colors = ['blue', 'green', 'purple', 'orange']
        
        for i, deg in enumerate(degrees):
            poly_features = PolynomialFeatures(degree=deg)
            x_poly_comp = poly_features.fit_transform(x)
            x_grid_poly_comp = poly_features.fit_transform(x_grid)
            
            lin_reg_comp = LinearRegression()
            lin_reg_comp.fit(x_poly_comp, y)
            
            y_grid_comp = lin_reg_comp.predict(x_grid_poly_comp)
            
            ax.plot(x_grid, y_grid_comp, color=colors[i], label=f'Degree {deg}')
            
            # Calculate and display R² for each degree
            y_pred_comp = lin_reg_comp.predict(x_poly_comp)
            r2 = r2_score(y, y_pred_comp)
            st.write(f"Degree {deg} - R² Score: {r2:.4f} - Accuracy: {int(r2*100)}%")
        
        ax.set_title('Comparison of Different Polynomial Degrees')
        ax.set_xlabel('Position Level')
        ax.set_ylabel('Salary')
        ax.legend()
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()