# Salary Prediction App

## Overview
This Streamlit application uses polynomial regression to predict salary based on position level. The app allows users to interactively adjust the polynomial degree and see how it affects the prediction accuracy. Salary predictions are displayed in Indian Rupees (₹), formatted in lakhs and crores for better readability.

## Features
- Interactive data visualization of salary trends
- Adjustable polynomial regression degree via slider
- Real-time prediction based on position level
- Performance metrics (R² score and accuracy percentage)
- Comparison of different polynomial regression models
- Salary values displayed in Indian Rupees (₹)
- Formatted in lakhs and crores following Indian numerical notation

## Dataset
The application uses the Position_Salaries dataset, which contains information about position levels and corresponding salaries. The original dataset has salaries in USD, which are converted to INR within the application.

## Technical Details
- **Language**: Python 3.x
- **Libraries**:
  - Streamlit: For the interactive web application
  - Pandas: For data manipulation
  - NumPy: For numerical operations
  - Matplotlib: For data visualization
  - Scikit-learn: For machine learning models

## Installation

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Steps
1. Clone this repository or download the source code
   ```
   git clone [<repository-url>](https://github.com/prathameshatkare/Polynomial_Regression_App)
   cd salary-prediction-app
   ```

2. Install the required packages
   ```
   pip install -r requirements.txt
   ```
   
   Or install them individually:
   ```
   pip install streamlit pandas numpy matplotlib scikit-learn
   ```

3. Run the application
   ```
   streamlit run app.py
   ```

## Usage
1. The application will open in your default web browser
2. Adjust the polynomial degree using the slider in the sidebar
3. Observe how the model's predictions change with different polynomial degrees
4. Use the "Select your position level" slider to predict salary for a specific level
5. Check the "Compare Different Polynomial Degrees" option to see a side-by-side comparison of models

## Model Explanation
The application uses polynomial regression to predict salaries. Polynomial regression is an extension of linear regression that adds polynomial terms to the model to capture non-linear relationships.

The model works by:
1. Transforming the input features into polynomial features
2. Applying linear regression to the transformed features
3. The degree of the polynomial determines the complexity of the model:
   - Degree 1: Simple linear regression
   - Higher degrees: More complex curves that can fit non-linear data

## Files
- `app.py`: The main Streamlit application
- `requirements.txt`: List of required Python packages

## Future Improvements
- Add more feature variables for more accurate predictions
- Implement other regression algorithms for comparison
- Add data upload capability for custom datasets
- Include data preprocessing options
- Add cross-validation for better model evaluation

## License
[MIT License](LICENSE)

## Contact
For questions or feedback, please contact [Your Name](mailto:prathmeshatkare07@gmail.com)
