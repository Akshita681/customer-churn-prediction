📊 Customer Churn Prediction
📌 Project Overview

This project focuses on predicting customer churn using machine learning techniques. Customer churn refers to when a customer stops using a company’s service. Identifying churn in advance helps businesses take proactive actions to retain customers.

The project implements:

A custom Perceptron (M-P neuron) model
A Logistic Regression model using scikit-learn

📂 Dataset
The dataset used in this project:
File: churn.csv
Contains customer-related features and a churn indicator
The target column is automatically detected (case-insensitive search for "churn")
⚙️ Features of the Project
Automatic detection of the target (churn) column
Data preprocessing:
Handling missing values
Selecting numeric features
Feature scaling using StandardScaler
Model training and evaluation using:
Perceptron (custom implementation)
Logistic Regression
Performance metrics:
Classification Report (Precision, Recall, F1-score)
ROC-AUC Score

🧠 Models Used
1. Perceptron (Custom Implementation)
Implements a basic neural model (M-P neuron)
Uses a step activation function
Trained using iterative weight updates
2. Logistic Regression
Uses sigmoid function for probability estimation
Handles class imbalance using class_weight="balanced"

🚀 How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/customer-churn.git
cd customer-churn
2. Install Dependencies
pip install numpy pandas scikit-learn
3. Run the Script
python churn_prediction.py

📊 Output

The script prints:

Dataset information (shape, columns, sample rows)
Model performance:
Perceptron Results
Classification report
ROC-AUC score
Logistic Regression Results
Classification report
ROC-AUC score

🛠️ Technologies Used
Python 🐍
NumPy
Pandas
Scikit-learn
📈 Future Improvements
Add more advanced models (Random Forest, XGBoost)
Perform feature engineering
Hyperparameter tuning
Add visualization (ROC curves, confusion matrix)
Deploy as a web app
🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

📜 License

This project is open-source and available under the MIT License.
