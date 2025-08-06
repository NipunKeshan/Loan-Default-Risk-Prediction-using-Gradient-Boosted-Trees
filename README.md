# Loan Default Risk Prediction using Gradient Boosted Trees

## 🎯 Project Overview

This project implements a comprehensive machine learning solution for predicting loan default risk using gradient boosted trees and other advanced algorithms. The system includes enhanced data preprocessing, feature engineering, feature selection, model training & tuning, and a user-friendly Streamlit web interface.

## 🚀 Features

### Data Processing & Analysis
- **Enhanced Data Preprocessing**: Outlier detection, data quality checks, and missing value handling
- **Feature Engineering**: Creation of 10+ new meaningful features including financial ratios and customer segments
- **Feature Selection**: Multiple methods including correlation analysis, F-score, and recursive feature elimination
- **Comprehensive EDA**: Interactive visualizations and statistical analysis

### Machine Learning Pipeline
- **Multiple Algorithms**: XGBoost, Random Forest, and Logistic Regression
- **Cross-Validation**: Stratified K-fold validation for robust model evaluation
- **Hyperparameter Tuning**: Grid search optimization for best performance
- **Model Comparison**: Systematic evaluation and selection of best performing model

### Web Interface
- **Interactive Streamlit App**: User-friendly interface for loan risk prediction
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Risk Factor Analysis**: Detailed breakdown of contributing risk factors
- **Data Visualization**: Interactive charts and dashboards
- **Model Information**: Performance metrics and feature importance

## 📁 Project Structure

```
├── main.ipynb              # Enhanced Jupyter notebook with complete ML pipeline
├── streamlit_app.py        # Streamlit web application
├── bank.csv               # Dataset
├── requirements.txt       # Python dependencies
├── loan_prediction_model.pkl  # Trained model (generated after running notebook)
└── README.md             # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project
```bash
git clone <repository-url>
cd Loan-Default-Risk-Prediction-using-Gradient-Boosted-Trees
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Jupyter Notebook
1. Open Jupyter Notebook or VS Code
2. Run all cells in `main.ipynb` to train the model
3. This will generate `loan_prediction_model.pkl`

### Step 4: Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

## 🎮 How to Use

### Training the Model
1. Open `main.ipynb` in Jupyter or VS Code
2. Run all cells sequentially
3. The notebook will:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Engineer new features
   - Select optimal features
   - Train multiple models
   - Perform hyperparameter tuning
   - Save the best model

### Using the Web Interface
1. Navigate to the Streamlit app (usually http://localhost:8501)
2. Use the sidebar to navigate between pages:
   - **🔮 Prediction**: Enter customer data for risk prediction
   - **📊 Data Analysis**: Explore the dataset with interactive visualizations
   - **🤖 Model Info**: View model performance and technical details

### Making Predictions
1. Go to the Prediction page
2. Fill in customer information:
   - Personal details (age, experience, family size, education)
   - Financial information (income, credit card spending, mortgage)
   - Account information (securities, CD, online banking, credit card)
3. Click "Predict Loan Default Risk"
4. View the risk assessment with probability and risk factors

## 📊 Model Performance

The enhanced model achieves:
- **Multiple Algorithm Comparison**: XGBoost, Random Forest, Logistic Regression
- **Cross-Validation**: 5-fold stratified validation
- **Hyperparameter Optimization**: Grid search for optimal parameters
- **Feature Engineering**: 10+ new features improving prediction accuracy
- **Comprehensive Evaluation**: ROC-AUC, precision, recall, F1-score

## 🔧 Technical Details

### Feature Engineering
- Age and experience groups
- Income categories
- Credit card usage intensity
- Financial ratios (CC-to-income, mortgage-to-income)
- Financial health score
- Digital banking usage
- Investment portfolio indicator
- High-value customer flag

### Feature Selection Methods
1. Correlation analysis with target variable
2. Univariate feature selection (F-score)
3. Recursive feature elimination (RFE)
4. Random Forest feature importance
5. Combined approach for optimal feature set

### Model Training
- Stratified train-test split (80:20)
- Cross-validation for model selection
- Grid search hyperparameter tuning
- Multiple evaluation metrics
- Model interpretation and feature importance

## 📈 Results Visualization

The system provides comprehensive visualizations:
- ROC curves and AUC scores
- Confusion matrices
- Feature importance plots
- Prediction probability distributions
- Cross-validation score comparisons
- Interactive correlation heatmaps
- Risk factor analysis charts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure you've run the complete notebook first
   - Check that `loan_prediction_model.pkl` exists

2. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

3. **Streamlit app won't start**
   - Check that Streamlit is installed
   - Verify all dependencies are available
   - Try running with `python -m streamlit run streamlit_app.py`

4. **Data file not found**
   - Ensure `bank.csv` is in the project directory
   - Check file permissions

### Support
For issues or questions, please check the troubleshooting section or create an issue in the repository.

## 🔮 Future Enhancements

- Advanced feature engineering with automated feature selection
- Deep learning models (Neural Networks)
- Real-time model monitoring and drift detection
- A/B testing framework for model comparison
- API endpoints for integration with banking systems
- Advanced explainability with SHAP values
- Mobile-responsive interface
- Multi-language support

---

**Note**: This model is for educational and demonstration purposes. In production environments, additional considerations such as regulatory compliance, fairness, and bias testing should be implemented.
