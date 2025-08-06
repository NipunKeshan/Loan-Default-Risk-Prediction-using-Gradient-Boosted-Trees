import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model_package = joblib.load('loan_prediction_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("Model file not found. Please run the training notebook first.")
        return None

@st.cache_data
def load_data():
    """Load the dataset for analysis"""
    try:
        df = pd.read_csv('bank.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure bank.csv is in the same directory.")
        return None

def engineer_features(input_data, label_encoders):
    """Apply feature engineering to input data"""
    df = input_data.copy()
    
    # Age Groups
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 30, 40, 50, 100], 
                            labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    
    # Experience Groups
    df['Experience_Group'] = pd.cut(df['Experience'], 
                                   bins=[-1, 5, 15, 25, 100], 
                                   labels=['Beginner', 'Intermediate', 'Experienced', 'Expert'])
    
    # Income Groups
    df['Income_Group'] = pd.cut(df['Income'], 
                               bins=[0, 50, 100, 150, 1000], 
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Credit Card Usage Intensity
    df['CC_Usage_Intensity'] = pd.cut(df['CCAvg'], 
                                     bins=[0, 1, 2, 3, 10], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Financial Ratios
    df['Income_per_Family_Member'] = df['Income'] / df['Family']
    df['CC_to_Income_Ratio'] = df['CCAvg'] / (df['Income'] + 1)
    df['Mortgage_to_Income_Ratio'] = df['Mortgage'] / (df['Income'] + 1)
    
    # Financial Health Score
    df['Financial_Health_Score'] = (
        df['Income'] * 0.4 + 
        df['CCAvg'] * 0.2 + 
        df['Education'] * 0.2 - 
        df['Mortgage'] * 0.1 + 
        (df['Securities Account'] + df['CD Account']) * 0.1
    )
    
    # Experience vs Age difference
    df['Experience_Age_Diff'] = df['Experience'] - (df['Age'] - 22)
    
    # Digital Banking User
    df['Digital_User'] = ((df['Online'] == 1) | (df['CreditCard'] == 1)).astype(int)
    
    # Investment Portfolio
    df['Has_Investment'] = ((df['Securities Account'] == 1) | (df['CD Account'] == 1)).astype(int)
    
    # High Value Customer
    df['High_Value_Customer'] = ((df['Income'] > 100) & (df['Family'] >= 3)).astype(int)
    
    # Encode categorical variables
    categorical_columns = ['Age_Group', 'Experience_Group', 'Income_Group', 'CC_Usage_Intensity']
    for col in categorical_columns:
        if col in df.columns and col in label_encoders:
            df[col] = label_encoders[col].transform(df[col].astype(str))
    
    return df

def make_prediction(model_package, input_data):
    """Make prediction using the trained model"""
    # Engineer features
    df_features = engineer_features(input_data, model_package['label_encoders'])
    
    # Select only the features used in training
    selected_features = model_package['selected_features']
    X = df_features[selected_features]
    
    # Scale features if needed (for logistic regression)
    if model_package['model_name'] == 'Logistic Regression':
        X_scaled = model_package['scaler'].transform(X)
        prediction = model_package['model'].predict(X_scaled)[0]
        prediction_proba = model_package['model'].predict_proba(X_scaled)[0]
    else:
        prediction = model_package['model'].predict(X)[0]
        prediction_proba = model_package['model'].predict_proba(X)[0]
    
    return prediction, prediction_proba

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 Loan Default Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # Load model and data
    model_package = load_model()
    df = load_data()
    
    if model_package is None or df is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🔮 Prediction", "📊 Data Analysis", "🤖 Model Info"])
    
    if page == "🔮 Prediction":
        prediction_page(model_package)
    elif page == "📊 Data Analysis":
        analysis_page(df)
    else:
        model_info_page(model_package)

def prediction_page(model_package):
    st.header("🔮 Loan Default Risk Prediction")
    st.write("Enter customer information to predict loan default risk.")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", min_value=18, max_value=80, value=35)
        experience = st.slider("Years of Experience", min_value=0, max_value=50, value=10)
        family = st.selectbox("Family Size", options=[1, 2, 3, 4, 5], index=2)
        education = st.selectbox("Education Level", options=[1, 2, 3], 
                                format_func=lambda x: {1: "Undergraduate", 2: "Graduate", 3: "Advanced/Professional"}[x])
    
    with col2:
        st.subheader("Financial Information")
        income = st.number_input("Annual Income (in thousands)", min_value=0, max_value=500, value=50)
        ccavg = st.number_input("Average Credit Card Spending per Month", min_value=0.0, max_value=20.0, value=1.5, step=0.1)
        mortgage = st.number_input("Mortgage Amount", min_value=0, max_value=1000, value=0)
    
    st.subheader("Account Information")
    col3, col4 = st.columns(2)
    
    with col3:
        securities_account = st.checkbox("Securities Account")
        cd_account = st.checkbox("CD Account")
    
    with col4:
        online = st.checkbox("Online Banking")
        credit_card = st.checkbox("Credit Card")
    
    # Prediction button
    if st.button("🔍 Predict Loan Default Risk", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Experience': [experience],
            'Income': [income],
            'Family': [family],
            'CCAvg': [ccavg],
            'Education': [education],
            'Mortgage': [mortgage],
            'Securities Account': [1 if securities_account else 0],
            'CD Account': [1 if cd_account else 0],
            'Online': [1 if online else 0],
            'CreditCard': [1 if credit_card else 0]
        })
        
        # Make prediction
        prediction, prediction_proba = make_prediction(model_package, input_data)
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        risk_probability = prediction_proba[1] * 100
        
        if prediction == 1:
            st.markdown(f'''
            <div class="prediction-result high-risk">
                ⚠️ HIGH RISK: This customer is likely to default on the loan<br>
                Risk Probability: {risk_probability:.1f}%
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="prediction-result low-risk">
                ✅ LOW RISK: This customer is unlikely to default on the loan<br>
                Risk Probability: {risk_probability:.1f}%
            </div>
            ''', unsafe_allow_html=True)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk Probability (%)"},
            delta = {'reference': 50},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "lightcoral"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors analysis
        st.subheader("📈 Risk Factor Analysis")
        
        # Calculate risk factors
        risk_factors = []
        
        if income < 50:
            risk_factors.append("Low income level")
        if ccavg > 3:
            risk_factors.append("High credit card spending")
        if mortgage > 200:
            risk_factors.append("High mortgage amount")
        if not securities_account and not cd_account:
            risk_factors.append("No investment accounts")
        if family > 3 and income < 100:
            risk_factors.append("Large family with moderate income")
        
        if risk_factors:
            st.warning("⚠️ Risk Factors Identified:")
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.success("✅ No major risk factors identified")

def analysis_page(df):
    st.header("📊 Data Analysis Dashboard")
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        loan_rate = (df['Personal Loan'].sum() / len(df)) * 100
        st.metric("Loan Approval Rate", f"{loan_rate:.1f}%")
    with col3:
        avg_income = df['Income'].mean()
        st.metric("Average Income", f"${avg_income:.0f}K")
    with col4:
        avg_age = df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.0f} years")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Loan distribution by income
        fig = px.histogram(df, x='Income', color='Personal Loan', 
                          title='Income Distribution by Loan Status',
                          labels={'Personal Loan': 'Loan Status'},
                          nbins=30)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age vs Experience scatter
        fig = px.scatter(df, x='Age', y='Experience', color='Personal Loan',
                        title='Age vs Experience by Loan Status',
                        labels={'Personal Loan': 'Loan Status'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                   title="Feature Correlation Matrix")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def model_info_page(model_package):
    st.header("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.write(f"**Algorithm:** {model_package['model_name']}")
        st.write(f"**Performance (ROC-AUC):** {model_package['performance']:.4f}")
        st.write(f"**Number of Features:** {len(model_package['selected_features'])}")
    
    with col2:
        st.subheader("Model Performance")
        # Performance gauge
        performance_pct = model_package['performance'] * 100
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = performance_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Model Accuracy (%)"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightcoral"},
                        {'range': [70, 85], 'color': "yellow"},
                        {'range': [85, 95], 'color': "lightgreen"},
                        {'range': [95, 100], 'color': "green"}]}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Selected Features")
    features_df = pd.DataFrame(model_package['selected_features'], columns=['Feature'])
    st.dataframe(features_df, use_container_width=True)
    
    st.subheader("How to Use This Model")
    st.write("""
    1. **Data Input**: Enter customer information in the Prediction page
    2. **Feature Engineering**: The model automatically creates additional features from your input
    3. **Prediction**: Get instant risk assessment with probability scores
    4. **Interpretation**: Review risk factors and recommendations
    
    **Note**: This model is trained on historical banking data and provides risk assessments 
    to support decision-making. Always consider additional factors and regulatory requirements 
    in actual lending decisions.
    """)

if __name__ == "__main__":
    main()
