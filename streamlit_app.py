import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🏥 Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-box {
        padding: 20px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        padding: 20px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 10px 0;
    }
    .danger-box {
        padding: 20px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        padding: 20px;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "https://yourusername.pythonanywhere.com"  # ⚠️ CHANGE THIS TO YOUR PYTHONANYWHERE URL

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/hospital.png", width=80)
    st.title("🏥 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "🔮 Prediction", "📊 History", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📱 Quick Info")
    st.info("This app uses AI to predict diabetes risk based on health parameters.")
    
    # API Status Check
    st.markdown("### 🔗 API Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Connected")
        else:
            st.error("❌ API Error")
    except:
        st.warning("⚠️ API Offline")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Your Name**")
    st.markdown("Data Science Project 2024")

# HOME PAGE
if page == "🏠 Home":
    st.title("🏥 Diabetes Prediction System")
    st.markdown("### Welcome to AI-Powered Diabetes Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accurate</h3>
            <p>Machine learning model trained on 100,000+ records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Fast</h3>
            <p>Get instant predictions in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔒 Secure</h3>
            <p>Your health data is never stored</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Step 1️⃣: Enter Your Information
        - Gender
        - Age
        - Medical history (hypertension, heart disease)
        - Lifestyle (smoking history)
        - Body metrics (BMI)
        - Lab results (HbA1c, blood glucose)
        """)
    
    with col2:
        st.markdown("""
        #### Step 2️⃣: Get AI Prediction
        - Instant risk assessment
        - Probability score
        - Personalized recommendations
        - Visual analytics
        """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Important Disclaimer")
    st.markdown("""
    <div class="warning-box">
        <strong>Medical Disclaimer:</strong><br>
        This tool is for educational purposes only and should NOT replace professional medical advice.
        Always consult with a qualified healthcare provider for medical diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 🤖 Model Information")
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", model_info.get('model_name', 'N/A'))
            with col2:
                st.metric("Accuracy", model_info.get('accuracy', 'N/A'))
            with col3:
                st.metric("ROC-AUC Score", model_info.get('roc_auc', 'N/A'))
    except:
        st.warning("⚠️ Unable to fetch model information")

# PREDICTION PAGE
elif page == "🔮 Prediction":
    st.title("🔮 Diabetes Risk Prediction")
    st.markdown("### Enter Your Health Information")
    
    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Personal Information")
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                help="Select your biological gender"
            )
            
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=45,
                step=1,
                help="Enter your age in years"
            )
            
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=10.0,
                max_value=100.0,
                value=25.0,
                step=0.1,
                help="Calculate: weight(kg) / height(m)²"
            )
            
            st.markdown("#### 🚬 Lifestyle")
            smoking_history = st.selectbox(
                "Smoking History",
                ["never", "former", "current", "not current", "ever", "No Info"],
                help="Select your smoking history"
            )
        
        with col2:
            st.markdown("#### 🏥 Medical History")
            hypertension = st.selectbox(
                "Hypertension (High Blood Pressure)",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Do you have hypertension?"
            )
            
            heart_disease = st.selectbox(
                "Heart Disease",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Do you have any heart disease?"
            )
            
            st.markdown("#### 🧪 Lab Results")
            HbA1c_level = st.number_input(
                "HbA1c Level (%)",
                min_value=3.0,
                max_value=15.0,
                value=5.7,
                step=0.1,
                help="Hemoglobin A1c level (Normal: < 5.7%)"
            )
            
            blood_glucose_level = st.number_input(
                "Blood Glucose Level (mg/dL)",
                min_value=50,
                max_value=500,
                value=100,
                step=1,
                help="Fasting blood glucose (Normal: 70-100 mg/dL)"
            )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Diabetes Risk", use_container_width=True)
    
    # Process prediction
    if submitted:
        with st.spinner("🔄 Analyzing your health data..."):
            # Prepare data
            input_data = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "smoking_history": smoking_history,
                "bmi": bmi,
                "HbA1c_level": HbA1c_level,
                "blood_glucose_level": blood_glucose_level
            }
            
            try:
                # Make API request
                response = requests.post(
                    f"{API_URL}/predict",
                    json=input_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Save to history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'prediction': result['prediction'],
                        'probability': result['probability_score'],
                        'risk_level': result['risk_level']
                    })
                    
                    st.success("✅ Prediction Complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    # Main prediction
                    if result['prediction'] == 'Diabetes':
                        st.markdown(f"""
                        <div class="danger-box">
                            <h2 style="color: #dc3545;">⚠️ {result['prediction']}</h2>
                            <h3>Risk Level: {result['risk_level']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <h2 style="color: #28a745;">✅ {result['prediction']}</h2>
                            <h3>Risk Level: {result['risk_level']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "No Diabetes Probability",
                            result['probability']['no_diabetes'],
                            delta=None
                        )
                    with col2:
                        st.metric(
                            "Diabetes Probability",
                            result['probability']['diabetes'],
                            delta=None
                        )
                    with col3:
                        st.metric(
                            "Risk Score",
                            f"{result['probability_score']:.2%}",
                            delta=None
                        )
                    
                    # Probability gauge chart
                    st.markdown("### 📈 Risk Visualization")
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = result['probability_score'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Diabetes Risk Score (%)", 'font': {'size': 24}},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': result['risk_color']},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#d4edda'},
                                {'range': [30, 60], 'color': '#fff3cd'},
                                {'range': [60, 100], 'color': '#f8d7da'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 60
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Input summary
                    st.markdown("### 📝 Your Input Summary")
                    input_df = pd.DataFrame([result['input_data']])
                    st.dataframe(input_df.T, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Health Recommendations")
                    for i, rec in enumerate(result['recommendations'], 1):
                        st.markdown(f"{i}. {rec}")
                    
                    # Download report
                    st.markdown("---")
                    report_data = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Prediction': result['prediction'],
                        'Probability Score': result['probability_score'],
                        'Risk Level': result['risk_level'],
                        **result['input_data']
                    }
                    
                    report_df = pd.DataFrame([report_data])
                    csv = report_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Report (CSV)",
                        data=csv,
                        file_name=f"diabetes_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                else:
                    st.error(f"❌ API Error: {response.json().get('error', 'Unknown error')}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection Error: Unable to connect to API. Please check if API is running.")
                st.error(f"Details: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# HISTORY PAGE
elif page == "📊 History":
    st.title("📊 Prediction History")
    
    if len(st.session_state.prediction_history) > 0:
        st.markdown(f"### Total Predictions: {len(st.session_state.prediction_history)}")
        
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display table
        st.dataframe(history_df, use_container_width=True)
        
        # Statistics
        st.markdown("### 📈 Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            diabetes_count = len(history_df[history_df['prediction'] == 'Diabetes'])
            st.metric("Diabetes Predictions", diabetes_count)
        
        with col2:
            no_diabetes_count = len(history_df[history_df['prediction'] == 'No Diabetes'])
            st.metric("No Diabetes Predictions", no_diabetes_count)
        
        with col3:
            avg_probability = history_df['probability'].mean()
            st.metric("Average Risk Score", f"{avg_probability:.2%}")
        
        # Visualization
        st.markdown("### 📊 Risk Distribution")
        
        fig = px.histogram(
            history_df,
            x='risk_level',
            color='risk_level',
            title='Risk Level Distribution',
            color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete History (CSV)",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    
    else:
        st.info("📭 No prediction history yet. Make your first prediction!")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This **Diabetes Prediction System** is an AI-powered web application that predicts the risk of diabetes 
    based on various health parameters using machine learning algorithms.
    
    ### 🛠️ Technology Stack
    
    #### Frontend (Streamlit)
    - **Streamlit**: Interactive web interface
    - **Plotly**: Data visualization
    - **Pandas**: Data manipulation
    
    #### Backend (Flask API)
    - **Flask**: RESTful API framework
    - **Scikit-learn**: Machine learning models
    - **XGBoost**: Advanced gradient boosting
    - **Pandas & NumPy**: Data processing
    
    #### Deployment
    - **PythonAnywhere**: Flask API hosting
    - **Streamlit Cloud**: Frontend hosting
    
    ### 📊 Model Details
    
    The prediction model is trained on a comprehensive diabetes dataset with 100,000+ records.
    
    **Features Used:**
    - Gender
    - Age
    - Hypertension
    - Heart Disease
    - Smoking History
    - BMI (Body Mass Index)
    - HbA1c Level
    - Blood Glucose Level
    
    **Algorithms Compared:**
    1. Logistic Regression
    2. Random Forest Classifier
    3. XGBoost Classifier
    4. Support Vector Machine (SVM)
    
    The best performing model is automatically selected based on accuracy and ROC-AUC score.
    
    ### ⚠️ Disclaimer
    
    This application is developed for **educational and research purposes only**. 
    It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.
    
    **Always consult with qualified healthcare professionals for medical concerns.**
    
    ### 👨‍💻 Developer Information
    
    **Project:** Diabetes Prediction System  
    **Course:** Data Science with Python  
    **Year:** 2024  
    **Technology:** Machine Learning, Flask, Streamlit  
    
    ### 📧 Contact
    
    For questions or feedback, please contact: **your.email@example.com**
    
    ### 📝 License
    
    This project is for educational purposes. All rights reserved.
    """)
    
    st.markdown("---")
    
    # API Endpoints Documentation
    st.markdown("### 🔗 API Endpoints")
    
    st.code(f"""
# Base URL
{API_URL}

# Endpoints
GET  /              - API information
GET  /health        - Health check
GET  /model-info    - Model details
POST /predict       - Make prediction

# Example Request (POST /predict)
{{
    "gender": "Female",
    "age": 45,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 28.5,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140
}}
    """, language='json')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🏥 Diabetes Prediction System | Made with ❤️ using Streamlit & Flask</p>
    <p>© 2024 - All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
