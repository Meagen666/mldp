import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from bs4 import BeautifulSoup
import base64

st.set_page_config(
    page_title="Smoking and Drinking Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    .main-header {
        background: #F2E2BA;
        padding: 2rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
        border: 3px solid #F2BAC9;
        box-shadow: 0 4px 15px rgba(242, 186, 201, 0.3);
    }
    .metric-card {
        background: #BAD7F2;
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid #F2BAC9;
    }
    .prediction-card {
        background: #F2BAC9;
        padding: 2rem;
        border-radius: 15px;
        color: #333;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(242, 186, 201, 0.2);
        border: 3px solid #F2E2BA;
    }
    .health-recommendation {
        background: #F2E2BA;
        padding: 1.5rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
        border: 2px solid #F2BAC9;
    }
    div.stButton {
    display: flex !important;
    justify-content: center !important;
    }
    .stButton > button {
        background: #bad7f2 !important;
        border: 3px solid #f2bac9 !important;
        border-radius: 25px !important;
        padding: 1rem 3rem !important;
        font-weight: bold !important;
        font-size: 1.2rem !important;
        transition: all 0.3s ease !important;
        display: block !important;
        margin: 2rem auto !important;
        width: 300px !important;
        text-align: center !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(242, 186, 201, 0.4) !important;
        background: #bad7f2 !important;
        border-color: #f2bac9 !important;
    }
    .sidebar .sidebar-content {
        background: #F2E2BA;
        border-right: 3px solid #F2BAC9;
    }
    .icon-text {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .tab-icon {
        margin-right: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border: 2px solid #F2BAC9;
        border-radius: 10px;
        color: #333;
        padding: 0.5rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: #F2E2BA;
        border-color: #F2BAC9;
    }
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
    }
    .stNumberInput > div > div > input {
        background: white;
        border: 2px solid #BAD7F2;
        border-radius: 8px;
    }
    .stMetric {
        background: #BAD7F2;
        border: 2px solid #F2BAC9;
        border-radius: 10px;
        padding: 1rem;
    }
    .stMarkdown {
        color: #333;
    }
    .stAlert {
        background: #F2E2BA;
        border: 2px solid #BAD7F2;
        border-radius: 10px;
    }
    .stSpinner > div {
        border-color: #F2BAC9;
        border-top-color: #BAD7F2;
    }
</style>
""", unsafe_allow_html=True)

# Models
SMOKING_MODELS = {
    "LDA Smoking": "lda_model_smoking.pkl",
    "GBC Smoking": "gbc_model_smoking.pkl"
}

DRINKING_MODELS = {
    "LDA Drinking": "lda_model_drinking.pkl", 
    "GBC Drinking": "gbc_model_drinking.pkl"
}

# Health features
HEALTH_FEATURES = {
    'age': {
        'min': 20, 'max': 80, 'default': 30, 'unit': 'years',
        'help': 'Your current age in years. Age is a key factor in health risk assessment.'
    },
    'weight': {
        'min': 30, 'max': 300, 'default': 70, 'unit': 'kg',
        'help': 'Body weight in kilograms. Used to calculate BMI along with height.'
    },
    'height': {
        'min': 100, 'max': 250, 'default': 170, 'unit': 'cm',
        'help': 'Height in centimeters. Used to calculate BMI along with weight.'
    },
    'hear_left': {
        'min': 0, 'max': 2.0, 'default': 1.0, 'unit': '',
        'help': 'Hearing capacity of left ear.'
    },
    'hear_right': {
        'min': 0, 'max': 2.0, 'default': 1.0, 'unit': '',
        'help': 'Hearing capacity of right ear.'
    },
    'sight_left': {
        'min': 0, 'max': 2.0, 'default': 1.0, 'unit': '',
        'help': 'Vision capacity of left eye.'
    },
    'sight_right': {
        'min': 0, 'max': 2.0, 'default': 1.0, 'unit': '',
        'help': 'Vision capacity of right eye.'
    },
    'SBP': {
        'min': 70, 'max': 200, 'default': 100, 'unit': 'mmHg',
        'help': 'Measures the pressure in your arteries when your heart beats, representing the maximum force exerted as blood is pumped out. Important indicator of cardiovascular health. Drinking and smoking raises SBP by narrowing arteries.'
    },
    'DBP': {
        'min': 60, 'max': 120, 'default': 70, 'unit': 'mmHg',
        'help': 'Measures the pressure between your heart beats, representing how well arteries relax and refill. Important indicator of cardiovascular health. Drinking and smoking raises DBP.'
    },
    'BLDS': {
        'min': 70, 'max': 400, 'default': 80, 'unit': 'mg/dL',
        'help': 'Measures the blood glucose level after fasting, an indicator of diabetes risk. Smoking can impair insulin function, raising BLDS while alcohol has mixed effects. Binge drinking raises BLDS, while moderate drinking may lower it.'
    },
    'tot_chole': {
        'min': 100, 'max': 400, 'default': 170, 'unit': 'mg/dL',
        'help': 'Sum of all blood cholesterol types: LDL, HDL, triglycerides and it is a key biomarker for cardiovascular health and atherosclerosis risk.'
    },
    'HDL_chole': {
        'min': 10, 'max': 120, 'default': 50, 'unit': 'mg/dL',
        'help': 'Often called the "good cholesterol" as it helps remove other forms of cholesterol from your bloodstream. So higher HDL is generally protective against heart disease. Smoking reduces HDL. Moderate alcohol (especially wine) can raise HDL.'
    },
    'LDL_chole': {
        'min': 30, 'max': 300, 'default': 90, 'unit': 'mg/dL',
        'help': 'Often called the "bad cholesterol" that builds up in arteries. High LDL is linked to heart disease. Smoking increases LDL. Alcohol may increase or have little effect, depending on quantity.'
    },
    'triglyceride': {
        'min': 30, 'max': 1000, 'default': 140, 'unit': 'mg/dL',
        'help': 'Type of fat in blood and high levels would mean higher heart disease risk. Both smoking and heavy drinking increase triglycerides. Alcohol, especially sugary drinks, is a key factor.'
    },
    'hemoglobin': {
        'min': 8, 'max': 25, 'default': 14, 'unit': 'g/dL',
        'help': 'Protein in red blood cells that carries oxygen. Low levels may indicate anemia. Smoking can reduce oxygen-carrying capacity while alcohol can affect red blood cell production.'
    },
    'serum_creatinine': {
        'min': 0.4, 'max': 20, 'default': 0.5, 'unit': 'mg/dL',
        'help': 'Waste products filtered by kidneys, high levels may indicate kidney dysfunction. Long-term alcoholism and smoking may impair kidney function.'
    },
    'SGOT_AST': {
        'min': 8, 'max': 1000, 'default': 30, 'unit': 'U/L',
        'help': 'Liver enzymes that leaks into the bloodstream when your liver or muscles are damaged. High levels suggest liver or muscle damage. AST and ALT are commonly used to access liver health. Strongly elevated in alcohol misuse. May also rise in smokers due to systemic inflammation.'
    },
    'SGOT_ALT': {
        'min': 7, 'max': 1000, 'default': 30, 'unit': 'U/L',
        'help': 'Liver enzyme more specific to liver injury. Key indication of liver inflammation or damage. Elevated in heavy drinkers. Less affected by smoking alone.'
    },
    'gamma_GTP': {
        'min': 5, 'max': 1000, 'default': 20, 'unit': 'U/L',
        'help': 'Gamma-glutamyl transpeptidase, an enzyme involved in liver and bile duct function and also oxidative stress marker. Very sensitive to alcohol intake. Often elevated in drinkers. Also slightly raised in smokers.'
    }
}

def calculate_derived_features(basic_values):
    derived = {}
    
    weight = basic_values.get('weight', 70)
    height = basic_values.get('height', 170)
    derived['BMI'] = weight / ((height / 100) ** 2)
    
    hear_left = basic_values.get('hear_left', 1.0)
    hear_right = basic_values.get('hear_right', 1.0)
    derived['avg_hearing'] = (hear_left + hear_right) / 2
    
    sight_left = basic_values.get('sight_left', 1.0)
    sight_right = basic_values.get('sight_right', 1.0)
    derived['avg_sight'] = (sight_left + sight_right) / 2
    
    sbp = basic_values.get('SBP', 130)
    dbp = basic_values.get('DBP', 80)
    derived['pulse_pressure'] = sbp - dbp
    
    hdl = basic_values.get('HDL_chole', 50)
    tot_chole = basic_values.get('tot_chole', 200)
    derived['chol_ratio'] = tot_chole / hdl if hdl > 0 else 4.0
    
    return derived

@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

def preprocess_inputs(values, derived_features):
    features = [
        values.get('age', 30),
        values.get('BLDS', 100),
        values.get('HDL_chole', 50),
        values.get('LDL_chole', 120),
        values.get('triglyceride', 150),
        values.get('hemoglobin', 14),
        values.get('serum_creatinine', 0.5),
        values.get('SGOT_AST', 25),
        values.get('SGOT_ALT', 25),
        values.get('gamma_GTP', 30),
        0,  # sex_Female
        1,  # sex_Male
        0,  # urine_protein_1.0
        0,  # urine_protein_2.0
        0,  # urine_protein_3.0
        0,  # urine_protein_4.0
        0,  # urine_protein_5.0
        0,  # urine_protein_6.0
        derived_features.get('BMI', 24.2),
        derived_features.get('avg_hearing', 1.0),
        derived_features.get('avg_sight', 1.0),
        derived_features.get('pulse_pressure', 50),
        derived_features.get('chol_ratio', 4.0)
    ]
    
    return np.array(features).reshape(1, -1)

def get_prediction_with_confidence(model, X):
    try:
        prediction = model.predict(X)[0]
        
        if isinstance(prediction, str):
            reverse_label_map = {
                'Currently Smoking': 3,  'Quit Smoking': 2, 'Never Smoked': 1,
                'Yes': 1, 'No': 0
            }
            prediction = reverse_label_map.get(prediction, prediction)
        
        try:
            probabilities = model.predict_proba(X)[0]
            confidence = max(probabilities) * 100
        except:
            confidence = 50.0
            
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0

def create_input_form(tab_key=""):
    st.markdown("### Health Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        age = st.number_input(
            "Age", 
            min_value=HEALTH_FEATURES['age']['min'], 
            max_value=HEALTH_FEATURES['age']['max'], 
            value=HEALTH_FEATURES['age']['default'],
            help=HEALTH_FEATURES['age']['help'],
            key=f"age_{tab_key}"
        )
        
        weight = st.number_input(
            "Weight (kg)", 
            min_value=HEALTH_FEATURES['weight']['min'], 
            max_value=HEALTH_FEATURES['weight']['max'], 
            value=HEALTH_FEATURES['weight']['default'],
            help=HEALTH_FEATURES['weight']['help'],
            key=f"weight_{tab_key}"
        )
        
        height = st.number_input(
            "Height (cm)", 
            min_value=HEALTH_FEATURES['height']['min'], 
            max_value=HEALTH_FEATURES['height']['max'], 
            value=HEALTH_FEATURES['height']['default'],
            help=HEALTH_FEATURES['height']['help'],
            key=f"height_{tab_key}"
        )
        
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Your biological sex for medical assessment",
            key=f"gender_{tab_key}"
        )
        
        urine_protein = st.selectbox(
            "Urine Protein Level",
            ["urine_protein_1.0", "urine_protein_2.0", "urine_protein_3.0", "urine_protein_4.0", "urine_protein_5.0", "urine_protein_6.0"],
            help="Protein level in urine, indicator of kidney function",
            key=f"urine_protein_{tab_key}"
        )
        
        st.markdown("#### Sensory Function")
        hear_left = st.number_input(
            "Left Ear Hearing", 
            min_value=float(HEALTH_FEATURES['hear_left']['min']), 
            max_value=float(HEALTH_FEATURES['hear_left']['max']), 
            value=float(HEALTH_FEATURES['hear_left']['default']),
            step=0.1,
            help=HEALTH_FEATURES['hear_left']['help'],
            key=f"hear_left_{tab_key}"
        )
        
        hear_right = st.number_input(
            "Right Ear Hearing", 
            min_value=float(HEALTH_FEATURES['hear_right']['min']), 
            max_value=float(HEALTH_FEATURES['hear_right']['max']), 
            value=float(HEALTH_FEATURES['hear_right']['default']),
            step=0.1,
            help=HEALTH_FEATURES['hear_right']['help'],
            key=f"hear_right_{tab_key}"
        )
        
        sight_left = st.number_input(
            "Left Eye Sight", 
            min_value=float(HEALTH_FEATURES['sight_left']['min']), 
            max_value=float(HEALTH_FEATURES['sight_left']['max']), 
            value=float(HEALTH_FEATURES['sight_left']['default']),
            step=0.1,
            help=HEALTH_FEATURES['sight_left']['help'],
            key=f"sight_left_{tab_key}"
        )
        
        sight_right = st.number_input(
            "Right Eye Sight", 
            min_value=float(HEALTH_FEATURES['sight_right']['min']), 
            max_value=float(HEALTH_FEATURES['sight_right']['max']), 
            value=float(HEALTH_FEATURES['sight_right']['default']),
            step=0.1,
            help=HEALTH_FEATURES['sight_right']['help'],
            key=f"sight_right_{tab_key}"
        )
        st.markdown("#### Blood Pressure")
        sbp = st.number_input(
            "Systolic BP (mmHg)", 
            min_value=HEALTH_FEATURES['SBP']['min'], 
            max_value=HEALTH_FEATURES['SBP']['max'], 
            value=HEALTH_FEATURES['SBP']['default'],
            help=HEALTH_FEATURES['SBP']['help'],
            key=f"sbp_{tab_key}"
        )
        
        dbp = st.number_input(
            "Diastolic BP (mmHg)", 
            min_value=HEALTH_FEATURES['DBP']['min'], 
            max_value=HEALTH_FEATURES['DBP']['max'], 
            value=HEALTH_FEATURES['DBP']['default'],
            help=HEALTH_FEATURES['DBP']['help'],
            key=f"dbp_{tab_key}"
        )
    
    with col2:
         
        st.markdown("#### Blood Tests")
        blds = st.number_input(
            "Blood Glucose (mg/dL)", 
            min_value=HEALTH_FEATURES['BLDS']['min'], 
            max_value=HEALTH_FEATURES['BLDS']['max'], 
            value=HEALTH_FEATURES['BLDS']['default'],
            help=HEALTH_FEATURES['BLDS']['help'],
            key=f"blds_{tab_key}"
        )
        
        tot_chole = st.number_input(
            "Total Cholesterol (mg/dL)", 
            min_value=HEALTH_FEATURES['tot_chole']['min'], 
            max_value=HEALTH_FEATURES['tot_chole']['max'], 
            value=HEALTH_FEATURES['tot_chole']['default'],
            help=HEALTH_FEATURES['tot_chole']['help'],
            key=f"tot_chole_{tab_key}"
        )
        
        hdl_chole = st.number_input(
            "HDL Cholesterol (mg/dL)", 
            min_value=HEALTH_FEATURES['HDL_chole']['min'], 
            max_value=HEALTH_FEATURES['HDL_chole']['max'], 
            value=HEALTH_FEATURES['HDL_chole']['default'],
            help=HEALTH_FEATURES['HDL_chole']['help'],
            key=f"hdl_chole_{tab_key}"
        )
        
        ldl_chole = st.number_input(
            "LDL Cholesterol (mg/dL)", 
            min_value=HEALTH_FEATURES['LDL_chole']['min'], 
            max_value=HEALTH_FEATURES['LDL_chole']['max'], 
            value=HEALTH_FEATURES['LDL_chole']['default'],
            help=HEALTH_FEATURES['LDL_chole']['help'],
            key=f"ldl_chole_{tab_key}"
        )
        
        triglyceride = st.number_input(
            "Triglycerides (mg/dL)", 
            min_value=HEALTH_FEATURES['triglyceride']['min'], 
            max_value=HEALTH_FEATURES['triglyceride']['max'], 
            value=HEALTH_FEATURES['triglyceride']['default'],
            help=HEALTH_FEATURES['triglyceride']['help'],
            key=f"triglyceride_{tab_key}"
        )
        
        hemoglobin = st.number_input(
            "Hemoglobin (g/dL)", 
            min_value=float(HEALTH_FEATURES['hemoglobin']['min']), 
            max_value=float(HEALTH_FEATURES['hemoglobin']['max']), 
            value=float(HEALTH_FEATURES['hemoglobin']['default']),
            step=0.1,
            help=HEALTH_FEATURES['hemoglobin']['help'],
            key=f"hemoglobin_{tab_key}"
        )
        
        serum_creatinine = st.number_input(
            "Serum Creatinine (mg/dL)", 
            min_value=float(HEALTH_FEATURES['serum_creatinine']['min']), 
            max_value=float(HEALTH_FEATURES['serum_creatinine']['max']), 
            value=float(HEALTH_FEATURES['serum_creatinine']['default']),
            step=0.1,
            help=HEALTH_FEATURES['serum_creatinine']['help'],
            key=f"serum_creatinine_{tab_key}"
        )
        
        sgot_ast = st.number_input(
            "AST (U/L)", 
            min_value=HEALTH_FEATURES['SGOT_AST']['min'], 
            max_value=HEALTH_FEATURES['SGOT_AST']['max'], 
            value=HEALTH_FEATURES['SGOT_AST']['default'],
            help=HEALTH_FEATURES['SGOT_AST']['help'],
            key=f"sgot_ast_{tab_key}"
        )
        
        sgot_alt = st.number_input(
            "ALT (U/L)", 
            min_value=HEALTH_FEATURES['SGOT_ALT']['min'], 
            max_value=HEALTH_FEATURES['SGOT_ALT']['max'], 
            value=HEALTH_FEATURES['SGOT_ALT']['default'],
            help=HEALTH_FEATURES['SGOT_ALT']['help'],
            key=f"sgot_alt_{tab_key}"
        )
        
        gamma_gtp = st.number_input(
            "Gamma-GTP (U/L)", 
            min_value=HEALTH_FEATURES['gamma_GTP']['min'], 
            max_value=HEALTH_FEATURES['gamma_GTP']['max'], 
            value=HEALTH_FEATURES['gamma_GTP']['default'],
            help=HEALTH_FEATURES['gamma_GTP']['help'],
            key=f"gamma_gtp_{tab_key}"
        )
    
    basic_values = {
        'age': age, 'weight': weight, 'height': height,
        'hear_left': hear_left, 'hear_right': hear_right,
        'sight_left': sight_left, 'sight_right': sight_right,
        'SBP': sbp, 'DBP': dbp, 'BLDS': blds, 'tot_chole': tot_chole,
        'HDL_chole': hdl_chole, 'LDL_chole': ldl_chole, 'triglyceride': triglyceride,
        'hemoglobin': hemoglobin, 'serum_creatinine': serum_creatinine,
        'SGOT_AST': sgot_ast, 'SGOT_ALT': sgot_alt, 'gamma_GTP': gamma_gtp
    }
    
    derived_features = calculate_derived_features(basic_values)
    
    st.markdown("### Calculated Metrics")
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("BMI", f"{derived_features['BMI']:.1f}")
    with col2:
        st.metric("Pulse Pressure", f"{derived_features['pulse_pressure']:.0f} mmHg")
    with col3:
        st.metric("Cholesterol Ratio", f"{derived_features['chol_ratio']:.1f}")
    
    _, col4, col5, _ = st.columns(4)
    with col4:
        st.metric("Avg Sight", f"{derived_features['avg_sight']:.1f}")
    with col5:
        st.metric("Avg Hearing", f"{derived_features['avg_hearing']:.1f}")
    

    sex_female = 1 if gender == "Female" else 0
    sex_male = 1 if gender == "Male" else 0
    

    urine_protein_map = {
        "urine_protein_1.0": 1, "urine_protein_2.0": 2, 
        "urine_protein_3.0": 3, "urine_protein_4.0": 4, "urine_protein_5.0": 5, "urine_protein_6.0": 6
    }
    urine_protein_value = urine_protein_map[urine_protein]
    

    urine_protein_features = [0] * 6
    if 0 <= urine_protein_value <= 5:
        urine_protein_features[urine_protein_value] = 1
    
    return {
        **basic_values,
        'sex_Female': sex_female,
        'sex_Male': sex_male,
        'urine_protein_features': urine_protein_features,
        'derived_features': derived_features
    }


def check_health_ranges(inputs):
    affected = {}
        
    if inputs.get('SBP', 0) > 180 or inputs.get('DBP', 0) > 120:
        affected['Heart'] = 'red'
    elif inputs.get('SBP', 0) >= 140 or inputs.get('DBP', 0) >= 90:
        affected['Heart'] = 'orange'
    elif inputs.get('SBP', 0) >= 130 or inputs.get('DBP', 0) >= 80:
        affected['Heart'] = 'yellow'

    if inputs.get('SGOT_AST', 0) > 200 or inputs.get('SGOT_ALT', 0) > 200 or inputs.get('gamma_GTP', 0) > 500:
        affected['Liver'] = 'red'
    elif inputs.get('SGOT_AST', 0) > 100 or inputs.get('SGOT_ALT', 0) > 100 or inputs.get('gamma_GTP', 0) > 150:
        affected['Liver'] = 'orange'
    elif inputs.get('SGOT_AST', 0) > 40 or inputs.get('SGOT_ALT', 0) > 56 or inputs.get('gamma_GTP', 0) > 100:
        affected['Liver'] = 'yellow'

    if inputs.get('serum_creatinine', 0) > 4:
        affected['Kidney'] = 'red'
    elif inputs.get('serum_creatinine', 0) > 1.4:
        affected['Kidney'] = 'orange'
    elif inputs.get('serum_creatinine', 0) > 1.2:
        affected['Kidney'] = 'yellow'

    if inputs.get('SBP', 0) > 180 or inputs.get('LDL_chole', 0) > 190 or inputs.get('tot_chole', 0) > 400:
        affected['Brain'] = 'red'
    elif inputs.get('SBP', 0) > 140 or inputs.get('LDL_chole', 0) > 160 or inputs.get('tot_chole', 0) > 240:
        affected['Brain'] = 'orange'
    elif inputs.get('SBP', 0) > 130 or inputs.get('LDL_chole', 0) > 130 or inputs.get('tot_chole', 0) > 200:
        affected['Brain'] = 'yellow'

    if inputs.get('hemoglobin', 0) < 8 or inputs.get('hemoglobin', 0) > 20:
        affected['Lungs'] = 'red'  
    elif inputs.get('hemoglobin', 0) < 11.6 or inputs.get('hemoglobin', 0) > 15:
        affected['Lungs'] = 'orange'
    elif inputs.get('hemoglobin', 0) < 13.2 or inputs.get('hemoglobin', 0) > 16.6:
        affected['Lungs'] = 'yellow'

    return affected






def load_body_svg():

    with open("body_map-01.svg", "r", encoding="utf-8") as f:
        return f.read()


def update_svg(svg_content, affected_organs):
    try:
        soup = BeautifulSoup(svg_content, 'xml')

        svg_root = soup.find('svg')
        if svg_root:
            svg_root['width'] = '100%'
            svg_root['height'] = '100%'
            svg_root['preserveAspectRatio'] = 'xMidYMid meet'

        color_map = {
            'red': '#ff4444',
            'orange': '#ff8800',
            'yellow': '#ffff00',
        }

        for organ_id, color in affected_organs.items():
            elements = soup.find_all(id=organ_id)
            for el in elements:
                if el.name in ['path', 'rect', 'circle', 'ellipse']:
                    el['fill'] = color_map.get(color, color)
                elif el.name == 'g':
                    for child in el.find_all(['path', 'rect', 'circle', 'ellipse'], recursive=True):
                        child['fill'] = color_map.get(color, color)

        return str(soup)

    except Exception as e:
        st.error(f"SVG update error: {str(e)}")
        return svg_content




def create_body_diagram(health_values):
    st.markdown("### Interactive Body Health Map")
    
    affected_organs = check_health_ranges(health_values)
    svg_content = load_body_svg()
    updated_svg = update_svg(svg_content, affected_organs)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: white; border-radius: 10px; border: 2px solid #BAD7F2;">
            <div style="width: 350px; height: 500px; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                {updated_svg}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Color Legend")
        st.markdown("""
        - **🔴 Red**: High risk - Immediate attention needed
        - **🟠 Orange**: Moderate risk - Monitor closely  
        - **🟡 Yellow**: Mild risk - Keep an eye on
        """)
        
        if affected_organs:
            st.markdown("### ⚠️ Affected Areas")
            for organ, risk in affected_organs.items():
                risk_emoji = {"red": "🔴", "orange": "🟠", "yellow": "🟡"}.get(risk, "⚪")
                st.markdown(f"{risk_emoji} **{organ.title()}**: {risk.upper()} risk")
        else:
            st.markdown("### All Systems Normal")
            st.markdown("Great job! All your health indicators are within normal ranges.")

def main():

    st.markdown("""
    <div class="main-header">
        <h1><i class="fas fa-heartbeat"></i> Smoking and Drinking Predictor</h1>
        <p>Predict smoking and drinking for early intervention</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## <i class='fas fa-cog'></i> Prediction Settings", unsafe_allow_html=True)

        

        prediction_type = st.selectbox(
            "Choose Prediction Target:", 
            ["Smoking", "Drinking"],
            help="Select whether to predict smoking or drinking"
        )
        
        if prediction_type == "Smoking":
            models = SMOKING_MODELS
            label_map = {3: "Currently Smoking", 2: "Quit Smoking", 1: "Never Smoked"}
        else:
            models = DRINKING_MODELS
            label_map = {0: "No", 1: "Yes"}
        
        selected_model_name = st.selectbox(
            "Choose Model:", 
            list(models.keys()),
            help="Select the machine learning model to use for prediction"
        )
        
        model_path = models[selected_model_name]
        model = load_model(model_path)
        if model is None:
            st.error(f" Failed to load model: {selected_model_name}")
            return
    

    tab1, tab2 = st.tabs([" Prediction", " Health Dashboard"])
    
    if 'prediction_values' not in st.session_state:
        st.session_state.prediction_values = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'prediction_confidence' not in st.session_state:
        st.session_state.prediction_confidence = None
    
    with tab1:
        st.markdown("##  Prediction")

        values = create_input_form("prediction")
        
        if st.button("Predict S/D", type="primary"):
            with st.spinner("Analyzing health data..."):
                time.sleep(1)
                
                X = preprocess_inputs(values, values['derived_features'])
                

                prediction, confidence = get_prediction_with_confidence(model, X)
                
                if prediction is not None:

                    st.session_state.prediction_values = values
                    st.session_state.prediction_result = prediction
                    st.session_state.prediction_confidence = confidence
                    
                    st.markdown("""
                    <div class="prediction-card">
                        <h2> Prediction Result</h2>
                        <h1>{}</h1>
                        <p>Confidence: {:.1f}%</p>
                    </div>
                    """.format(label_map.get(prediction, str(prediction)), confidence), 
                    unsafe_allow_html=True)
                    

                    st.markdown("## Health Recommendations")
                    if prediction_type == "Smoking":
                        if prediction == 3:
                            st.markdown("""
                            <div class="health-recommendation">
                                <h3> Smoking Cessation Support</h3>
                                <ul>
                                    <li>Consider nicotine replacement therapy</li>
                                    <li>Join smoking cessation programs</li>
                                    <li>Set a quit date and stick to it</li>
                                    <li>Seek support from family and friends</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        elif prediction == 1:
                            st.markdown("""
                            <div class="health-recommendation">
                                <h3> Great Job!</h3>
                                <p>Keep up the healthy lifestyle and avoid smoking.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="health-recommendation">
                                <h3> Congratulations!</h3>
                                <p>Stay strong and maintain your smoke-free lifestyle.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        if prediction == 1:
                            st.markdown("""
                            <div class="health-recommendation">
                                <h3> Alcohol Moderation</h3>
                                <ul>
                                    <li>Limit alcohol intake to moderate levels</li>
                                    <li>Consider alcohol-free days</li>
                                    <li>Monitor your drinking patterns</li>
                                    <li>Seek help if needed</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:  
                            st.markdown("""
                            <div class="health-recommendation">
                                <h3> Excellent Choice!</h3>
                                <p>Maintaining an alcohol-free lifestyle is beneficial for your health.</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## Health Dashboard")
        
        if st.session_state.prediction_values is None:
            st.info("Please make a prediction first to see the health dashboard.")
        else:

            create_body_diagram(st.session_state.prediction_values)
            

            st.markdown("###  Latest Prediction")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction", label_map.get(st.session_state.prediction_result, str(st.session_state.prediction_result)))
            
            with col2:
                st.metric("Confidence", f"{st.session_state.prediction_confidence:.1f}%")
            
            # Health Analysis section
            st.markdown("### Key Health Indicators")
            
            key_indicators = {
                'SBP': st.session_state.prediction_values['SBP'],
                'DBP': st.session_state.prediction_values['DBP'],
                'LDL': st.session_state.prediction_values['LDL_chole'],
                'HDL': st.session_state.prediction_values['HDL_chole'],
                'AST': st.session_state.prediction_values['SGOT_AST'],
                'ALT': st.session_state.prediction_values['SGOT_ALT'],
                'Creatinine': st.session_state.prediction_values['serum_creatinine'] * 10, 
                'Glucose': st.session_state.prediction_values['BLDS']
            }
            
            def get_color(metric, value):
                if metric == 'SBP':
                    if value >= 180: return '#ff6f61'
                    elif value >= 140: return '#ffb84c'
                    elif value >= 130: return '#ffe29a'
                    else: return '#b6e388'
                elif metric == 'DBP':
                    if value >= 120: return '#ff6f61'
                    elif value >= 90: return '#ffb84c'
                    elif value >= 80: return '#ffe29a'
                    else: return '#b6e388'
                elif metric == 'LDL':
                    if value >= 190: return '#ff6f61'
                    elif value >= 160: return '#ffb84c'
                    elif value >= 130: return '#ffe29a'
                    else: return '#b6e388'
                elif metric == 'HDL':
                    if value < 40: return '#ff6f61'
                    elif value < 50: return '#ffb84c'
                    elif value < 60: return '#ffe29a'
                    else: return '#b6e388'
                elif metric == 'AST':
                    if value > 200: return '#ff6f61'
                    elif value > 100: return '#ffb84c'
                    elif value > 40: return '#ffe29a'
                    else: return '#b6e388'
                elif metric == 'ALT':
                    if value > 200: return '#ff6f61'
                    elif value > 100: return '#ffb84c'
                    elif value > 56: return '#ffe29a'
                    else: return '#b6e388'
                elif metric == 'Creatinine':
                    if value > 4: return '#ff6f61'
                    elif value > 1.4: return '#ffb84c'
                    elif value > 1.2: return '#ffe29a'
                    else: return '#b6e388'
                elif metric == 'Glucose':
                    if value > 300 or value < 70: return '#ff6f61'
                    elif value > 126: return '#ffb84c'
                    elif value > 100: return '#ffe29a'
                    else: return '#b6e388'
                else:
                    return '#b6e388'
            
            colors = [get_color(metric, value) for metric, value in key_indicators.items()]
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=list(key_indicators.keys()),
                    y=list(key_indicators.values()),
                    marker_color=colors
                )
            ])
            fig_bar.update_layout(
                xaxis_title="Metrics",
                yaxis_title="Values",
                height=300
            )
            st.plotly_chart(fig_bar, use_container_width=True)
if __name__ == "__main__":
    main()
