import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import hashlib
import sqlite3
from pathlib import Path
import pygad

# Database functions

def get_polyculture_suggestions_ga(crop, soil_data, recommendation_df, yield_df, price_df, state, max_crops=5):
    """
    Suggests a polyculture crop set using a Genetic Algorithm (GA).
    Returns up to `max_crops` crops (excluding the main crop) that maximize total yield and price.
    Each suggestion is a dict: {'Crop', 'Average Yield', 'Estimated Price'}.
    """
    crop = crop.lower()
    all_crops = recommendation_df['label'].str.lower().unique().tolist()
    # Exclude the main crop
    candidate_crops = [c for c in all_crops if c != crop]
    num_candidates = len(candidate_crops)
    if num_candidates < 2:
        return []

    # Helper functions
    def get_yield(crop_name):
        data = yield_df[(yield_df['Crop'].str.lower() == crop_name) & (yield_df['State'].str.lower() == state.lower())]
        if not data.empty:
            return data['Yield'].mean()
        return 0
    def get_price(crop_name):
        data = price_df[price_df['Commodity'].str.lower() == crop_name]
        if not data.empty:
            return data['Modal Price'].mean()
        return 0

    # Fitness function for PyGAD 2.20.0
    def fitness_func(ga_instance, solution, solution_idx):
        selected_indices = np.where(solution == 1)[0]
        if len(selected_indices) < 2:
            return 0
        total_yield = sum(get_yield(candidate_crops[i]) for i in selected_indices)
        total_profit = sum(get_yield(candidate_crops[i]) * get_price(candidate_crops[i]) for i in selected_indices)
        # All crops compatible for now; can add compatibility score later
        return total_profit + 1000 * total_yield

    gene_space = [0, 1]
    num_genes = num_candidates
    ga_instance = pygad.GA(
        num_generations=30,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type="rank",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
        random_seed=42
    )
    ga_instance.run()
    solution, solution_fitness, _ = ga_instance.best_solution()
    selected_indices = np.where(solution == 1)[0]
    # Limit to max_crops best by yield*price
    crop_scores = [
        (candidate_crops[i], get_yield(candidate_crops[i]), get_price(candidate_crops[i]))
        for i in selected_indices
    ]
    crop_scores = sorted(crop_scores, key=lambda x: x[1]*x[2], reverse=True)[:max_crops]
    suggestions = [
        {
            'Crop': name.capitalize(),
            'Average Yield': round(yld, 2),
            'Estimated Price': round(price, 2) if price else "N/A"
        }
        for name, yld, price in crop_scores if yld > 0 and price > 0
    ]
    return suggestions

def create_users_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, email TEXT)''')
    conn.commit()
    conn.close()

def add_user(username, password, email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users VALUES (?, ?, ?)", 
                 (username, hashlib.sha256(password.encode()).hexdigest(), email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
             (username, hashlib.sha256(password.encode()).hexdigest()))
    result = c.fetchone()
    conn.close()
    return result is not None

# Initialize database
create_users_db()

# Set page configuration
st.set_page_config(
    page_title="FarmForecast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    :root {
        --primary: #2e7d32;
        --secondary: #388e3c;
        --light-green: #e8f5e9;
        --dark-green: #1b5e20;
        --blue: #1976d2;
        --light-blue: #e3f2fd;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: var(--primary);
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .tagline {
        font-size: 1.2rem;
        color: var(--secondary);
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--secondary);
        margin-bottom: 1rem;
    }
    
    .result-box {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #c5e1a5;
        margin-bottom: 1.5rem;
    }
    
    .stat-card {
        background-color: var(--light-green);
        padding: 12px 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid var(--secondary);
    }
    
    .polyculture-card {
        background-color: var(--light-blue);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid var(--blue);
    }
    
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    
    .form-input {
        margin-bottom: 1.2rem;
    }
    
    .form-input label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 500;
        color: var(--dark-green);
    }
    
    .form-input input {
        width: 100%;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1rem;
    }
    
    .btn {
        background-color: var(--primary);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
        transition: background-color 0.3s;
        width: 100%;
    }
    
    .btn:hover {
        background-color: var(--dark-green);
    }
    
    .secondary-btn {
        background-color: white;
        color: var(--primary);
        border: 1px solid var(--primary);
    }
    
    .secondary-btn:hover {
        background-color: var(--light-green);
    }
    
    .nav-container {
        display: flex;
        justify-content: flex-end;
        padding: 1rem;
    }
    
    .nav-btn {
        margin-left: 0.5rem;
    }
    
    .footer {
        text-align: center;
        color: var(--secondary);
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    
    .error-message {
        color: #d32f2f;
        margin-bottom: 1rem;
    }
    
    .success-message {
        color: var(--primary);
        margin-bottom: 1rem;
    }
    
    .logo-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .logo {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading and model training functions
@st.cache_data
def load_data():
    recommendation_df = pd.read_csv('datasets/Crop_recommendation.csv')
    yield_df = pd.read_csv('datasets/crop_yield.csv')
    price_df = pd.read_csv('datasets/Price_Agriculture_commodities_Week.csv')
    return recommendation_df, yield_df, price_df

@st.cache_resource
def train_model(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y_encoded)
    explainer = shap.TreeExplainer(rf_model)
    return rf_model, label_encoder, explainer

# Application functions
def is_crop_suitable(yield_df, state, crop):
    df = yield_df.copy()
    df['Crop'] = df['Crop'].str.lower()
    df['State'] = df['State'].str.lower()
    state_crop_data = df[(df['State'] == state.lower()) & (df['Crop'] == crop.lower())]
    return not state_crop_data.empty, state_crop_data

def get_crop_stats(data):
    return {
        'Estimated Yield': data['Yield'].mean(),
        'Estimated Fertilizer': data['Fertilizer'].mean(),
        'Estimated Pesticide': data['Pesticide'].mean()
    }

def get_average_price(price_df, crop):
    crop_data = price_df[price_df['Commodity'].str.lower() == crop.lower()]
    if not crop_data.empty:
        return crop_data['Modal Price'].mean()
    return None

def recommend_crop(rf_model, label_encoder, soil_data, X):
    soil_df = pd.DataFrame([soil_data], columns=X.columns)
    prediction = rf_model.predict(soil_df)[0]
    crop = label_encoder.inverse_transform([prediction])[0]
    return crop

def explain_recommendation(explainer, soil_data, X):
    soil_df = pd.DataFrame([soil_data], columns=X.columns)
    shap_values = explainer.shap_values(soil_df)
    prediction_idx = np.argmax([abs(sv).sum() for sv in shap_values])
    feature_names = X.columns.tolist()
    explanations = list(zip(feature_names, shap_values[prediction_idx][0]))
    explanations.sort(key=lambda x: abs(x[1]), reverse=True)
    return explanations[:3]

def get_polyculture_suggestions_v2(crop, soil_data, recommendation_df, yield_df, price_df, state):
    crop = crop.lower()
    df = recommendation_df.copy()
    df['label'] = df['label'].str.lower()
    input_vec = np.array([soil_data[col] for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])
    df['distance'] = df.drop('label', axis=1).apply(lambda row: np.linalg.norm(input_vec - row.values), axis=1)
    similar_crops = df[df['label'] != crop].sort_values('distance').drop_duplicates('label')
    suggestions = []
    for rc in similar_crops['label'].head(5):
        data = yield_df[yield_df['Crop'].str.lower() == rc.lower()]
        price = get_average_price(price_df, rc)
        if not data.empty:
            suggestions.append({
                'Crop': rc.capitalize(),
                'Average Yield': round(data['Yield'].mean(), 2),
                'Estimated Price': round(price, 2) if price else "N/A"
            })
    return suggestions

def analyze_crop(rf_model, label_encoder, explainer, yield_df, price_df, recommendation_df, state, input_crop, soil_data, X):
    suitable, data = is_crop_suitable(yield_df, state, input_crop)
    if suitable:
        stats = get_crop_stats(data)
        price = get_average_price(price_df, input_crop)
        polyculture_suggestions = get_polyculture_suggestions_v2(input_crop, soil_data, recommendation_df, yield_df, price_df, state)        
        return {
            'Status': 'Suitable',
            'Crop': input_crop,
            **stats,
            'Average Market Price': price,
            'Polyculture Suggestions': polyculture_suggestions
        }
    else:
        recommended_crop = recommend_crop(rf_model, label_encoder, soil_data, X)
        explanations = explain_recommendation(explainer, soil_data, X)
        suitable, new_data = is_crop_suitable(yield_df, state, recommended_crop)
        stats = get_crop_stats(new_data) if suitable else {}
        price = get_average_price(price_df, recommended_crop)
        polyculture_suggestions = get_polyculture_suggestions_v2(input_crop, soil_data, recommendation_df, yield_df, price_df, state)        
        return {
            'Status': 'Not Suitable',
            'Recommended Crop': recommended_crop,
            'Why Not Suitable': explanations,
            **stats,
            'Average Market Price': price,
            'Polyculture Suggestions': polyculture_suggestions
        }

def plot_feature_importance(rf_model, X):
    feature_imp = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value','Feature'])
    fig = px.bar(
        feature_imp.sort_values(by="Value", ascending=False),
        x="Value", 
        y="Feature",
        orientation='h',
        title='Feature Importance for Crop Recommendation',
        color="Value",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(plot_bgcolor='white', xaxis_title="Importance", yaxis_title="Features", height=500)
    return fig

# Authentication pages
def login_page():
    st.markdown("""
        <div class="login-container">
            <div class="logo-container">
                <div class="logo">🌾</div>
                <h1>FarmForecast</h1>
            </div>
            <h2 style="text-align: center; color: var(--primary); margin-bottom: 1.5rem;">Login</h2>
    """, unsafe_allow_html=True)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username and password:
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.show_login = False
                st.rerun()
            else:
                st.error("Invalid username or password")
        else:
            st.error("Please enter both username and password")
    
    if st.button("Go to Registration", key="go_to_register"):
        st.session_state.show_login = False
        st.session_state.show_register = True
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def register_page():
    st.markdown("""
        <div class="login-container">
            <div class="logo-container">
                <div class="logo">🌾</div>
                <h1>FarmForecast</h1>
            </div>
            <h2 style="text-align: center; color: var(--primary); margin-bottom: 1.5rem;">Register</h2>
    """, unsafe_allow_html=True)
    
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Register"):
        if not (username and email and password and confirm_password):
            st.error("Please fill in all fields")
        elif password != confirm_password:
            st.error("Passwords do not match")
        else:
            if add_user(username, password, email):
                st.success("Registration successful! Please login.")
                st.session_state.show_register = False
                st.session_state.show_login = True
                st.rerun()
            else:
                st.error("Username already exists")
    
    if st.button("Back to Login", key="back_to_login"):
        st.session_state.show_register = False
        st.session_state.show_login = True
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

def explain_feature_impact_readable(feature, impact, crop):
    direction = "high" if impact > 0 else "low"
    return f"⚠️ {feature.capitalize()} is too {direction} for growing {crop.capitalize()}."



# Main application
def main_app():
    try:
        recommendation_df, yield_df, price_df = load_data()
        X = recommendation_df.drop('label', axis=1)
        y = recommendation_df['label']
        rf_model, label_encoder, explainer = train_model(X, y)

        # Navigation
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.show_login = True
            st.rerun()
        
        st.markdown("<h1 class='main-header'>🌾 FarmForecast</h1>", unsafe_allow_html=True)
        st.markdown("<p class='tagline'>Predict, plant, prosper.</p>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<h2 class='sub-header'>Input Parameters</h2>", unsafe_allow_html=True)
            with st.form("crop_form"):
                state = st.text_input("State/Province", "Karnataka")
                crop = st.text_input("Crop you want to grow", "rice")
                st.markdown("### Soil and Climate Conditions")
                col_left, col_right = st.columns(2)
                with col_left:
                    n = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0, 1.0)
                    p = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0, 1.0)
                    k = st.number_input("Potassium (K)", 0.0, 200.0, 43.0, 1.0)
                    temp = st.number_input("Temperature (°C)", 0.0, 50.0, 20.0, 0.1)
                with col_right:
                    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 82.0, 1.0)
                    ph = st.number_input("pH value", 0.0, 14.0, 6.5, 0.1)
                    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 202.9, 0.1)
                submit_button = st.form_submit_button("Analyze Crop Suitability")

        with col2:
            fig = plot_feature_importance(rf_model, X)
            st.plotly_chart(fig, use_container_width=True)
            unique_crops = sorted(recommendation_df['label'].unique())
            with st.expander("Available Crops in Database"):
                st.write(", ".join(unique_crops))

        if submit_button:
            soil = {'N': n, 'P': p, 'K': k, 'temperature': temp, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
            with st.spinner('Analyzing crop suitability...'):
                result = analyze_crop(rf_model, label_encoder, explainer, yield_df, price_df, recommendation_df, state, crop, soil, X)

            st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
            result_col1, result_col2 = st.columns([3, 2])

            with result_col1:
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                if result['Status'] == 'Suitable':
                    st.success(f"✅ {crop.capitalize()} is suitable for your conditions!")
                    for key, value in result.items():
                        if key not in ['Status', 'Crop', 'Polyculture Suggestions']:
                            st.markdown(f"<div class='stat-card'><b>{key}:</b> {value}</div>", unsafe_allow_html=True)
                else:
                    st.error(f"❌ {crop.capitalize()} is not suitable for your conditions.")
                    st.info(f"🌱 Recommended crop: **{result['Recommended Crop'].capitalize()}**")
                    st.markdown("### Why Not Suitable:")
                    for feature, impact in result['Why Not Suitable']:
                         explanation = explain_feature_impact_readable(feature, impact, crop)
                         st.markdown(f"- {explanation}")

                    for key, value in result.items():
                        if key not in ['Status', 'Recommended Crop', 'Why Not Suitable', 'Polyculture Suggestions']:
                            st.markdown(f"<div class='stat-card'><b>{key}:</b> {value}</div>", unsafe_allow_html=True)
                if result['Polyculture Suggestions']:
                    st.markdown("### 🌿 Polyculture Suggestions")
                    for suggestion in result['Polyculture Suggestions']:
                        st.markdown(
                            f"<div class='polyculture-card'><b>{suggestion['Crop']}</b><br>"
                            f"Yield: {suggestion['Average Yield']} tons/ha<br>"
                            f"Market Price: ₹{suggestion['Estimated Price']}</div>",
                            unsafe_allow_html=True)
                else:
                    st.info("No suitable polyculture crops found based on current soil and yield data.")

            with result_col2:
                radar_df = pd.DataFrame({
                    'Parameter': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
                    'Value': [n, p, k, temp, humidity, ph, rainfall]
                })
                fig = px.line_polar(radar_df, r='Value', theta='Parameter', line_close=True, title="Soil & Climate Parameter Radar")
                fig.update_traces(fill='toself')
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
            <div class="footer">
                <h3>FarmForecast</h3>
                <p><em>Predict, plant, prosper.</em></p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please make sure all the required CSV files are uploaded and accessible.")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = True
if 'show_register' not in st.session_state:
    st.session_state.show_register = False

# App routing
if st.session_state.logged_in:
    main_app()
else:
    if st.session_state.show_register:
        register_page()
    else:
        login_page()