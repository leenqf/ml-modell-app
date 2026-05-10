
import streamlit as st
import numpy as np
import joblib


st.set_page_config(page_title="Road Severity Predictor", layout="centered")

st.markdown("""
    <style>
        /* Main background */
        .stApp, .stApp > div, section[data-testid="stAppViewContainer"] {
            background-color: #FFCBA4 !important;
        }

        /* Sidebar and other panels */
        section[data-testid="stSidebar"] {
            background-color: #FFB87A !important;
        }

        /* Top header bar */
        header[data-testid="stHeader"] {
            background-color: #FFCBA4 !important;
        }

        /* ALL text white */
        .stApp * {
            color: white !important;
        }

        /* Title */
        h1, h2, h3 {
            color: white !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        }

        /* Labels above inputs */
        label, .stSelectbox label, .stNumberInput label {
            color: white !important;
            font-weight: 600 !important;
        }

        /* Selectbox box itself */
        .stSelectbox > div > div {
            background-color: #FFE0C2 !important;
            border: 1px solid #FFB87A !important;
            color: #333 !important;
        }

        /* Dropdown text */
        .stSelectbox > div > div > div {
            color: #333333 !important;
        }

        /* Number input box */
        .stNumberInput input {
            background-color: #FFE0C2 !important;
            border: 1px solid #FFB87A !important;
            color: #333333 !important;
        }

        /* Predict button */
        div.stButton > button {
            background-color: #FFE57F !important;
            color: #333333 !important;
            font-weight: bold !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 30px !important;
            font-size: 16px !important;
            width: 100% !important;
            margin-top: 10px !important;
        }

        div.stButton > button:hover {
            background-color: #FFD740 !important;
            color: #111111 !important;
            transform: scale(1.02);
        }

        /* Success/info boxes */
        .stSuccess {
            background-color: #FFB87A !important;
            color: white !important;
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Road Accident Severity Predictor")

# Load model and scaler
final_rf = joblib.load('final_rf-2.pkl')
mm_scaler = joblib.load('mm_scaler.pkl')

severity_map = {0: "🟡 Slight Injury", 1: "🟠 Serious Injury", 2: "🔴 Fatal Injury"}

# --- Row 1 ---
col1, col2 = st.columns(2)
with col1:
    weather = st.selectbox("Weather Condition", ["Normal", "Rain", "Cloudy", "Windy", "Other", "Unknown"])
with col2:
    light = st.selectbox("Light Condition", ["Daylight", "Darkness (lit)", "Darkness (no light)"])

# --- Row 2 ---
col3, col4 = st.columns(2)
with col3:
    road = st.selectbox("Road Surface", ["Dry", "Wet", "Snow"])
with col4:
    junction = st.selectbox("Junction Type", ["Y Junction", "No Junction", "Crossing", "Other", "Unknown"])

# --- Row 3 ---
col5, col6 = st.columns(2)
with col5:
    lane = st.selectbox("Lane Marking", ["Broken", "Solid", "Undivided", "One Way", "Divided"])
with col6:
    age_band = st.selectbox("Driver Age Band", [1, 2, 3, 4, 5])

# --- Row 4 ---
col7, col8 = st.columns(2)
with col7:
    vehicle = st.selectbox("Type of Vehicle", ["Car", "Pickup", "Truck", "Bus", "Two Wheeler", "Other"])
with col8:
    service_year = st.number_input("Vehicle Service Year", min_value=0, max_value=20, value=3)

# --- Row 5 ---
col9, col10 = st.columns(2)
with col9:
    num_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=2)
with col10:
    cause = st.selectbox("Cause of Accident", ["Careless", "Lane Change", "Speeding", "Priority Violation", "Maneuver", "Other"])

st.write("")  # spacing

if st.button("🔍 Predict Severity"):

    row = np.zeros(37)

    weather_map = {"Normal":0, "Rain":1, "Cloudy":2, "Windy":3, "Other":4, "Unknown":5}
    row[weather_map[weather]] = 1

    light_map = {"Daylight":6, "Darkness (lit)":7, "Darkness (no light)":8}
    row[light_map[light]] = 1

    road_map = {"Dry":9, "Wet":10, "Snow":11}
    row[road_map[road]] = 1

    junction_map = {"Y Junction":12, "No Junction":13, "Crossing":14, "Other":15, "Unknown":16}
    row[junction_map[junction]] = 1

    lane_map = {"Broken":17, "Solid":18, "Undivided":19, "One Way":20, "Divided":21}
    row[lane_map[lane]] = 1

    row[22] = age_band

    vehicle_map = {"Car":23, "Pickup":24, "Truck":25, "Bus":26, "Two Wheeler":27, "Other":28}
    row[vehicle_map[vehicle]] = 1

    row[29] = service_year
    row[30] = num_vehicles

    cause_map = {"Careless":31, "Lane Change":32, "Speeding":33, "Priority Violation":34, "Maneuver":35, "Other":36}
    row[cause_map[cause]] = 1

    row_scaled = mm_scaler.transform([row])
    prediction = final_rf.predict(row_scaled)[0]

    st.success(f"Predicted Severity: {severity_map[prediction]}")
