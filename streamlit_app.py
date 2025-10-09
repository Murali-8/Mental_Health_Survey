import streamlit as st
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ====================================================
#  Function: preprocess_user_input()
# ====================================================
def preprocess_user_input(user_input_df):
    """
    Takes a single-row DataFrame of raw user inputs and
    returns a processed tensor ready for PyTorch model prediction.
    """

    df = user_input_df.copy()
    df.columns = df.columns.str.strip()  # Clean column names

    # ------------------------------------------------
    # Step 1: Derived Feature - Age_Group
    # ------------------------------------------------
    def get_age_group(age):
        if age < 26:
            return "18-25"
        elif age < 36:
            return "26-35"
        elif age < 46:
            return "36-45"
        else:
            return "46-60"
    df["Age_Group"] = df["Age"].apply(get_age_group)

    # ------------------------------------------------
    # Step 2: Derived Feature - Age × Profession Combo
    # ------------------------------------------------
    df["age_profession_combo"] = (
        df["Age_Group"].astype(str) + "_" + df["Working Professional or Student"].astype(str)
    )

    # ------------------------------------------------
    # Step 3: Derived Feature - Lifestyle Risk Score
    # ------------------------------------------------
    df["lifestyle_risk_score"] = (
        df["Work Pressure"].astype(float)
        + df["Financial Stress"].astype(float)
        + df["Academic Pressure"].astype(float)
    ) / 3.0

    # ------------------------------------------------
    # Step 4: Derived Feature - Mental Health Risk Score
    # ------------------------------------------------
    df["mental_health_risk_score"] = (
        df["Job Satisfaction"].astype(float)
        + df["Study Satisfaction"].astype(float)
        + (df["Have you ever had suicidal thoughts ?"].map({"yes": 10, "no": 0}).fillna(0))
    ) / 3.0

    # ------------------------------------------------
    # Step 5: City → Region Mapping
    # ------------------------------------------------
    north = ['delhi', 'chandigarh', 'lucknow', 'kanpur', 'varanasi', 'amritsar', 'ludhiana']
    south = ['chennai', 'bangalore', 'hyderabad', 'kochi', 'coimbatore', 'visakhapatnam', 'madurai']
    east = ['kolkata', 'bhubaneswar', 'patna', 'guwahati', 'ranchi']
    west = ['mumbai', 'pune', 'ahmedabad', 'surat', 'vadodara', 'nashik', 'nagpur']
    central = ['bhopal', 'indore', 'gwalior']

    def map_city_to_region(city):
        city = str(city).strip().lower()
        if city in north:
            return "North India"
        elif city in south:
            return "South India"
        elif city in east:
            return "East India"
        elif city in west:
            return "West India"
        elif city in central:
            return "Central India"
        else:
            return "Other Region"

    df["City"] = df["City"].apply(map_city_to_region)

    # ------------------------------------------------
    # Step 6: Profession Simplification
    # ------------------------------------------------
    top3_prof = ["teacher", "student", "content writer"]
    df["Profession"] = df["Profession"].apply(lambda x: x if x in top3_prof else "Other")

    # ------------------------------------------------
    # Step 7: Degree Simplification
    # ------------------------------------------------
    def simplify_degree(x):
        x = str(x).lower()
        if "phd" in x or "doctorate" in x:
            return "PhD"
        elif any(k in x for k in ["m.", "master", "mba", "pg", "msc", "mtech", "ma", "mca"]):
            return "PG"
        elif any(k in x for k in ["b.", "bachelor", "ug", "btech", "ba", "bsc", "bcom"]):
            return "UG"
        elif "class 12" in x or "12" in x:
            return "UG"
        else:
            return "Other"
    df["Degree"] = df["Degree"].apply(simplify_degree)

    # ------------------------------------------------
    # Step 8: Simplify Dietary Habits & Sleep Duration
    # ------------------------------------------------
    top3_diet = ["moderate", "unhealthy", "healthy"]
    df["Dietary Habits"] = df["Dietary Habits"].apply(lambda x: x if x in top3_diet else "Other")

    top3_sleep = ["less than 5 hours", "7-8 hours", "more than 8 hours"]
    df["Sleep Duration"] = df["Sleep Duration"].apply(lambda x: x if x in top3_sleep else "Other")

    # ------------------------------------------------
    # Step 9: Encoding Categorical Columns
    # ------------------------------------------------
    label_maps = {
        "Working Professional or Student": {"student": 0, "working professional": 1},
        "Depression": {"no": 0, "yes": 1},
        "City": {
            "North India": 0, "South India": 1, "East India": 2,
            "West India": 3, "Central India": 4, "Other Region": 5
        },
        "Profession": {"teacher": 0, "student": 1, "content writer": 2, "Other": 3},
        "Degree": {"UG": 0, "PG": 1, "PhD": 2, "Other": 3},
        "Dietary Habits": {"moderate": 0, "unhealthy": 1, "healthy": 2, "Other": 3},
        "Sleep Duration": {"less than 5 hours": 0, "7-8 hours": 1, "more than 8 hours": 2, "Other": 3},
        "age_profession_combo": {
            "18-25_student": 0, "26-35_student": 1, "18-25_working professional": 2,
            "26-35_working professional": 3, "36-45_working professional": 4,
            "46-60_working professional": 5, "36-45_student": 6, "46-60_student": 7
        }
    }

    for col, mapping in label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # ------------------------------------------------
    # Step 10: One-Hot Encode Age_Group
    # ------------------------------------------------
    age_dummies = pd.get_dummies(df["Age_Group"], prefix="Age_Group", drop_first=True)
    df = pd.concat([df, age_dummies], axis=1)
    df.drop(columns=["Age_Group"], inplace=True)

    # ------------------------------------------------
    # Step 11: Drop Target Column if Exists
    # ------------------------------------------------
    df = df.drop(columns=["Depression"], errors="ignore")

    # ------------------------------------------------
    # Step 12: Load Saved Feature Columns & Reindex
    # ------------------------------------------------
    feature_columns = joblib.load("feature_columns.pkl")
    df = df.reindex(columns=feature_columns, fill_value=0)

    # ------------------------------------------------
    #  Step 13: Convert All Non-Numeric to Numeric Safely
    # ------------------------------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ------------------------------------------------
    # Step 14: Convert to Tensor
    # ------------------------------------------------
    X_tensor = torch.tensor(df.values, dtype=torch.float32)
    return X_tensor


# ====================================================
#  PyTorch Model Definition
# ====================================================
class DepressionNet(nn.Module):
    def __init__(self, input_dim):
        super(DepressionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


# ====================================================
#  Streamlit App UI
# ====================================================
st.title("🧠 Mental Health Depression Prediction App")
st.write("Enter your details below to check depression likelihood.")

# Collect user inputs
user_input = {
    "Age": st.number_input("Age", min_value=10, max_value=80, value=25),
    "City": st.text_input("City (e.g., Mumbai, Delhi)"),
    "Working Professional or Student": st.selectbox("Working Professional or Student", ["student", "working professional"]),
    "Profession": st.text_input("Profession (e.g., teacher, content writer, student)"),
    "Academic Pressure": st.number_input("Academic Pressure (1–10)", 0, 10, 5),
    "Work Pressure": st.number_input("Work Pressure (1–10)", 0, 10, 5),
    "CGPA": st.number_input("CGPA (0–10)", 0.0, 10.0, 7.5),
    "Study Satisfaction": st.number_input("Study Satisfaction (1–10)", 0, 10, 5),
    "Job Satisfaction": st.number_input("Job Satisfaction (1–10)", 0, 10, 5),
    "Sleep Duration": st.selectbox("Sleep Duration", ["less than 5 hours", "5-6 hours", "7-8 hours", "more than 8 hours"]),
    "Dietary Habits": st.selectbox("Dietary Habits", ["healthy", "moderate", "unhealthy"]),
    "Degree": st.text_input("Degree (e.g., B.Tech, MBA, PhD)"),
    "Have you ever had suicidal thoughts ?": st.selectbox("Have you ever had suicidal thoughts?", ["yes", "no"]),
    "Work/Study Hours": st.number_input("Work/Study Hours per day", 0, 24, 8),
    "Financial Stress": st.number_input("Financial Stress (1–10)", 0, 10, 5),
    "Family History of Mental Illness": st.selectbox("Family History of Mental Illness", ["yes", "no"]),
    "Depression": st.selectbox("Do you feel depressed currently?", ["yes", "no"])
}

# Convert to DataFrame
user_input_df = pd.DataFrame([user_input])

# ====================================================
#  Prediction Section
# ====================================================
if st.button("Predict Depression Status"):
    try:
        # Preprocess
        processed_tensor = preprocess_user_input(user_input_df)

        # Load model
        input_dim = processed_tensor.shape[1]
        model = DepressionNet(input_dim)
        model.load_state_dict(torch.load("depression_model.pth", map_location=torch.device("cpu")))
        model.eval()

        # Predict
        with torch.no_grad():
            pred = model(processed_tensor)
            pred_label = int(pred.item() >= 0.5)

        # Output
        if pred_label == 1:
            st.error(" The person is likely experiencing **Depression**.")
        else:
            st.success(" The person is **Not Depressed**.")

    except Exception as e:
        st.error(f" Error during prediction: {e}")
