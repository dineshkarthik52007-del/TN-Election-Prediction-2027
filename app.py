import streamlit as st
import pandas as pd
import requests
from streamlit_js_eval import streamlit_js_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SUPABASE_URL = "https://xkiaxjhifmnsxthxkwlk.supabase.co"
SUPABASE_KEY = "sb_publishable__WT7GeEGizjxAKilJEQjhg_LbjejXtq"

st.set_page_config(
    page_title="Chennai Election Analysis 2027",
    layout="wide"
)

st.title("Chennai Local Body Election 2027")

st.caption(
    "A ward-wise dashboard for historical election analysis "
    "and public opinion polling across Chennai's 200 wards."
)

st.subheader("Ward-wise Election Analysis Dashboard")

data = pd.read_csv("chennai_complete_2022.csv")

parties = ["TVK", "DMK", "AIADMK", "NTK"]

party_columns = {
    "DMK": "தி.மு.க",
    "AIADMK": "அ.இ.அ.தி.மு.க",
    "NTK": "நா.த.க."
}

all_wards = sorted(data["ward_no"].dropna().unique())

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": "Bearer " + SUPABASE_KEY,
    "Content-Type": "application/json"
}

# Random Forest historical model

ml_features = ["electors", "candidates"]

ml_data = data[ml_features + ["winner_party_ta"]].copy()

for column in ml_features:
    ml_data[column] = pd.to_numeric(
        ml_data[column],
        errors="coerce"
    )

ml_data = ml_data.dropna()

X = ml_data[ml_features]
y = ml_data["winner_party_ta"]

if len(ml_data) < 2 or y.nunique() < 2:
    st.error("Not enough valid historical data to train the Random Forest model.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
majority_baseline = y_test.value_counts(normalize=True).max() * 100

st.info(
    "This dashboard compares historical 2022 election results "
    "with voluntary public opinion poll responses."
)

# Ward selection

ward = st.selectbox(
    "Select Chennai Ward",
    all_wards
)

selected = data[data["ward_no"] == ward].iloc[0]
