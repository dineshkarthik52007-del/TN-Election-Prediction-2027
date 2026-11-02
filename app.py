import streamlit as st
import pandas as pd
import requests
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
ml_features = [
    "electors",
    "polled",
    "polled_pct",
    "valid_votes",
    "candidates"
]

ml_data = data[ml_features + ["winner_party_ta"]].copy()

for column in ml_features:
    ml_data[column] = pd.to_numeric(
        ml_data[column], errors="coerce"
    )

ml_data = ml_data.dropna()

X = ml_data[ml_features]
y = ml_data["winner_party_ta"]

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


parties = ["TVK", "DMK", "AIADMK", "NTK"]

party_columns = {
    "DMK": "தி.மு.க",
    "AIADMK": "அ.இ.அ.தி.மு.க",
    "NTK": "நா.த.க."
}

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": "Bearer " + SUPABASE_KEY,
    "Content-Type": "application/json"
}

st.info(
    "This dashboard compares historical 2022 election results "
    "with voluntary public opinion poll responses."
)

ward = st.selectbox(
    "Select Chennai Ward",
    sorted(data["ward_no"].unique())
)

selected = data[data["ward_no"] == ward].iloc[0]

st.write("## 2022 Historical Vote Share")

historical = {}

for party, column in party_columns.items():
    historical[party] = selected[column]

st.bar_chart(pd.Series(historical))

st.caption(
    "TVK did not contest in 2022, so its historical vote share is unavailable."
)

st.divider()

st.write("## Current Public Opinion Poll")

with st.form("public_poll"):
    st.write(
        "If the election were held today, which party would you support?"
    )

    party = st.radio("Select your preferred party", parties)

    st.write("Selected Ward:", int(ward))

    submit = st.form_submit_button("Submit Vote")

    if submit:
        vote = {
            "party": party,
            "ward": int(ward)
        }

        try:
            response = requests.post(
                SUPABASE_URL + "/rest/v1/poll_responses",
                headers=headers,
                json=vote
            )

            if response.status_code in [200, 201, 204]:
                st.success("Your response has been recorded!")
            else:
                st.error("Unable to submit vote: " + response.text)

        except Exception as e:
            st.error("Connection error: " + str(e))

st.write("### Four-Party Poll Comparison")

try:
    response = requests.get(
        SUPABASE_URL + "/rest/v1/poll_responses",
        headers=headers,
        params={"select": "party,ward", "limit": "10000"}
    )

    response.raise_for_status()

    poll = pd.DataFrame(
        response.json(),
        columns=["party", "ward"]
    )

    results = poll["party"].value_counts().reindex(
        parties, fill_value=0
    )

    st.metric("Total Poll Responses", len(poll))
    st.metric("Wards with Responses", poll["ward"].nunique())

    leading_party = results.idxmax() if len(poll) > 0 else "No responses"
    st.metric("Overall Leading Party", leading_party)
    st.bar_chart(results)
       
    if len(poll) > 0:
        percentages = (results / len(poll)) * 100
        st.write("### Party-wise Poll Percentage")
        st.bar_chart(percentages)

    st.write("### Selected Ward Results")

    ward_poll = poll[poll["ward"] == int(ward)]

    ward_results = ward_poll["party"].value_counts().reindex(
        parties, fill_value=0
    )

    st.metric("Responses in Selected Ward", len(ward_poll))
    st.bar_chart(ward_results)
    st.write("### Ward-wise Party Comparison")

    ward_comparison = poll.pivot_table(
        index="ward",
        columns="party",
        aggfunc="size",
        fill_value=0
    )

    ward_comparison = ward_comparison.reindex(
        columns=parties,
        fill_value=0
    ).sort_index()

    st.bar_chart(ward_comparison)
    st.divider()
    st.write("## 2027 Experimental Ward-wise Projection")

    projection_rows = []

    for ward_no in ward_comparison.index:
        ward_data = ward_comparison.loc[ward_no]
        total_responses = int(ward_data.sum())

        if total_responses > 0:
            leading_party = ward_data.idxmax()
            leading_votes = int(ward_data.max())
            poll_percentage = round(
                (leading_votes / total_responses) * 100, 2
            )
        else:
            leading_party = "Not enough responses"
            leading_votes = 0
            poll_percentage = 0

        historical_row = data[
            data["ward_no"] == int(ward_no)
        ]

        if not historical_row.empty:
            historical_winner = historical_row.iloc[0]["winner_party_ta"]
        else:
            historical_winner = "Data unavailable"

        projection_rows.append({
            "Ward": ward_no,
            "2022 Historical Winner": historical_winner,
            "2027 Poll Leading Party": leading_party,
            "Poll Responses": total_responses,
            "Leading Party Poll Share (%)": poll_percentage
        })

    projection_data = pd.DataFrame(projection_rows)

    st.dataframe(projection_data, use_container_width=True)

    st.caption(
        "Experimental projection based on voluntary public poll responses. "
        "It is not an official election forecast or a trained machine-learning prediction."
    )
    st.write("### Leading Party in Each Ward")

    leading_results = pd.DataFrame({
        "Leading Party": ward_comparison.idxmax(axis=1),
        "Poll Responses": ward_comparison.max(axis=1)
    })

    st.dataframe(leading_results, use_container_width=True)

except Exception as e:
    st.error("Unable to load poll results: " + str(e))

st.divider()

st.write("## All Chennai Ward Information")

st.dataframe(
    data[["ward_no", "winner_party_ta", "margin", "polled_pct"]],
    use_container_width=True
)

st.caption(
    "Public opinion poll responses are voluntary and are not "
    "representative election forecasts."
)
st.divider()

st.write("## Random Forest Model - Historical Test Results")

majority_baseline = y_test.value_counts(normalize=True).max() * 100

st.metric("Historical Test Accuracy", f"{accuracy * 100:.2f}%")
st.metric("Majority-Class Baseline", f"{majority_baseline:.2f}%")

st.write("### Actual vs Predicted Historical Winners")

comparison = pd.DataFrame({
    "Actual Winner": y_test.values,
    "Predicted Winner": predictions
})

st.dataframe(comparison, use_container_width=True)

st.caption(
    "This test measures classification of held-out 2022 wards only. "
    "It does not measure or guarantee 2027 election prediction accuracy."
)
