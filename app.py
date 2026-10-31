import streamlit as st
import pandas as pd
import requests

SUPABASE_URL = "https://xkiaxjhifmnsxthxkwlk.supabase.co"
SUPABASE_KEY = "sb_publishable__WT7GeEGizjxAKilJEQjhg_LbjejXtq"

st.set_page_config(
    page_title="Chennai Election Analysis 2027",
    layout="wide"
)

st.title("Chennai Local Body Election 2027")
st.subheader("Ward-wise Election Analysis Dashboard")

data = pd.read_csv("chennai_complete_2022.csv")

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
    st.bar_chart(results)

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
