
import streamlit as st
import pandas as pd
import os

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

historical_chart = pd.Series(historical)
st.bar_chart(historical_chart)

st.caption("TVK did not contest in 2022, so its historical vote share is unavailable.")

st.divider()

st.write("## Current Public Opinion Poll")

poll_file = "poll/ward_responses.csv"

if os.path.exists(poll_file):

    poll = pd.read_csv(poll_file)

    if "ward" in poll.columns:

        ward_name = "Ward " + str(int(ward))

        ward_poll = poll[poll["ward"] == ward_name]

        poll_results = ward_poll["party"].value_counts().reindex(
            parties, fill_value=0
        )

        total = len(ward_poll)

        st.metric("Poll Responses in Selected Ward", total)

        st.write("### Four-Party Poll Comparison")
        st.bar_chart(poll_results)

        st.write("### Poll Response Count")

        st.dataframe(
            poll_results.rename("Responses").reset_index()
        )

    else:
        st.warning("No ward information found in poll data.")

else:
    st.info("No poll responses collected yet.")

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
