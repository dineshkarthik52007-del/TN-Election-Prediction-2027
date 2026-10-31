
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Chennai Public Opinion Poll")

st.title("Chennai Local Body Election 2027")
st.subheader("Public Opinion Poll")

parties = ["TVK", "DMK", "AIADMK", "NTK"]

st.write("If the election were held today, which party would you support?")

party = st.radio("Select your preferred party", parties)

ward = st.selectbox(
    "Select your Chennai ward",
    ["Ward " + str(i) for i in range(1, 201)]
)

file =  "poll/ward_responses.csv"

if st.button("Submit Vote"):
    new_vote = pd.DataFrame({
        "party": [party],
        "ward": [ward]
    })

    if os.path.exists(file):
        new_vote.to_csv(file, mode="a", header=False, index=False)
    else:
        new_vote.to_csv(file, index=False)

    st.success("Your response has been recorded!")

st.divider()

st.subheader("Public Opinion Poll Results")

if os.path.exists(file):
    data = pd.read_csv(file)

    if "ward" not in data.columns:
        data["ward"] = "Not recorded"

    results = data["party"].value_counts().reindex(
        parties, fill_value=0
    )

    st.metric("Total Responses", len(data))

    st.write("### Party-wise Responses")
    st.bar_chart(results)

    st.write("### Ward-wise Analysis")

    selected_ward = st.selectbox(
        "Choose ward to analyse",
        ["Ward " + str(i) for i in range(1, 201)],
        key="analysis_ward"
    )

    ward_data = data[data["ward"] == selected_ward]

    ward_results = ward_data["party"].value_counts().reindex(
        parties, fill_value=0
    )

    st.write("Responses for " + selected_ward)
    st.bar_chart(ward_results)

    st.write("### Ward-wise Response Details")
    st.dataframe(
        pd.crosstab(data["ward"], data["party"]).reindex(
            columns=parties, fill_value=0
        )
    )
else:
    st.info("No responses submitted yet.")

st.caption(
    "This is a voluntary public opinion poll, not an official election survey."
)
