
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
    "Ward-wise historical election analysis, public opinion polling "
    "and experimental 2027 projections across Chennai's 200 wards."
)

st.info(
    "Public poll responses are voluntary and may not represent the views "
    "of all Chennai voters. All 2027 projections are experimental estimates."
)

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

# Historical Random Forest model

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
    st.error(
        "Not enough valid historical data to train the Random Forest model."
    )
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

majority_baseline = (
    y_test.value_counts(normalize=True).max() * 100
)

# Ward selection

st.subheader("Ward-wise Election Analysis Dashboard")

ward = st.selectbox(
    "Select Chennai Ward",
    all_wards
)

selected = data[data["ward_no"] == ward].iloc[0]

# Historical vote share

st.write("## 2022 Historical Vote Share")

valid_votes = pd.to_numeric(
    selected["valid_votes"],
    errors="coerce"
)

historical = {}

for party_name, column in party_columns.items():
    votes = pd.to_numeric(
        selected[column],
        errors="coerce"
    )

    if pd.notna(valid_votes) and valid_votes > 0 and pd.notna(votes):
        historical[party_name] = round(
            (votes / valid_votes) * 100,
            2
        )
    else:
        historical[party_name] = 0

st.bar_chart(pd.Series(historical))

st.caption(
    "Historical percentages use valid votes. TVK did not contest in 2022. "
    "The historical dataset's displayed party columns do not include every party."
)

st.divider()

# Browser identification

st.write("## Current Public Opinion Poll")

browser_id = streamlit_js_eval(
    js_expressions="""
    (() => {
        let id = localStorage.getItem("chennai_poll_browser_id");

        if (!id) {
            id = crypto.randomUUID();
            localStorage.setItem("chennai_poll_browser_id", id);
        }

        return id;
    })()
    """,
    key="browser_id",
    want_output=True
)

if not browser_id:
    st.info("Preparing your browser ID. Please wait before voting.")

# Public poll form

with st.form("public_poll"):

    st.write(
        "If the election were held today, which party would you support?"
    )

    party = st.radio(
        "Select your preferred party",
        parties
    )

    st.write("Selected Ward:", int(ward))

    submit = st.form_submit_button("Submit Vote")

    if submit:

        if not browser_id:
            st.warning(
                "Please wait while your browser is being identified."
            )

        else:

            vote = {
                "party": party,
                "ward": int(ward),
                "browser_id": str(browser_id)
            }

            try:

                response = requests.post(
                    SUPABASE_URL + "/rest/v1/poll_responses",
                    headers=headers,
                    json=vote,
                    timeout=20
                )

                if response.status_code in [200, 201, 204]:

                    st.success(
                        "Your response has been recorded!"
                    )

                elif response.status_code == 409:

                    st.warning(
                        "You have already voted from this browser. "
                        "Only one response is allowed."
                    )

                    duplicate_log = {
                        "browser_id": str(browser_id),
                        "ward": int(ward)
                    }

                    try:

                        log_response = requests.post(
                            SUPABASE_URL + "/rest/v1/duplicate_vote_attempts",
                            headers=headers,
                            json=duplicate_log,
                            timeout=20
                        )

                        if log_response.status_code in [200, 201, 204]:
                            st.info("Duplicate attempt recorded.")
                        else:
                            st.error(
                                "Could not record duplicate attempt: "
                                + log_response.text
                            )

                    except requests.RequestException as e:
                        st.error("Logging error: " + str(e))

                else:
                    st.error(
                        "Unable to submit vote: " + response.text
                    )

            except requests.RequestException as e:
                st.error("Connection error: " + str(e))

st.divider()

# Load public poll results

st.write("## Four-Party Poll Comparison")

try:

    response = requests.get(
        SUPABASE_URL + "/rest/v1/poll_responses",
        headers=headers,
        params={
            "select": "party,ward",
            "limit": "10000"
        },
        timeout=20
    )

    response.raise_for_status()

    poll = pd.DataFrame(
        response.json(),
        columns=["party", "ward"]
    )

    poll["ward"] = pd.to_numeric(
        poll["ward"],
        errors="coerce"
    )

    poll = poll.dropna(
        subset=["ward", "party"]
    )

    poll["ward"] = poll["ward"].astype(int)

    poll = poll[poll["party"].isin(parties)]

    # Overall public poll results

    results = poll["party"].value_counts().reindex(
        parties,
        fill_value=0
    )

    st.metric("Total Poll Responses", len(poll))
    st.metric("Wards with Responses", poll["ward"].nunique())

    if len(poll) > 0:
        leading_party = results.idxmax()
    else:
        leading_party = "No responses"

    st.metric("Overall Poll Leading Party", leading_party)

    st.write("### Overall Poll Votes")
    st.bar_chart(results)

    if len(poll) > 0:
        percentages = results / len(poll) * 100

        st.write("### Party-wise Poll Percentage")
        st.bar_chart(percentages)

    # Selected ward poll results

    st.write("### Selected Ward Results")

    ward_poll = poll[poll["ward"] == int(ward)]

    ward_results = ward_poll["party"].value_counts().reindex(
        parties,
        fill_value=0
    )

    st.metric("Responses in Selected Ward", len(ward_poll))
    st.bar_chart(ward_results)

    # Ward-wise poll comparison

    st.write("### Ward-wise Party Comparison")

    ward_comparison = poll.pivot_table(
        index="ward",
        columns="party",
        aggfunc="size",
        fill_value=0
    )

    ward_comparison = ward_comparison.reindex(
        index=all_wards,
        columns=parties,
        fill_value=0
    ).sort_index()

    st.bar_chart(ward_comparison)

    # Experimental ward-wise projection

    st.divider()
    st.write("## 2027 Experimental Ward-wise Projection")

    projection_rows = []

    for ward_no in all_wards:

        ward_data = ward_comparison.loc[ward_no]
        total_responses = int(ward_data.sum())

        historical_row = data[
            data["ward_no"] == ward_no
        ]

        historical_shares = {
            party_name: 0.0 for party_name in parties
        }

        if not historical_row.empty:

            history = historical_row.iloc[0]

            valid = pd.to_numeric(
                history["valid_votes"],
                errors="coerce"
            )

            if pd.notna(valid) and valid > 0:

                for party_name, column in party_columns.items():

                    votes = pd.to_numeric(
                        history[column],
                        errors="coerce"
                    )

                    if pd.notna(votes):
                        historical_shares[party_name] = (
                            votes / valid
                        ) * 100

        # Normalize the three available historical party shares
        # so the historical baseline is comparable among four parties.

        historical_total = sum(historical_shares.values())

        if historical_total > 0:

            for party_name in parties:
                historical_shares[party_name] = (
                    historical_shares[party_name]
                    / historical_total
                ) * 100

        # Poll percentages for this ward

        if total_responses > 0:

            poll_shares = (
                ward_data / total_responses
            ) * 100

            # More poll responses increase the poll's weight.
            # This is a chosen experimental weighting rule,
            # not a statistically validated survey adjustment.

            poll_weight = total_responses / (
                total_responses + 20
            )

        else:

            poll_shares = pd.Series(
                0.0,
                index=parties
            )

            poll_weight = 0.0

        # Combine historical baseline and current poll

        estimated_shares = {}

        for party_name in parties:

            historical_value = historical_shares[party_name]
            poll_value = float(poll_shares[party_name])

            estimated_shares[party_name] = (
                historical_value * (1 - poll_weight)
                + poll_value * poll_weight
            )

        estimated_total = sum(estimated_shares.values())

        if estimated_total > 0:

            for party_name in parties:
                estimated_shares[party_name] = (
                    estimated_shares[party_name]
                    / estimated_total
                ) * 100

        projected_winner = max(
            estimated_shares,
            key=estimated_shares.get
        )

        projection_rows.append({

            "Ward": ward_no,
            "2022 Historical Winner": (
                historical_row.iloc[0]["winner_party_ta"]
                if not historical_row.empty
                else "Data unavailable"
            ),
            "Poll Responses": total_responses,
            "Poll Weight (%)": round(poll_weight * 100, 2),
            "TVK Estimated Share (%)": round(
                estimated_shares["TVK"], 2
            ),
            "DMK Estimated Share (%)": round(
                estimated_shares["DMK"], 2
            ),
            "AIADMK Estimated Share (%)": round(
                estimated_shares["AIADMK"], 2
            ),
            "NTK Estimated Share (%)": round(
                estimated_shares["NTK"], 2
            ),
            "Projected Leading Party": projected_winner

        })

    projection_data = pd.DataFrame(projection_rows)

    st.dataframe(
        projection_data,
        use_container_width=True
    )

    # Overall Chennai-level prediction

    st.divider()
    st.title("Chennai Overall 2027 Experimental Prediction")

    st.write("### Party-wise Estimated Vote Share")

    share_columns = {
        "TVK": "TVK Estimated Share (%)",
        "DMK": "DMK Estimated Share (%)",
        "AIADMK": "AIADMK Estimated Share (%)",
        "NTK": "NTK Estimated Share (%)"
    }

    overall_shares = {}

    for party_name, column in share_columns.items():
        overall_shares[party_name] = round(
            projection_data[column].mean(),
            2
        )

    overall_share_series = pd.Series(overall_shares)

    st.bar_chart(overall_share_series)

    share_table = pd.DataFrame({
        "Party": overall_share_series.index,
        "Estimated Vote Share (%)": overall_share_series.values
    })

    st.dataframe(
        share_table,
        use_container_width=True
    )

    # Projected seat calculation

    st.write("### Projected Seats Across 200 Wards")

    seat_results = (
        projection_data["Projected Leading Party"]
        .value_counts()
        .reindex(parties, fill_value=0)
    )

    seat_table = pd.DataFrame({
        "Party": seat_results.index,
        "Projected Seats": seat_results.values
    })

    st.bar_chart(seat_results)

    st.dataframe(
        seat_table,
        use_container_width=True
    )

    total_projected_seats = int(seat_results.sum())

    st.metric(
        "Total Projected Seats",
        total_projected_seats
    )

    # Ward coverage

    wards_with_poll = int(
        (projection_data["Poll Responses"] > 0).sum()
    )

    wards_without_poll = len(projection_data) - wards_with_poll

    st.write("### Prediction Coverage")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Wards",
        len(projection_data)
    )

    col2.metric(
        "Wards with Poll Responses",
        wards_with_poll
    )

    col3.metric(
        "Wards without Poll Responses",
        wards_without_poll
    )

    st.caption(
        "Method: historical DMK, AIADMK and NTK vote shares are normalized "
        "within the three available historical party columns. TVK has no "
        "2022 baseline. Each ward's historical baseline is blended with its "
        "four-party poll share using poll_weight = responses / (responses + 20). "
        "Overall vote shares are simple averages across wards, not elector-weighted "
        "Chennai-wide vote shares. Projected seats count the highest estimated "
        "share in each ward. These are experimental estimates, not validated "
        "predictions or official election results."
    )

    # Leading party in each ward

    st.write("### Leading Party in Each Ward")

    st.dataframe(
        projection_data[
            [
                "Ward",
                "Poll Responses",
                "Projected Leading Party"
            ]
        ],
        use_container_width=True
    )

    # Original historical comparison

    st.divider()
    st.write("## All Chennai Ward Information")

    st.dataframe(
        data[
            [
                "ward_no",
                "winner_party_ta",
                "margin",
                "polled_pct"
            ]
        ],
        use_container_width=True
    )

except requests.RequestException as e:

    st.error(
        "Unable to load poll results: " + str(e)
    )

except Exception as e:

    st.error(
        "Poll results error: " + str(e)
    )

# Random Forest results

st.divider()

st.write("## Random Forest Model - Historical Test Results")

st.metric(
    "Historical Test Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.metric(
    "Majority-Class Baseline",
    f"{majority_baseline:.2f}%"
)

comparison = pd.DataFrame({

    "Actual Winner": y_test.values,
    "Predicted Winner": predictions

})

st.write("### Actual vs Predicted Historical Winners")

st.dataframe(
    comparison,
    use_container_width=True
)

st.caption(
    "This model uses elector count and candidate count to classify "
    "historical 2022 ward winners. Its test accuracy measures held-out "
    "2022 wards only. It does not establish 2027 prediction accuracy."
)

# Admin Dashboard

st.divider()
st.title("Admin Dashboard")

admin_password = st.text_input(
    "Enter Admin Password",
    type="password"
)

if st.button("Admin Login"):

    if admin_password == st.secrets.get("ADMIN_PASSWORD", ""):
        st.session_state["admin_logged_in"] = True
        st.success("Admin login successful!")
    else:
        st.session_state["admin_logged_in"] = False
        st.error("Incorrect password!")

if st.session_state.get("admin_logged_in", False):

    st.subheader("Election Poll Monitoring")

    try:

        admin_response = requests.get(
            SUPABASE_URL + "/rest/v1/poll_responses",
            headers=headers,
            params={
                "select": "party,ward",
                "limit": "10000"
            },
            timeout=20
        )

        admin_response.raise_for_status()

        admin_poll = pd.DataFrame(
            admin_response.json(),
            columns=["party", "ward"]
        )

        st.metric(
            "Total Poll Responses",
            len(admin_poll)
        )

        st.metric(
            "Participating Wards",
            admin_poll["ward"].nunique()
        )

        st.write("### Party-wise Poll Results")

        admin_results = admin_poll["party"].value_counts().reindex(
            parties,
            fill_value=0
        )

        st.bar_chart(admin_results)

        st.write("### Ward-wise Poll Participation")

        admin_ward_results = (
            admin_poll.groupby("ward")
            .size()
            .reindex(all_wards, fill_value=0)
        )

        st.bar_chart(admin_ward_results)

        st.write("### Poll Response Data")

        st.dataframe(
            admin_poll,
            use_container_width=True
        )

    except requests.RequestException as e:
        st.error("Poll monitoring error: " + str(e))

    except Exception as e:
        st.error("Poll monitoring error: " + str(e))

    # Duplicate attempt monitoring

    st.divider()
    st.subheader("Duplicate Vote Attempt Monitoring")

    service_key = str(
        st.secrets.get("SUPABASE_SERVICE_KEY", "")
    ).strip()

    if not service_key:

        st.error(
            "Supabase server-side secret key is missing from Streamlit Secrets."
        )

    elif not service_key.isascii():

        st.error(
            "The Supabase server-side secret key contains unsupported characters. "
            "Please check Streamlit Secrets."
        )

    else:

        private_headers = {
            "apikey": service_key,
            "Authorization": "Bearer " + service_key,
            "Content-Type": "application/json"
        }

        try:

            duplicate_response = requests.get(
                SUPABASE_URL + "/rest/v1/duplicate_vote_attempts",
                headers=private_headers,
                params={
                    "select": "id,browser_id,ward,attempted_at",
                    "order": "attempted_at.desc",
                    "limit": "10000"
                },
                timeout=20
            )

            duplicate_response.raise_for_status()

            duplicate_data = pd.DataFrame(
                duplicate_response.json(),
                columns=[
                    "id",
                    "browser_id",
                    "ward",
                    "attempted_at"
                ]
            )

            st.metric(
                "Total Duplicate Attempts",
                len(duplicate_data)
            )

            if not duplicate_data.empty:

                st.write("### Ward-wise Duplicate Attempts")

                ward_duplicates = (
                    duplicate_data.groupby("ward")
                    .size()
                    .sort_index()
                )

                st.bar_chart(ward_duplicates)

                st.write("### Duplicate Attempt History")

                st.dataframe(
                    duplicate_data,
                    use_container_width=True
                )

            else:

                st.info(
                    "No duplicate voting attempts have been recorded yet."
                )

        except requests.RequestException as e:

            st.error(
                "Unable to load duplicate attempts: " + str(e)
            )

        except Exception as e:

            st.error(
                "Duplicate monitoring error: " + str(e)
            )

    if st.button("Admin Logout"):

        st.session_state["admin_logged_in"] = False
        st.rerun()
