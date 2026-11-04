import streamlit as st
import pandas as pd
import requests
import hashlib
import hmac
import re
import os

from streamlit_js_eval import streamlit_js_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from prediction_model import build_prediction_model
from ml_prediction_model import build_ml_predictions


# Supabase configuration

SUPABASE_URL = "https://xkiaxjhifmnsxthxkwlk.supabase.co"

SUPABASE_KEY = str(
    st.secrets.get("SUPABASE_KEY", "")
).strip()

PHONE_HASH_SECRET = str(
    st.secrets.get("PHONE_HASH_SECRET", "")
).strip()

if not SUPABASE_KEY:
    st.error("Supabase API key is missing from Streamlit Secrets.")
    st.stop()


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
    "Public poll responses are voluntary and may not represent all Chennai voters. "
    "All 2027 projections are experimental estimates."
)


# Load historical data

data = pd.read_csv("chennai_complete_2022.csv")

parties = ["TVK", "DMK", "AIADMK", "NTK"]

party_columns = {
    "DMK": "தி.மு.க",
    "AIADMK": "அ.இ.அ.தி.மு.க",
    "NTK": "நா.த.க."
}

party_leaders = {
    "TVK": {
        "leader": "Vijay",
        "image": "website/images/vijay.jpg"
    },
    "DMK": {
        "leader": "M. K. Stalin",
        "image": "website/images/stalin.jpg"
    },
    "AIADMK": {
        "leader": "Edappadi K. Palaniswami",
        "image": "website/images/eps.jpg"
    },
    "NTK": {
        "leader": "Seeman",
        "image": "website/images/seeman.jpg"
    }
}

all_wards = sorted(data["ward_no"].dropna().unique())


# Supabase headers

headers = {
    "apikey": SUPABASE_KEY,
    "Content-Type": "application/json"
}


# Historical Random Forest model

ml_features = ["electors", "candidates"]

ml_data = data[
    ml_features + ["winner_party_ta"]
].copy()

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

selected = data[
    data["ward_no"] == ward
].iloc[0]


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

    if (
        pd.notna(valid_votes)
        and valid_votes > 0
        and pd.notna(votes)
    ):
        historical[party_name] = round(
            (votes / valid_votes) * 100,
            2
        )
    else:
        historical[party_name] = 0

st.bar_chart(pd.Series(historical))

st.caption(
    "Historical percentages use valid votes. TVK did not contest in 2022. "
    "The displayed historical party columns do not include every party."
)

st.divider()


# Browser identification

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


# Party selection

st.write("## Current Public Opinion Poll")

st.caption(
    "Verify your mobile number before submitting your response. "
    "A keyed hash is stored instead of your actual mobile number."
)

st.write(
    "If the election were held today, which party would you support?"
)

if "selected_party" not in st.session_state:
    st.session_state["selected_party"] = None


def choose_party(party_name):

    st.session_state["selected_party"] = party_name

    for other_party in parties:

        if other_party != party_name:

            st.session_state["party_radio_" + other_party] = None


st.markdown("""
<style>
.st-key-poll_party_rows [data-testid="stHorizontalBlock"] {
    gap: 0.25rem;
    align-items: center;
}

.st-key-poll_party_rows [data-testid="column"] {
    padding-left: 0;
    padding-right: 0;
}

.st-key-poll_party_rows [data-testid="stImage"] {
    margin: 0;
}

.st-key-poll_party_rows [data-testid="stRadio"] {
    margin: 0;
    padding: 0;
}
</style>
""", unsafe_allow_html=True)


# Updated party images with file checking

with st.container(key="poll_party_rows"):

    for party_name in parties:

        photo_col, name_col, radio_col = st.columns(
            [0.9, 1.5, 0.35],
            gap="small",
            vertical_alignment="center"
        )

        with photo_col:

            image_path = party_leaders[party_name]["image"]

            if os.path.isfile(image_path):

                st.image(
                    image_path,
                    width=100
                )

            else:

                st.error(
                    "Image not found: " + image_path
                )

        with name_col:

            st.markdown(
                f"**{party_name}**  \n"
                f"{party_leaders[party_name]['leader']}"
            )

        with radio_col:

            st.radio(
                "Select " + party_name,
                [party_name],
                index=None,
                key="party_radio_" + party_name,
                on_change=choose_party,
                args=(party_name,),
                label_visibility="collapsed"
            )


# Phone number normalization

def normalize_phone(number):

    cleaned = re.sub(r"\D", "", number)

    if cleaned.startswith("91") and len(cleaned) == 12:
        cleaned = cleaned[2:]

    if re.fullmatch(r"[6-9][0-9]{9}", cleaned):

        return "+91" + cleaned, cleaned

    return None, None


# OTP session initialization

if "otp_pending_phone" not in st.session_state:
    st.session_state["otp_pending_phone"] = None

if "otp_sent" not in st.session_state:
    st.session_state["otp_sent"] = False

if "verified_phone" not in st.session_state:
    st.session_state["verified_phone"] = None

if "phone_access_token" not in st.session_state:
    st.session_state["phone_access_token"] = None


# Mobile verification

st.divider()

st.write("## Mobile Number Verification")

mobile_number = st.text_input(
    "Enter your 10-digit Indian mobile number",
    max_chars=15,
    placeholder="Example: 9876543210",
    key="otp_mobile"
)

phone_e164, cleaned_number = normalize_phone(mobile_number)


# Reset verification if number changes

if (
    st.session_state["otp_pending_phone"]
    and phone_e164 != st.session_state["otp_pending_phone"]
):

    st.session_state["otp_pending_phone"] = None
    st.session_state["otp_sent"] = False
    st.session_state["verified_phone"] = None
    st.session_state["phone_access_token"] = None


if (
    st.session_state["verified_phone"]
    and phone_e164 != st.session_state["verified_phone"]
):

    st.session_state["verified_phone"] = None
    st.session_state["phone_access_token"] = None


# Send OTP

if not st.session_state["verified_phone"]:

    if st.button("Send OTP"):

        if not phone_e164:

            st.warning(
                "Enter a valid 10-digit Indian mobile number."
            )

        else:

            try:

                auth_response = requests.post(
                    SUPABASE_URL + "/auth/v1/otp",
                    headers=headers,
                    json={
                        "phone": phone_e164,
                        "create_user": True
                    },
                    timeout=20
                )

                if auth_response.status_code in [200, 201]:

                    st.session_state["otp_pending_phone"] = phone_e164
                    st.session_state["otp_sent"] = True

                    st.success(
                        "OTP request accepted. Enter your configured test OTP."
                    )

                else:

                    st.error(
                        "OTP request failed: "
                        + auth_response.text
                    )

            except requests.RequestException as e:

                st.error("Connection error: " + str(e))


# Verify OTP

if (
    st.session_state["otp_sent"]
    and not st.session_state["verified_phone"]
):

    with st.form("verify_phone_otp"):

        otp_code = st.text_input(
            "Enter your 6-digit OTP",
            max_chars=6,
            type="password"
        )

        verify_button = st.form_submit_button("Verify OTP")

        if verify_button:

            if not re.fullmatch(r"[0-9]{6}", otp_code):

                st.warning("Enter a valid 6-digit OTP.")

            else:

                try:

                    verify_response = requests.post(
                        SUPABASE_URL + "/auth/v1/verify",
                        headers=headers,
                        json={
                            "phone": st.session_state["otp_pending_phone"],
                            "token": otp_code,
                            "type": "sms"
                        },
                        timeout=20
                    )

                    if verify_response.status_code == 200:

                        auth_data = verify_response.json()

                        access_token = auth_data.get("access_token")

                        if not access_token:

                            st.error(
                                "Verification response did not contain "
                                "an access token."
                            )

                        else:

                            st.session_state["verified_phone"] = (
                                st.session_state["otp_pending_phone"]
                            )

                            st.session_state["phone_access_token"] = (
                                access_token
                            )

                            st.success(
                                "Mobile number verified successfully!"
                            )

                            st.rerun()

                    else:

                        st.error(
                            "OTP verification failed: "
                            + verify_response.text
                        )

                except requests.RequestException as e:

                    st.error(
                        "Verification connection error: " + str(e)
                    )


# Vote submission after verification

phone_verified = (
    st.session_state["verified_phone"] == phone_e164
    and phone_e164 is not None
    and st.session_state["phone_access_token"] is not None
)

if phone_verified:

    st.success("Mobile number verified successfully.")

    st.write(
        "Verified number: +91******"
        + cleaned_number[-4:]
    )

    st.write("## Submit Your Vote")

    st.write("Selected Ward:", int(ward))

    selected_party = st.session_state.get("selected_party")

    if selected_party:
        st.success("Selected Party: " + selected_party)

    with st.form("public_poll"):

        submit = st.form_submit_button("Submit Vote")

        if submit:

            party = st.session_state.get("selected_party")

            if not PHONE_HASH_SECRET:

                st.error(
                    "Mobile-number protection is not configured."
                )

            elif party is None:

                st.warning(
                    "Please select a party before submitting."
                )

            elif not browser_id:

                st.warning(
                    "Please wait while your browser is identified."
                )

            else:

                phone_hash = hmac.new(
                    PHONE_HASH_SECRET.encode("utf-8"),
                    cleaned_number.encode("utf-8"),
                    hashlib.sha256
                ).hexdigest()

                vote = {
                    "party": party,
                    "ward": int(ward),
                    "phone_hash": phone_hash,
                    "browser_id": str(browser_id)
                }

                authenticated_headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": (
                        "Bearer "
                        + st.session_state["phone_access_token"]
                    ),
                    "Content-Type": "application/json"
                }

                try:

                    response = requests.post(
                        SUPABASE_URL + "/rest/v1/poll_responses",
                        headers=authenticated_headers,
                        json=vote,
                        timeout=20
                    )

                    if response.status_code in [200, 201, 204]:

                        st.success(
                            "Your response has been recorded!"
                        )

                    elif response.status_code == 409:

                        st.warning(
                            "This mobile number or browser has already voted. "
                            "Only one response is allowed per mobile number."
                        )

                        duplicate_log = {
                            "browser_id": str(browser_id),
                            "ward": int(ward)
                        }

                        try:

                            log_response = requests.post(
                                SUPABASE_URL
                                + "/rest/v1/duplicate_vote_attempts",
                                headers=authenticated_headers,
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
                            "Unable to submit vote: "
                            + response.text
                        )

                except requests.RequestException as e:

                    st.error(
                        "Connection error: " + str(e)
                    )

else:

    st.info(
        "Verify your mobile number before submitting your vote."
    )
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

    results = poll["party"].value_counts().reindex(
        parties,
        fill_value=0
    )

    st.metric("Total Poll Responses", len(poll))

    st.metric(
        "Wards with Responses",
        poll["ward"].nunique()
    )

    leading_party = (
        results.idxmax() if len(poll) > 0 else "No responses"
    )

    st.metric(
        "Overall Poll Leading Party",
        leading_party
    )

    st.write("### Overall Poll Votes")
    st.bar_chart(results)

    if len(poll) > 0:

        percentages = results / len(poll) * 100

        st.write("### Party-wise Poll Percentage")
        st.bar_chart(percentages)

    st.write("### Selected Ward Results")

    ward_poll = poll[poll["ward"] == int(ward)]

    ward_results = ward_poll["party"].value_counts().reindex(
        parties,
        fill_value=0
    )

    st.metric(
        "Responses in Selected Ward",
        len(ward_poll)
    )

    st.bar_chart(ward_results)

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


    # Statistical prediction model

    st.divider()

    st.title("Chennai Overall 2027 Prediction Model")

    projection_data, overall_shares, projected_seats = (
        build_prediction_model(data, poll)
    )

    if projection_data is None:

        st.warning(
            "No public poll responses are available yet. "
            "The prediction model needs poll data."
        )

    else:

        st.write("### Party-wise Estimated Vote Share")

        share_series = pd.Series(overall_shares)

        st.bar_chart(share_series)

        st.dataframe(
            pd.DataFrame({
                "Party": share_series.index,
                "Estimated Vote Share (%)": share_series.values
            }),
            use_container_width=True
        )

        st.write("### Projected Seats Across 200 Wards")

        st.bar_chart(projected_seats)

        st.dataframe(
            pd.DataFrame({
                "Party": projected_seats.index,
                "Projected Seats": projected_seats.values
            }),
            use_container_width=True
        )

        st.metric(
            "Total Projected Seats",
            int(projected_seats.sum())
        )

        st.write("### Prediction Coverage")

        col1, col2 = st.columns(2)

        col1.metric(
            "Wards with Poll Responses",
            int(
                (projection_data["Poll Responses"] > 0).sum()
            )
        )

        col2.metric(
            "Wards without Poll Responses",
            int(
                (projection_data["Poll Responses"] == 0).sum()
            )
        )

        st.write("### Ward-wise Prediction")

        st.dataframe(
            projection_data,
            use_container_width=True
        )

        st.caption(
            "This experimental model uses four-party public poll responses "
            "with statistical smoothing. It does not use TVK historical votes, "
            "which are unavailable for 2022. Chennai-wide shares are weighted "
            "using 2022 ward elector counts. This is not a validated election "
            "forecast or a representative survey."
        )


    # Machine Learning prediction model

    st.divider()

    st.title("Machine Learning 2027 Prediction Model")

    ml_projection = build_ml_predictions(data, poll)

    if ml_projection is None or ml_projection.empty:

        st.warning(
            "ML estimates are unavailable. Check historical data "
            "and public poll responses."
        )

    else:

        st.write("### ML Party-wise Estimated Vote Share")

        ml_weights = data[
            ["ward_no", "electors"]
        ].copy()

        ml_weights["electors"] = pd.to_numeric(
            ml_weights["electors"],
            errors="coerce"
        )

        ml_projection = ml_projection.merge(
            ml_weights,
            left_on="Ward",
            right_on="ward_no",
            how="left"
        )

        weights = (
            ml_projection["electors"]
            .fillna(0)
            .clip(lower=0)
        )

        ml_shares = {}

        for p in parties:

            column = p + " ML Share (%)"

            if weights.sum() > 0:

                ml_shares[p] = round(
                    float(
                        (ml_projection[column] * weights).sum()
                        / weights.sum()
                    ),
                    2
                )

            else:

                ml_shares[p] = round(
                    float(ml_projection[column].mean()),
                    2
                )

        ml_share_series = pd.Series(ml_shares)

        st.bar_chart(ml_share_series)

        st.dataframe(
            pd.DataFrame({
                "Party": ml_share_series.index,
                "ML Estimated Vote Share (%)": ml_share_series.values
            }),
            use_container_width=True
        )

        st.write("### ML Projected Seats")

        ml_seats = (
            ml_projection["ML Estimated Leading Party"]
            .value_counts()
            .reindex(parties, fill_value=0)
        )

        st.bar_chart(ml_seats)

        st.dataframe(
            pd.DataFrame({
                "Party": ml_seats.index,
                "ML Projected Seats": ml_seats.values
            }),
            use_container_width=True
        )

        st.metric(
            "Wards with ML Estimates",
            len(ml_projection)
        )

        st.metric(
            "Total ML Projected Seats",
            int(ml_seats.sum())
        )

        st.write("### ML Ward-wise Predictions")

        st.dataframe(
            ml_projection.drop(
                columns=["ward_no", "electors"],
                errors="ignore"
            ),
            use_container_width=True
        )

        st.caption(
            "These are experimental ML-assisted estimates. The Random "
            "Forest learns from 2022 historical ward winners using "
            "elector and candidate counts. Public poll data is blended "
            "with historical probabilities using a heuristic. TVK has "
            "no 2022 historical training data. These estimates are not "
            "validated 2027 election forecasts."
        )


    # All Chennai ward information

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


# Random Forest historical results

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


# Admin dashboard

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
            "The Supabase server-side secret key contains unsupported "
            "characters. Please check Streamlit Secrets."
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

st.divider()
