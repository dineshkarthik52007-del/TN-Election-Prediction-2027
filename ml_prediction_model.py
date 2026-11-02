
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PARTIES = ["TVK", "DMK", "AIADMK", "NTK"]


def train_ml_model(historical_data):

    data = historical_data.copy()

    data["electors"] = pd.to_numeric(
        data["electors"], errors="coerce"
    )

    data["candidates"] = pd.to_numeric(
        data["candidates"], errors="coerce"
    )

    data = data.dropna(
        subset=["electors", "candidates", "winner_party_ta"]
    )

    data = data[
        data["winner_party_ta"].isin([
            "தி.மு.க",
            "அ.இ.அ.தி.மு.க",
            "நா.த.க."
        ])
    ]

    if len(data) < 5:
        return None

    X = data[["electors", "candidates"]]
    y = data["winner_party_ta"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X, y)

    return model


def predict_ward(model, electors, candidates):

    if model is None:
        return None

    features = pd.DataFrame([{
        "electors": electors,
        "candidates": candidates
    }])

    return model.predict(features)[0]


def build_ml_predictions(historical_data, poll_data):

    model = train_ml_model(historical_data)

    if model is None:
        return None

    data = historical_data.copy()
    poll = poll_data.copy()

    data["ward_no"] = pd.to_numeric(
        data["ward_no"], errors="coerce"
    )

    data["electors"] = pd.to_numeric(
        data["electors"], errors="coerce"
    )

    data["candidates"] = pd.to_numeric(
        data["candidates"], errors="coerce"
    )

    poll["ward"] = pd.to_numeric(
        poll["ward"], errors="coerce"
    )

    poll = poll.dropna(subset=["ward", "party"])
    poll["ward"] = poll["ward"].astype(int)

    poll = poll[poll["party"].isin(PARTIES)]

    overall_counts = (
        poll["party"]
        .value_counts()
        .reindex(PARTIES, fill_value=0)
    )

    if overall_counts.sum() == 0:
        return None

    overall_poll = overall_counts / overall_counts.sum()

    party_mapping = {
        "தி.மு.க": "DMK",
        "அ.இ.அ.தி.மு.க": "AIADMK",
        "நா.த.க.": "NTK"
    }

    wards = sorted(data["ward_no"].dropna().unique())

    results = []

    for ward in wards:

        row = data[data["ward_no"] == ward].iloc[0]

        electors = row["electors"]
        candidates = row["candidates"]

        if pd.isna(electors) or pd.isna(candidates):
            continue

        features = pd.DataFrame([{
            "electors": electors,
            "candidates": candidates
        }])

        historical_probabilities = {
            party: 0.0 for party in PARTIES
        }

        probabilities = model.predict_proba(features)[0]

        for label, probability in zip(
            model.classes_, probabilities
        ):

            party = party_mapping.get(label)

            if party:
                historical_probabilities[party] = float(probability)

        ward_poll = poll[poll["ward"] == ward]

        ward_counts = (
            ward_poll["party"]
            .value_counts()
            .reindex(PARTIES, fill_value=0)
        )

        ward_total = int(ward_counts.sum())

        smoothed_poll = (
            ward_counts + 20 * overall_poll
        ) / (ward_total + 20)

        poll_weight = ward_total / (ward_total + 20)

        estimates = {}

        for party in PARTIES:

            poll_estimate = (
                poll_weight * smoothed_poll[party]
                + (1 - poll_weight) * overall_poll[party]
            )

            if party == "TVK":
                estimates[party] = poll_estimate
            else:
                estimates[party] = (
                    0.5 * historical_probabilities[party]
                    + 0.5 * poll_estimate
                )

        total = sum(estimates.values())

        if total <= 0:
            continue

        estimates = {
            party: value / total * 100
            for party, value in estimates.items()
        }

        winner = max(estimates, key=estimates.get)

        result = {
            "Ward": int(ward),
            "Poll Responses": ward_total,
            "ML Estimated Leading Party": winner
        }

        for party in PARTIES:
            result[party + " ML Share (%)"] = round(
                estimates[party], 2
            )

        results.append(result)

    return pd.DataFrame(results)
