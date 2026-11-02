
import pandas as pd

PARTIES = ["TVK", "DMK", "AIADMK", "NTK"]


def build_prediction_model(historical_data, poll_data):

    wards = sorted(
        historical_data["ward_no"].dropna().unique()
    )

    poll_data = poll_data.copy()

    poll_data["ward"] = pd.to_numeric(
        poll_data["ward"],
        errors="coerce"
    )

    poll_data = poll_data.dropna(
        subset=["ward", "party"]
    )

    poll_data["ward"] = poll_data["ward"].astype(int)

    poll_data = poll_data[
        poll_data["party"].isin(PARTIES)
    ]

    overall_counts = (
        poll_data["party"]
        .value_counts()
        .reindex(PARTIES, fill_value=0)
    )

    total_poll = int(overall_counts.sum())

    if total_poll == 0:
        return None, None, None

    overall_shares = overall_counts / total_poll

    ward_rows = []

    for ward in wards:

        ward_poll = poll_data[
            poll_data["ward"] == ward
        ]

        ward_counts = (
            ward_poll["party"]
            .value_counts()
            .reindex(PARTIES, fill_value=0)
        )

        ward_total = int(ward_counts.sum())

        # Bayesian-style smoothing:
        # sparse ward polls are adjusted towards the
        # overall Chennai public poll distribution.

        prior_strength = 20

        estimated_counts = (
            ward_counts
            + prior_strength * overall_shares
        )

        estimated_shares = (
            estimated_counts / estimated_counts.sum()
        ) * 100

        historical_row = historical_data[
            historical_data["ward_no"] == ward
        ]

        if not historical_row.empty:

            historical_winner = (
                historical_row.iloc[0]["winner_party_ta"]
            )

            electors = pd.to_numeric(
                historical_row.iloc[0]["electors"],
                errors="coerce"
            )

            if pd.isna(electors):
                electors = 0

        else:

            historical_winner = "Unavailable"
            electors = 0

        projected_winner = estimated_shares.idxmax()

        result = {
            "Ward": int(ward),
            "2022 Historical Winner": historical_winner,
            "2022 Electors": int(electors),
            "Poll Responses": ward_total,
            "Projected Leading Party": projected_winner
        }

        for party in PARTIES:
            result[party + " Estimated Share (%)"] = round(
                estimated_shares[party], 2
            )

        ward_rows.append(result)

    projection = pd.DataFrame(ward_rows)

    # Approximate Chennai-wide share, weighted by
    # 2022 ward elector counts.

    weights = projection["2022 Electors"].clip(lower=0)

    if weights.sum() > 0:
        overall_vote_shares = {}

        for party in PARTIES:
            column = party + " Estimated Share (%)"

            overall_vote_shares[party] = round(
                (
                    projection[column] * weights
                ).sum() / weights.sum(),
                2
            )
    else:
        overall_vote_shares = {
            party: round(
                projection[
                    party + " Estimated Share (%)"
                ].mean(),
                2
            )
            for party in PARTIES
        }

    # Projected seats

    projected_seats = (
        projection["Projected Leading Party"]
        .value_counts()
        .reindex(PARTIES, fill_value=0)
    )

    return (
        projection,
        overall_vote_shares,
        projected_seats
    )
