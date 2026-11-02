
import pandas as pd

PARTIES = ["TVK", "DMK", "AIADMK", "NTK"]

PRIOR_STRENGTH = 20


def build_prediction_model(historical_data, poll_data):

    historical_data = historical_data.copy()
    poll_data = poll_data.copy()

    historical_data["ward_no"] = pd.to_numeric(
        historical_data["ward_no"],
        errors="coerce"
    )

    historical_data = historical_data.dropna(
        subset=["ward_no"]
    )

    historical_data["ward_no"] = (
        historical_data["ward_no"].astype(int)
    )

    wards = sorted(
        historical_data["ward_no"].unique()
    )

    if not wards:
        return None, None, None

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
        & poll_data["ward"].isin(wards)
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

        estimated_counts = (
            ward_counts
            + PRIOR_STRENGTH * overall_shares
        )

        estimated_shares = (
            estimated_counts / estimated_counts.sum()
        ) * 100

        historical_row = historical_data[
            historical_data["ward_no"] == ward
        ]

        if not historical_row.empty:

            historical_row = historical_row.iloc[0]

            historical_winner = historical_row.get(
                "winner_party_ta", "Unavailable"
            )

            electors = pd.to_numeric(
                historical_row.get("electors", 0),
                errors="coerce"
            )

            if pd.isna(electors):
                electors = 0

        else:
            historical_winner = "Unavailable"
            electors = 0

        projected_winner = estimated_shares.idxmax()

        if ward_total > 0:
            projection_basis = (
                "Ward poll + Chennai-wide smoothing"
            )
            poll_coverage = "Poll responses available"
        else:
            projection_basis = (
                "Chennai-wide poll prior only"
            )
            poll_coverage = "No ward poll responses"

        result = {
            "Ward": int(ward),
            "2022 Historical Winner": historical_winner,
            "2022 Electors": int(electors),
            "Poll Responses": ward_total,
            "Poll Coverage": poll_coverage,
            "Projection Basis": projection_basis,
            "Projected Leading Party": projected_winner
        }

        for party in PARTIES:
            result[party + " Estimated Share (%)"] = round(
                float(estimated_shares[party]), 2
            )

        ward_rows.append(result)

    projection = pd.DataFrame(ward_rows)

    # Chennai-wide elector-weighted estimated shares

    weights = pd.to_numeric(
        projection["2022 Electors"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    overall_vote_shares = {}

    if weights.sum() > 0:

        for party in PARTIES:

            column = party + " Estimated Share (%)"

            overall_vote_shares[party] = round(
                float(
                    (projection[column] * weights).sum()
                    / weights.sum()
                ),
                2
            )

    else:

        for party in PARTIES:

            column = party + " Estimated Share (%)"

            overall_vote_shares[party] = round(
                float(projection[column].mean()),
                2
            )

    # Projected seats across all available wards

    projected_seats = (
        projection["Projected Leading Party"]
        .value_counts()
        .reindex(PARTIES, fill_value=0)
    )

    projected_seats = projected_seats.astype(int)

    return (
        projection,
        overall_vote_shares,
        projected_seats
    )
