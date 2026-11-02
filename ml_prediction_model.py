
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
        subset=[
            "electors",
            "candidates",
            "winner_party_ta"
        ]
    )

    data = data[
        data["winner_party_ta"].isin(
            ["தி.மு.க", "அ.இ.அ.தி.மு.க", "நா.த.க."]
        )
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

    prediction = model.predict(
        pd.DataFrame([{
            "electors": electors,
            "candidates": candidates
        }])
    )

    return prediction[0]
