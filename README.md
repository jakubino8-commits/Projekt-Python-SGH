# Jakub Kukieła 121813

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Generowanie danych wyjściowych

def generate_data():
    np.random.seed(42)

    timestamps = pd.date_range(
        start="2022-08-09 10:00",
        periods=120,
        freq="120min"
    )

    df = pd.DataFrame({"timestamp": timestamps})

# Konstrukcja zmiennych czasowych

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

# Konstrukcja szeregu energii

    base = (
        2.5 * np.exp(-((df["hour"] - 12) ** 2) / 20)
        + 0.1 * (df["dayofweek"] < 5)
    )

    df["energy"] = base + np.random.normal(0, 0.05, len(df))

# Konstrukcja głównej zmiennej
    df["energy_next_120"] = df["energy"].shift(-1)

 # Utworzenie opóźnień
    df["lag_15"] = df["energy"].shift(1)
    df["lag_90"] = df["energy"].shift(2)
    df["lag_75"] = df["energy"].shift(3)

    df = df.dropna().reset_index(drop=True)
    return df

# FUNKCJA RYSUJĄCA WYKRES

def plot_forecast_like_example(
    data,
    cut_of_date,
    history_len=30,
    forecast_len=20
):
    cut_of_date = pd.to_datetime(cut_of_date)

    train = data[data["timestamp"] < cut_of_date]
    test = data[data["timestamp"] >= cut_of_date].iloc[:forecast_len]

    history = train.iloc[-history_len:]

    models = {
        "LR | LAG_90+TIME": ["lag_90", "hour"],
        "LR | LAG_75+TIME": ["lag_75", "hour"],
        "LR | LAG_15+TIME": ["lag_15", "hour"]
    }

    plt.figure(figsize=(14, 6))

    # Historia
    plt.plot(
        history["timestamp"],
        history["energy_next_120"],
        color="gray",
        linewidth=2,
        label="Historia (energy_next_120)"
    )

    # Rzeczywiste wartości po cut_of_date
    plt.plot(
        test["timestamp"],
        test["energy_next_120"],
        linestyle="--",
        color="black",
        label="Rzeczywiste (target)"
    )

    # Modele
    for name, features in models.items():

        X_train = train[features]
        y_train = train["energy_next_120"]

        X_test = test[features]
        y_test = test["energy_next_120"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        plt.plot(
            test["timestamp"],
            y_pred,
            label=f"{name} (R2={r2:.3f})"
        )

    # Oznaczenie daty odcięcia
    plt.axvline(
        cut_of_date,
        color="red",
        linestyle=":",
        linewidth=2,
        label="cut_of_date"
    )

    plt.title("Prognoza zużycia energii: następne 120 min | LR")
    plt.xlabel("timestamp")
    plt.ylabel("energy_next_120")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Uruchomienie funkcji

if __name__ == "__main__":

    df = generate_data()

    plot_forecast_like_example(
        data=df,
        cut_of_date="2022-08-12 00:00",
        history_len=30,
        forecast_len=20
    )
