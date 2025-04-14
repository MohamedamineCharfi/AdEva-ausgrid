# === Importation des bibliothèques ===
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === Chargement et prétraitement des données ===
df = pd.read_csv('../data/preprocessed/dataframe_Preprocessed.csv', parse_dates=['date'])
df.sort_values(['Customer', 'date'], inplace=True)

# Pivot pour structurer les données
pivot_df = df.pivot(index='date', columns='Customer', values='consumption_daily_normalized')
pivot_df = pivot_df.interpolate(method='time').fillna(method='bfill')

# Normalisation
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(pivot_df)

# Features calendaires
calendar_df = df.drop_duplicates('date')[['date', 'is_holiday_or_weekend', 'saison']]
encoder = OneHotEncoder()
calendar_features = encoder.fit_transform(calendar_df[['is_holiday_or_weekend', 'saison']]).toarray()

# Concaténation des données
final_features = np.hstack((scaled_data, calendar_features))

# Paramètres du modèle
look_back = 30  # période d'observation plus longue
forecast_horizon = 7
X, y = [], []

# Préparer les séquences
for i in range(len(final_features) - look_back - forecast_horizon):
    X.append(final_features[i:(i + look_back)])
    y.append(scaled_data[(i + look_back):(i + look_back + forecast_horizon)])

X, y = np.array(X), np.array(y)

# Séparation Train/Test
split_idx = int(len(X) - (90 - forecast_horizon))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# === Définition du modèle amélioré ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(look_back, X.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(y.shape[1] * y.shape[2])
])

# Compilation
model.compile(optimizer='adam', loss='mse')

# Callback EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entraînement du modèle
history = model.fit(
    X_train, y_train.reshape(y_train.shape[0], -1),
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# === Évaluation et visualisation des résultats ===
predictions = model.predict(X_test).reshape(y_test.shape)

# Dénormalisation
predictions_inv = scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

# Métrique MAAPE
def MAAPE(actual, predicted):
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + 1e-10))))

maape_score = MAAPE(y_test_inv, predictions_inv)
print(f"MAAPE améliorée : {maape_score:.4f}")

# Exemple visualisation d'un client
client_index = 0  # client problématique dans l'image
plt.figure(figsize=(12, 6))
dates_test = pivot_df.index[-len(y_test):]

plt.plot(dates_test[:forecast_horizon], y_test_inv[0, :, client_index], label='Réel', marker='o')
plt.plot(dates_test[:forecast_horizon], predictions_inv[0, :, client_index], label='Prédit', marker='x')

plt.xlabel('Date')
plt.ylabel('Consommation d\'énergie')
plt.title(f'Prédiction améliorée pour Client {client_index} (7 jours)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

