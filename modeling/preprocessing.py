# === Standard Library ===
import math
import time
from datetime import datetime
from calendar import day_abbr, month_abbr
import itertools
import glob

# === Data Handling ===
import pandas as pd
import numpy as np

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# === Machine Learning & Preprocessing ===
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# === External Libraries ===
import requests
#https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
pd.options.display.max_rows = 999

# Récupérer la liste de tous les fichiers CSV dans le répertoire
file_paths = glob.glob('../data/raw1/*.csv')

# Initialiser une liste pour stocker les DataFrames individuels
dfs = []

# Lire chaque fichier CSV, utiliser la deuxième ligne comme noms de colonnes et supprimer la première ligne
for file_path in file_paths:
    df = pd.read_csv(file_path, header=1)  # Utiliser la deuxième ligne comme noms de colonnes
    dfs.append(df)

# Concaténer tous les DataFrames dans un seul DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
column_names = combined_df.columns
print(column_names)
# Afficher les valeurs uniques de la colonne 'Row Quality'
print('Valeurs possibles de Row Quality a priori tous NAN',combined_df['Row Quality'].unique())
combined_df = combined_df.drop(columns=['Row Quality'])
# Afficher le DataFrame combiné
print(combined_df.head(20))

if combined_df.isna().any().any():
    print("La DataFrame contient des valeurs NaN")
else:
    print("La DataFrame ne contient pas de valeurs NaN")

# Convertir l'index en type temporel 
combined_df.index = pd.to_datetime(combined_df.index)


## format the dataset :

import pandas as pd
from datetime import datetime
saison=[]
consumption = []
date_ = []  # Ajouter la colonne 'date'
Customer=[]
Postcode=[]



# Loop through each row of the DataFrame
for index, row in combined_df.iterrows():
    # Filtrer les lignes où 'consumtion catagory' est égale à 'GC'
    if row['Consumption Category'] != 'GC':
        continue

    # Extract the date string and convert it to a datetime object
    date_str = row['date']
    cust= row['Customer']
    code= row['Postcode']

    try:
        # Essayez de convertir la date avec le format '%d/%m/%Y'
        dt = datetime.strptime(date_str, '%d/%m/%Y')
    except ValueError:
        try:
            # Si la conversion échoue, essayez de reformater la date au format '%d-%b-%y'
            date_str = datetime.strptime(date_str, '%d-%b-%y').strftime('%d/%m/%Y')
            # Essayez maintenant de convertir la date avec le format '%d/%m/%Y'
            dt = datetime.strptime(date_str, '%d/%m/%Y')
        except ValueError:
            # Si les deux formats échouent, imprimez un message d'erreur
            print("Format de date non reconnu:", date_str)
            continue  # Sort de la boucle pour passer à la prochaine ligne

    # Loop through each time slot in the row and extract the required values
    for time_slot, cons in row.items():
        if time_slot in ['Customer', 'Generator Capacity', 'Postcode', 'Consumption Category', 'date']:
            continue  # Skip non-data columns
        # Append the extracted values to their respective lists
        Customer.append(cust)
        Postcode.append(code)
        consumption.append(float(cons))
        date_.append(dt)  # Ajouter la date extraite à la liste
        
output_df = pd.DataFrame({
    'date': date_,  # Ajouter la colonne 'date'
    'consumption': consumption,
    'Customer':Customer,
    'Postcode':Postcode,
})

Enc_1 = OrdinalEncoder()
# Convertir la colonne "Customer" en entiers
output_df["Customer"] = output_df["Customer"].astype(int)
output_df["Customer"] = Enc_1.fit_transform(output_df[["Customer"]]).astype('int64')
output_df[["Customer"]].value_counts()


df_grouped = output_df.groupby(['date', 'Customer','Postcode'], as_index=False).agg({
    'consumption': 'mean'}
)

# Définir le pays
country = "AU"

# Récupérer les jours fériés pour chaque année
holidays = []

for year in range(2011, 2014):
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
    response = requests.get(url)
    holidays.extend(response.json())

# Extraction des dates et conversion en datetime
holidays_dates = pd.to_datetime([holiday['date'] for holiday in holidays])
df_grouped['date'] = pd.to_datetime(df_grouped['date'])

# Créer la colonne "is_holiday_or_weekend"
df_grouped["is_holiday_or_weekend"] = 0
holiday_dates = pd.to_datetime([h["date"] for h in holidays])
df_grouped.loc[
    df_grouped["date"].isin(holiday_dates), 
    "is_holiday_or_weekend"
] = 1
df_grouped.loc[df_grouped["date"].dt.day_name().isin(["Saturday", "Sunday"]), "is_holiday_or_weekend"] = 1

df_grouped['saison'] = df_grouped['date'].dt.month % 12 // 3 + 1  # Hiver:1, Printemps:2, etc.

print(df_grouped)



#Filter the DataFrame for Customer 
customer_number = 200
filtered_df = df_grouped

scaler = MinMaxScaler()
filtered_df['consumption_daily_normalized'] = scaler.fit_transform(filtered_df[['consumption']])


a = filtered_df.set_index('date')['consumption_daily_normalized']
ax = a.plot(figsize=(15, 5))
ax.set_title('consumption_daily_normalized', fontsize='large', fontweight='bold')
ax.set_xlabel('date')
ax.set_ylabel('Consommation')
plt.grid(True)
plt.show()
    
print(filtered_df)

filtered_df.to_csv('dataframe_Preprocessed.csv', index=False)