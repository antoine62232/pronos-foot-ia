import pandas as pd

df = pd.read_csv("data/matchs_entrainement.csv")
print("Colonnes disponibles :")
print(df.columns.tolist())
print(f"\nNombre de matchs : {len(df)}")
print(f"\nApercu :")
print(df.head(3))
print(f"\nValeurs uniques de 'tournament' (top 15) :")
print(df['tournament'].value_counts().head(15))