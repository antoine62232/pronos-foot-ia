# ============================================================
# FICHIER : src/diagnostic_cdm2026.py
# ROLE    : Verifier le contenu du fichier des 72 matchs
# ============================================================

import pandas as pd

df = pd.read_csv("data/matchs_a_predire.csv")
print("=" * 60)
print("DIAGNOSTIC - Fichier matchs_a_predire.csv")
print("=" * 60)

print(f"\nNombre de matchs : {len(df)}")
print(f"\nColonnes disponibles :")
print(df.columns.tolist())

print(f"\n5 premieres lignes :")
print(df.head())

print(f"\n5 dernieres lignes :")
print(df.tail())

# Verification : combien d'equipes uniques ?
equipes_dom = df['home_team'].unique()
equipes_ext = df['away_team'].unique()
toutes_equipes = set(list(equipes_dom) + list(equipes_ext))
print(f"\nNombre d'equipes uniques : {len(toutes_equipes)}")

# Si la colonne 'group' existe
if 'group' in df.columns:
    print(f"\nGroupes presents : {df['group'].unique()}")
else:
    print(f"\n[!!!] La colonne 'group' n'existe pas dans le fichier")

# Si la colonne 'date' existe
if 'date' in df.columns:
    print(f"\nPlage de dates : {df['date'].min()} a {df['date'].max()}")