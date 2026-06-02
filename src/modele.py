"""Entrainement du modele de production (forme + Elo) et sauvegarde."""

import os
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from elo import calculer_elo, elo_final


def determiner_resultat(ligne):
    if ligne["home_score"] > ligne["away_score"]:
        return "1"
    if ligne["home_score"] == ligne["away_score"]:
        return "N"
    return "2"


# 1. Chargement + Elo + features de forme
print("Chargement des donnees...")
df = pd.read_csv("data/matchs_entrainement.csv")
df["date"] = pd.to_datetime(df["date"])
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df = calculer_elo(df)

for n in (5, 10):
    suf = "" if n == 5 else "_10"
    df[f"forme_attaque_domicile{suf}"] = df.groupby("home_team")["home_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_attaque_exterieur{suf}"] = df.groupby("away_team")["away_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_defense_domicile{suf}"] = df.groupby("home_team")["away_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))
    df[f"forme_defense_exterieur{suf}"] = df.groupby("away_team")["home_score"].transform(lambda x: x.rolling(n, min_periods=1).mean().shift(1))

df["match_neutre"] = df["neutral"].astype(int)
df["resultat"] = df.apply(determiner_resultat, axis=1)
df = df.fillna(0)
print(f"{df.shape[0]} matchs prets.")

# 2. Features de production : forme (5 et 10 matchs) + Elo
features = [
    "forme_attaque_domicile", "forme_attaque_exterieur",
    "forme_defense_domicile", "forme_defense_exterieur",
    "forme_attaque_domicile_10", "forme_attaque_exterieur_10",
    "forme_defense_domicile_10", "forme_defense_exterieur_10",
    "elo_domicile", "elo_exterieur",
    "match_neutre",
]

X = df[features]
y = df["resultat"]

# 3. Encodage des resultats + reequilibrage des classes
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
poids = compute_sample_weight(class_weight="balanced", y=y_encoded)

# 4. Entrainement sur TOUTES les donnees (pour la prod, on exploite tout l'historique).
print("Entrainement du modele (forme + Elo)...")
modele = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
modele.fit(X, y_encoded, sample_weight=poids)

# 4bis. Export du classement Elo courant (lu par l'app au moment de predire)
elos = elo_final(df)
pd.DataFrame(sorted(elos.items(), key=lambda x: -x[1]), columns=["equipe", "elo"]).to_csv("data/elo_actuel.csv", index=False)
print(f"Classement Elo courant sauvegarde ({len(elos)} equipes) dans data/elo_actuel.csv.")

# 5. Sauvegarde (regenere aussi label_encoder.pkl, ce qui corrige le fichier corrompu)
os.makedirs("models", exist_ok=True)
dump(modele, "models/modele_football.pkl")
dump(encoder, "models/label_encoder.pkl")
print("Modele et encoder sauvegardes dans models/.")
print("Features attendues par le modele :", list(modele.feature_names_in_))