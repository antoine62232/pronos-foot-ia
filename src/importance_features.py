"""Affiche l'importance reelle des features du modele de production."""

import pandas as pd
from joblib import load

# On charge le modele entraine
modele = load("models/modele_football.pkl")

# On recupere les noms de features DIRECTEMENT depuis le modele.
# C'est la methode la plus fiable : impossible de se tromper d'ordre.
try:
    features = list(modele.feature_names_in_)
except AttributeError:
    features = list(modele.get_booster().feature_names)

# XGBoost donne un score d'importance par feature ; on le passe en pourcentage
importances = modele.feature_importances_
df = pd.DataFrame({"feature": features, "importance": importances})
df["pourcentage"] = (df["importance"] / df["importance"].sum() * 100).round(1)
df = df.sort_values("pourcentage", ascending=False)

print(df.to_string(index=False))