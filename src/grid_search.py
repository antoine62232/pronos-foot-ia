# ============================================================
# ROLE    : Trouver les meilleurs hyperparametres XGBoost
# DUREE   : 10-20 minutes selon ta machine
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
import time


# ============================================================
# POINTS FIFA (memes que dans le backtest)
# ============================================================

POINTS_FIFA = {
    'France': 1877.32, 'Spain': 1876.40, 'Argentina': 1874.81,
    'England': 1825.97, 'Portugal': 1763.83, 'Brazil': 1761.16,
    'Netherlands': 1757.87, 'Morocco': 1755.87, 'Belgium': 1734.71,
    'Germany': 1730.37, 'Croatia': 1717.07, 'Italy': 1700.37,
    'Colombia': 1693.09, 'Senegal': 1688.99, 'Mexico': 1681.03,
    'United States': 1673.13, 'Uruguay': 1673.07, 'Japan': 1660.43,
    'Switzerland': 1649.40, 'Denmark': 1620.81, 'Iran': 1615.30,
    'Turkey': 1599.04, 'Ecuador': 1594.78, 'Austria': 1593.45,
    'South Korea': 1588.66, 'Nigeria': 1585.09, 'Australia': 1580.67,
    'Algeria': 1564.26, 'Egypt': 1563.24, 'Canada': 1556.48,
    'Norway': 1550.94, 'Ukraine': 1546.88, 'Panama': 1540.64,
    'Ivory Coast': 1532.98, 'Poland': 1528.00, 'Sweden': 1514.77,
    'Serbia': 1508.65, 'Paraguay': 1503.50, 'Czech Republic': 1501.38,
    'Hungary': 1500.58, 'Scotland': 1498.35, 'Tunisia': 1483.05,
    'Cameroon': 1481.24, 'DR Congo': 1478.35, 'Greece': 1475.82,
    'Slovakia': 1473.94, 'Venezuela': 1468.05, 'Uzbekistan': 1465.34,
    'Costa Rica': 1459.90, 'Mali': 1459.13, 'Peru': 1455.87,
    'Chile': 1455.28, 'Qatar': 1454.96, 'Romania': 1451.16,
    'Iraq': 1447.14, 'Slovenia': 1446.44, 'South Africa': 1429.73,
    'Saudi Arabia': 1421.43, 'Burkina Faso': 1412.49, 'Jordan': 1391.45,
    'Albania': 1388.06, 'Bosnia and Herzegovina': 1385.84,
    'Honduras': 1380.27, 'Wales': 1370.00, 'Cape Verde': 1366.13,
    'Jamaica': 1358.00, 'Georgia': 1350.18, 'Finland': 1346.41,
    'Ghana': 1346.31, 'Iceland': 1345.07, 'Bolivia': 1329.42,
    'Kosovo': 1318.83, 'Guinea': 1300.01, 'Montenegro': 1295.52,
}


# ============================================================
# CHARGEMENT ET PREPARATION DES DONNEES
# ============================================================

print("=" * 60)
print("GRID SEARCH - Recherche des meilleurs hyperparametres")
print("=" * 60)

print("\n[1/4] Chargement des donnees...")
df = pd.read_csv("data/matchs_entrainement.csv")
df['date'] = pd.to_datetime(df['date'])
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

# Limite jusqu'a fin 2022 (on garde 2023-2025 pour test futur si besoin)
df = df[df['date'] < '2023-01-01'].copy()
df = df.sort_values(by='date').reset_index(drop=True)

print(f"      {len(df)} matchs disponibles pour le grid search")


# ============================================================
# CALCUL DES FEATURES (idem que backtest)
# ============================================================

print("[2/4] Calcul des features (V2 - 11 features)...")

# Forme sur 5 matchs
df['forme_attaque_domicile'] = df.groupby('home_team')['home_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)
df['forme_attaque_exterieur'] = df.groupby('away_team')['away_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)
df['forme_defense_domicile'] = df.groupby('home_team')['away_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)
df['forme_defense_exterieur'] = df.groupby('away_team')['home_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)

# Forme sur 10 matchs (V2)
df['forme_attaque_domicile_10'] = df.groupby('home_team')['home_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)
df['forme_attaque_exterieur_10'] = df.groupby('away_team')['away_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)
df['forme_defense_domicile_10'] = df.groupby('home_team')['away_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)
df['forme_defense_exterieur_10'] = df.groupby('away_team')['home_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)

# Points FIFA et contexte
df['points_fifa_domicile'] = df['home_team'].map(POINTS_FIFA).fillna(1200)
df['points_fifa_exterieur'] = df['away_team'].map(POINTS_FIFA).fillna(1200)
df['match_neutre'] = df['neutral'].astype(int)


# Resultat
def determiner_resultat(row):
    if row['home_score'] > row['away_score']:
        return "1"
    elif row['home_score'] < row['away_score']:
        return "2"
    else:
        return "N"


df['resultat'] = df.apply(determiner_resultat, axis=1)
df = df.fillna(0)


# ============================================================
# PREPARATION DES DONNEES POUR LE MODELE
# ============================================================

features = [
    'forme_attaque_domicile', 'forme_attaque_exterieur',
    'forme_defense_domicile', 'forme_defense_exterieur',
    'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
    'forme_defense_domicile_10', 'forme_defense_exterieur_10',
    'points_fifa_domicile', 'points_fifa_exterieur',
    'match_neutre'
]

X = df[features]
y = df['resultat']

# Encoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Reequilibrage
poids = compute_sample_weight(class_weight='balanced', y=y_encoded)


# ============================================================
# GRILLE DE PARAMETRES A TESTER
# ============================================================

print("[3/4] Lancement du Grid Search...")
print("      Ca peut prendre 10-20 minutes, sois patient.")

# La GRILLE : on teste toutes les combinaisons de ces valeurs
grille_params = {
    'n_estimators':  [100, 200, 300, 400],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
}

# Calcul du nombre total de combinaisons
nb_combinaisons = (
    len(grille_params['n_estimators']) *
    len(grille_params['max_depth']) *
    len(grille_params['learning_rate'])
)
print(f"      Nombre de combinaisons a tester : {nb_combinaisons}")

# Modele de base
modele_base = XGBClassifier(
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0  # Pas de logs XGBoost (sinon c'est l'enfer)
)

# Cross-validation temporelle (3 splits)
# On respecte l'ordre des dates pour eviter le data leakage
tscv = TimeSeriesSplit(n_splits=3)

# Grid Search
grid_search = GridSearchCV(
    estimator=modele_base,
    param_grid=grille_params,
    cv=tscv,                  # Cross-validation temporelle
    scoring='accuracy',       # Metrique : precision
    n_jobs=-1,                # Utilise tous les coeurs CPU dispos
    verbose=2                 # Affiche la progression
)

# Lancement
debut = time.time()
grid_search.fit(X, y_encoded, sample_weight=poids)
duree = time.time() - debut


# ============================================================
# RESULTATS
# ============================================================

print(f"\n[4/4] Grid Search termine en {duree:.0f} secondes\n")

print("=" * 60)
print("RESULTATS DU GRID SEARCH")
print("=" * 60)

print(f"\nMeilleurs parametres :")
for param, valeur in grid_search.best_params_.items():
    print(f"   {param:20} : {valeur}")

print(f"\nMeilleure precision (CV) : {grid_search.best_score_ * 100:.2f}%")

print("\nTop 5 des combinaisons testees :")
resultats = pd.DataFrame(grid_search.cv_results_)
resultats = resultats.sort_values('mean_test_score', ascending=False).head(5)

for idx, row in resultats.iterrows():
    print(f"   Score: {row['mean_test_score']*100:.2f}% | "
          f"n_est: {row['param_n_estimators']:3} | "
          f"depth: {row['param_max_depth']} | "
          f"lr: {row['param_learning_rate']}")

print("\n" + "=" * 60)
print("PROCHAINE ETAPE")
print("=" * 60)
print("\nCopie ces parametres dans src/backtest.py et relance pour")
print("voir le gain sur les CdM 2018 et 2022.")