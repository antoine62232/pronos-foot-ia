# ============================================================
# ROLE    : Identifier quelles features apportent reellement
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


# ============================================================
# RECUPERER LE TRAINING DEJA PREPARE PAR LE BACKTEST V3
# ============================================================

# On va reutiliser la logique du backtest V3 pour preparer les donnees
# avec les 21 features

print("=" * 60)
print("ANALYSE D'IMPORTANCE DES FEATURES")
print("=" * 60)

# Import des fonctions du backtest V3
import sys
sys.path.append('src')
from backtest_v3 import (
    POINTS_FIFA, CONFEDERATION,
    determiner_resultat, calculer_features_v3_globales
)

print("\n[1/4] Chargement des donnees...")
df = pd.read_csv("data/matchs_entrainement.csv")
df['date'] = pd.to_datetime(df['date'])
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

# Limite jusqu'a fin 2022
df = df[df['date'] < '2023-01-01'].copy()
df = df.sort_values(by='date').reset_index(drop=True)
print(f"      {len(df)} matchs disponibles")


print("\n[2/4] Calcul des features V2 et V3...")

# Forme V2
for n in [5, 10]:
    suffix = '' if n == 5 else '_10'
    df[f'forme_attaque_domicile{suffix}'] = df.groupby('home_team')['home_score'].transform(
        lambda x: x.rolling(n, min_periods=1).mean().shift(1)
    )
    df[f'forme_attaque_exterieur{suffix}'] = df.groupby('away_team')['away_score'].transform(
        lambda x: x.rolling(n, min_periods=1).mean().shift(1)
    )
    df[f'forme_defense_domicile{suffix}'] = df.groupby('home_team')['away_score'].transform(
        lambda x: x.rolling(n, min_periods=1).mean().shift(1)
    )
    df[f'forme_defense_exterieur{suffix}'] = df.groupby('away_team')['home_score'].transform(
        lambda x: x.rolling(n, min_periods=1).mean().shift(1)
    )

df['points_fifa_domicile'] = df['home_team'].map(POINTS_FIFA).fillna(1200)
df['points_fifa_exterieur'] = df['away_team'].map(POINTS_FIFA).fillna(1200)
df['match_neutre'] = df['neutral'].astype(int)

# Features V3
print("      Calcul des features V3 (vectorisees)...")
df_features_v3 = calculer_features_v3_globales(df)

df_v3_dom = df_features_v3.rename(columns={
    'equipe': 'home_team',
    'diff_buts_10': 'diff_buts_dom',
    'clean_sheets_10': 'clean_sheets_dom',
    'streak_W': 'streak_victoires_dom',
    'streak_L': 'streak_defaites_dom',
})
df_v3_ext = df_features_v3.rename(columns={
    'equipe': 'away_team',
    'diff_buts_10': 'diff_buts_ext',
    'clean_sheets_10': 'clean_sheets_ext',
    'streak_W': 'streak_victoires_ext',
    'streak_L': 'streak_defaites_ext',
})

df = df.merge(df_v3_dom, on=['date', 'home_team'], how='left')
df = df.merge(df_v3_ext, on=['date', 'away_team'], how='left')

df['conf_dom'] = df['home_team'].map(CONFEDERATION).fillna(0)
df['conf_ext'] = df['away_team'].map(CONFEDERATION).fillna(0)

df['resultat'] = df.apply(
    lambda row: determiner_resultat(row['home_score'], row['away_score']),
    axis=1
)
df = df.fillna(0)


# ============================================================
# ENTRAINEMENT DU MODELE ET CALCUL DES IMPORTANCES
# ============================================================

print("\n[3/4] Entrainement du modele pour analyse...")

features = [
    'forme_attaque_domicile', 'forme_attaque_exterieur',
    'forme_defense_domicile', 'forme_defense_exterieur',
    'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
    'forme_defense_domicile_10', 'forme_defense_exterieur_10',
    'points_fifa_domicile', 'points_fifa_exterieur',
    'match_neutre',
    'streak_victoires_dom', 'streak_victoires_ext',
    'streak_defaites_dom', 'streak_defaites_ext',
    'diff_buts_dom', 'diff_buts_ext',
    'clean_sheets_dom', 'clean_sheets_ext',
    'conf_dom', 'conf_ext',
]

X = df[features]
y = df['resultat']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
poids = compute_sample_weight(class_weight='balanced', y=y_encoded)

modele = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
modele.fit(X, y_encoded, sample_weight=poids)


# ============================================================
# ANALYSE
# ============================================================

print("\n[4/4] Analyse des importances...\n")

# Importances XGBoost
importances = pd.DataFrame({
    'feature': features,
    'importance': modele.feature_importances_
})
importances = importances.sort_values('importance', ascending=False)

print("=" * 60)
print("CLASSEMENT DES FEATURES PAR IMPORTANCE")
print("=" * 60)
print(f"\n{'Rang':<5} {'Feature':<35} {'Importance':<15}")
print("-" * 60)
for rang, (_, row) in enumerate(importances.iterrows(), 1):
    barre = "█" * int(row['importance'] * 200)
    print(f"{rang:<5} {row['feature']:<35} {row['importance']*100:>5.1f}%  {barre}")


# Identification des features peu utiles (< 3%)
print("\n" + "=" * 60)
print("FEATURES A PROBABLEMENT SUPPRIMER (importance < 3%)")
print("=" * 60)
features_inutiles = importances[importances['importance'] < 0.03]
if len(features_inutiles) > 0:
    for _, row in features_inutiles.iterrows():
        print(f"   {row['feature']:<35} ({row['importance']*100:.1f}%)")
else:
    print("   Aucune feature a supprimer !")


# Identification des top features
print("\n" + "=" * 60)
print("TOP 5 DES FEATURES LES PLUS IMPORTANTES")
print("=" * 60)
for _, row in importances.head(5).iterrows():
    print(f"   {row['feature']:<35} ({row['importance']*100:.1f}%)")


print("\n" + "=" * 60)
print("RECOMMANDATIONS")
print("=" * 60)
print("\nFeatures a CONSERVER imperativement (top 5)")
print("Features a TESTER en supprimant (< 3%) :")
print("   Si suppression -> precision pareille : ENLEVE-LES")
print("   Si suppression -> precision baisse   : GARDE-LES")