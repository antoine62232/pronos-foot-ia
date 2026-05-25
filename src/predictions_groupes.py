# ============================================================
# ROLE    : Predire le classement de chaque groupe de la CdM 2026
# VERSION : Auto-contenue (entraine le modele V2 a la volee)
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# POINTS FIFA (memes que V2)
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
    'Curaçao': 1200.00,  # Equipe absente de la liste FIFA standard
    'Haiti': 1200.00,    # Equipe absente de la liste FIFA standard
    'New Zealand': 1200.00,  # Equipe absente
}


# ============================================================
# LES 12 GROUPES DE LA CDM 2026
# ============================================================

GROUPES = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czech Republic'],
    'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
    'D': ['United States', 'Australia', 'Paraguay', 'Turkey'],
    'E': ['Germany', 'Ecuador', 'Ivory Coast', 'Curaçao'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'G': ['Belgium', 'Iran', 'Egypt', 'New Zealand'],
    'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Austria', 'Algeria', 'Jordan'],
    'K': ['Portugal', 'Colombia', 'Uzbekistan', 'DR Congo'],
    'L': ['England', 'Croatia', 'Panama', 'Ghana'],
}


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def determiner_resultat(home_score, away_score):
    """Convertit un score en resultat 1/N/2."""
    if home_score > away_score:
        return "1"
    elif home_score < away_score:
        return "2"
    else:
        return "N"


def calculer_forme(df_historique, equipe, date_match, type_calcul, nb_matchs=5):
    """Forme attaque/defense sur N derniers matchs."""
    matchs_avant = df_historique[df_historique['date'] < date_match]

    if type_calcul == 'attaque':
        m_dom = matchs_avant[matchs_avant['home_team'] == equipe]['home_score']
        m_ext = matchs_avant[matchs_avant['away_team'] == equipe]['away_score']
        valeur_defaut = 0.0
    else:
        m_dom = matchs_avant[matchs_avant['home_team'] == equipe]['away_score']
        m_ext = matchs_avant[matchs_avant['away_team'] == equipe]['home_score']
        valeur_defaut = 1.0

    tous_buts = pd.concat([m_dom, m_ext])
    if len(tous_buts) == 0:
        return valeur_defaut
    return round(tous_buts.tail(nb_matchs).mean(), 2)


# ============================================================
# ENTRAINEMENT DU MODELE V2
# ============================================================

print("=" * 60)
print("PREDICTION DES CLASSEMENTS DE GROUPES - CdM 2026")
print("=" * 60)

print("\n[1/6] Chargement des donnees d'entrainement...")
df_train = pd.read_csv("data/matchs_entrainement.csv")
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['home_score'] = df_train['home_score'].astype(int)
df_train['away_score'] = df_train['away_score'].astype(int)
df_train = df_train.sort_values(by='date').reset_index(drop=True)
print(f"      {len(df_train)} matchs charges")

print("\n[2/6] Calcul des features V2...")

# Forme sur 5 matchs
df_train['forme_attaque_domicile'] = df_train.groupby('home_team')['home_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)
df_train['forme_attaque_exterieur'] = df_train.groupby('away_team')['away_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)
df_train['forme_defense_domicile'] = df_train.groupby('home_team')['away_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)
df_train['forme_defense_exterieur'] = df_train.groupby('away_team')['home_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)

# Forme sur 10 matchs
df_train['forme_attaque_domicile_10'] = df_train.groupby('home_team')['home_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)
df_train['forme_attaque_exterieur_10'] = df_train.groupby('away_team')['away_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)
df_train['forme_defense_domicile_10'] = df_train.groupby('home_team')['away_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)
df_train['forme_defense_exterieur_10'] = df_train.groupby('away_team')['home_score'].transform(
    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
)

df_train['points_fifa_domicile'] = df_train['home_team'].map(POINTS_FIFA).fillna(1200)
df_train['points_fifa_exterieur'] = df_train['away_team'].map(POINTS_FIFA).fillna(1200)
df_train['match_neutre'] = df_train['neutral'].astype(int)

# Resultat
df_train['resultat'] = df_train.apply(
    lambda row: determiner_resultat(row['home_score'], row['away_score']),
    axis=1
)
df_train = df_train.fillna(0)

print("\n[3/6] Entrainement du modele V2...")

features = [
    'forme_attaque_domicile', 'forme_attaque_exterieur',
    'forme_defense_domicile', 'forme_defense_exterieur',
    'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
    'forme_defense_domicile_10', 'forme_defense_exterieur_10',
    'points_fifa_domicile', 'points_fifa_exterieur',
    'match_neutre'
]

X_train = df_train[features]
y_train = df_train['resultat']

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
poids = compute_sample_weight(class_weight='balanced', y=y_train_encoded)

modele = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
modele.fit(X_train, y_train_encoded, sample_weight=poids)
print(f"      Modele V2 entraine sur {len(X_train)} matchs ({len(features)} features)")


# ============================================================
# PREDICTION DES MATCHS DE LA CDM 2026
# ============================================================

print("\n[4/6] Prediction des 72 matchs de la CdM 2026...")

df_matchs = pd.read_csv("data/matchs_a_predire.csv")
df_matchs['date'] = pd.to_datetime(df_matchs['date'])

predictions = []

for _, match in df_matchs.iterrows():
    eq_dom = match['home_team']
    eq_ext = match['away_team']
    date_m = match['date']

    # Calcul des features pour ce match
    forme_att_dom = calculer_forme(df_train, eq_dom, date_m, 'attaque', nb_matchs=5)
    forme_att_ext = calculer_forme(df_train, eq_ext, date_m, 'attaque', nb_matchs=5)
    forme_def_dom = calculer_forme(df_train, eq_dom, date_m, 'defense', nb_matchs=5)
    forme_def_ext = calculer_forme(df_train, eq_ext, date_m, 'defense', nb_matchs=5)
    forme_att_dom_10 = calculer_forme(df_train, eq_dom, date_m, 'attaque', nb_matchs=10)
    forme_att_ext_10 = calculer_forme(df_train, eq_ext, date_m, 'attaque', nb_matchs=10)
    forme_def_dom_10 = calculer_forme(df_train, eq_dom, date_m, 'defense', nb_matchs=10)
    forme_def_ext_10 = calculer_forme(df_train, eq_ext, date_m, 'defense', nb_matchs=10)
    fifa_dom = POINTS_FIFA.get(eq_dom, 1200)
    fifa_ext = POINTS_FIFA.get(eq_ext, 1200)
    match_neutre = int(match['neutral'])

    donnees_match = pd.DataFrame([[
        forme_att_dom, forme_att_ext,
        forme_def_dom, forme_def_ext,
        forme_att_dom_10, forme_att_ext_10,
        forme_def_dom_10, forme_def_ext_10,
        fifa_dom, fifa_ext,
        match_neutre,
    ]], columns=features)

    proba = modele.predict_proba(donnees_match)[0]
    classes = encoder.inverse_transform(modele.classes_)
    probas_dict = dict(zip(classes, proba))
    pronostic = max(probas_dict, key=probas_dict.get)

    predictions.append({
        'pronostic': pronostic,
        'proba_1': probas_dict.get('1', 0),
        'proba_N': probas_dict.get('N', 0),
        'proba_2': probas_dict.get('2', 0),
    })

df_matchs['pronostic'] = [p['pronostic'] for p in predictions]
df_matchs['proba_1'] = [p['proba_1'] for p in predictions]
df_matchs['proba_N'] = [p['proba_N'] for p in predictions]
df_matchs['proba_2'] = [p['proba_2'] for p in predictions]

print(f"      [OK] {df_matchs['pronostic'].notna().sum()}/72 matchs predits")


# ============================================================
# CALCUL DES CLASSEMENTS DE GROUPES
# ============================================================

print("\n[5/6] Calcul des classements par groupe...")


def calculer_classement_groupe(df_groupe, equipes_groupe):
    """Calcule le classement d'un groupe a partir des predictions."""
    classement = {}
    for equipe in equipes_groupe:
        classement[equipe] = {
            'equipe': equipe,
            'points': 0,
            'victoires': 0,
            'nuls': 0,
            'defaites': 0,
            'force_predite': 0.0,
        }

    for _, match in df_groupe.iterrows():
        eq_dom = match['home_team']
        eq_ext = match['away_team']
        pronostic = match['pronostic']

        if pronostic == '1':
            classement[eq_dom]['points'] += 3
            classement[eq_dom]['victoires'] += 1
            classement[eq_ext]['defaites'] += 1
        elif pronostic == '2':
            classement[eq_ext]['points'] += 3
            classement[eq_ext]['victoires'] += 1
            classement[eq_dom]['defaites'] += 1
        elif pronostic == 'N':
            classement[eq_dom]['points'] += 1
            classement[eq_ext]['points'] += 1
            classement[eq_dom]['nuls'] += 1
            classement[eq_ext]['nuls'] += 1

        # Force predite : somme des probas de victoire (critere de departage)
        classement[eq_dom]['force_predite'] += match['proba_1']
        classement[eq_ext]['force_predite'] += match['proba_2']

    df_classement = pd.DataFrame(classement.values())
    df_classement = df_classement.sort_values(
        by=['points', 'force_predite'],
        ascending=[False, False]
    ).reset_index(drop=True)
    df_classement['position'] = range(1, len(df_classement) + 1)

    return df_classement


classements = {}
for lettre, equipes in GROUPES.items():
    df_groupe = df_matchs[df_matchs['group'] == lettre]
    classements[lettre] = calculer_classement_groupe(df_groupe, equipes)


# ============================================================
# AFFICHAGE DES CLASSEMENTS
# ============================================================

print("\n[6/6] CLASSEMENTS PAR GROUPE :")

for lettre in sorted(classements.keys()):
    print(f"\n{'=' * 60}")
    print(f"GROUPE {lettre}")
    print(f"{'=' * 60}")
    df_c = classements[lettre]
    print(f"{'Pos':<5}{'Equipe':<25}{'Pts':<6}{'V':<4}{'N':<4}{'D':<4}{'Force':<8}")
    print("-" * 60)
    for _, row in df_c.iterrows():
        statut = ""
        if row['position'] == 1:
            statut = "  <-- 1er (qualifie)"
        elif row['position'] == 2:
            statut = "  <-- 2eme (qualifie)"
        elif row['position'] == 3:
            statut = "  <-- 3eme (a evaluer)"
        elif row['position'] == 4:
            statut = "  <-- elimine"
        print(f"{row['position']:<5}{row['equipe']:<25}{row['points']:<6}"
              f"{row['victoires']:<4}{row['nuls']:<4}{row['defaites']:<4}"
              f"{row['force_predite']:<8.2f}{statut}")


# ============================================================
# IDENTIFICATION DES QUALIFIES
# ============================================================

print("\n\n" + "=" * 60)
print("CLASSEMENT DES TROISIEMES (8 meilleurs qualifies)")
print("=" * 60)

troisiemes = []
for lettre, df_c in classements.items():
    troisiemes.append({
        'groupe': lettre,
        'equipe': df_c.iloc[2]['equipe'],
        'points': df_c.iloc[2]['points'],
        'force': df_c.iloc[2]['force_predite'],
    })

df_troisiemes = pd.DataFrame(troisiemes)
df_troisiemes = df_troisiemes.sort_values(
    by=['points', 'force'],
    ascending=[False, False]
).reset_index(drop=True)

print(f"\n{'Rang':<6}{'Groupe':<8}{'Equipe':<25}{'Pts':<6}{'Force':<8}")
print("-" * 55)
for idx, row in df_troisiemes.iterrows():
    rang = idx + 1
    statut = "  <-- QUALIFIE" if rang <= 8 else "  elimine"
    print(f"{rang:<6}{row['groupe']:<8}{row['equipe']:<25}{row['points']:<6}"
          f"{row['force']:<8.2f}{statut}")


# ============================================================
# SAUVEGARDE
# ============================================================

print("\n\n[Sauvegarde] Resultats...")

df_matchs.to_csv("data/predictions_groupes.csv", index=False)
print("      Predictions sauvegardees : data/predictions_groupes.csv")

# Liste finale des 32 qualifies
qualifies_finaux = []
for lettre, df_c in classements.items():
    qualifies_finaux.append({
        'qualification': f"1er Groupe {lettre}",
        'equipe': df_c.iloc[0]['equipe'],
        'groupe': lettre,
        'mode': 'direct'
    })
    qualifies_finaux.append({
        'qualification': f"2eme Groupe {lettre}",
        'equipe': df_c.iloc[1]['equipe'],
        'groupe': lettre,
        'mode': 'direct'
    })
for idx, row in df_troisiemes.head(8).iterrows():
    qualifies_finaux.append({
        'qualification': f"3eme Groupe {row['groupe']} (repechage)",
        'equipe': row['equipe'],
        'groupe': row['groupe'],
        'mode': 'repechage'
    })

df_qualifies = pd.DataFrame(qualifies_finaux)
df_qualifies.to_csv("data/qualifies_1_16.csv", index=False)
print("      Liste des qualifies sauvegardee : data/qualifies_1_16.csv")

print(f"\n>>> {len(qualifies_finaux)} equipes qualifiees pour les 1/16 de finale")