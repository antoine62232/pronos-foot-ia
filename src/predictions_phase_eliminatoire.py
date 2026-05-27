# ============================================================
# FICHIER : src/predictions_phase_eliminatoire.py
# ROLE    : Simuler la phase eliminatoire de la CdM 2026
# VERSION : V2 - Utilise la VRAIE forme des equipes (etat juin 2026)
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# POINTS FIFA (memes que dans les autres scripts)
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
    'Curaçao': 1200.00,
    'Haiti': 1200.00,
    'New Zealand': 1200.00,
}


# ============================================================
# FONCTIONS UTILITAIRES (identiques aux autres scripts)
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
    """
    Calcule la VRAIE forme d'une equipe sur ses N derniers matchs avant date_match.
    Identique a la fonction utilisee dans predictions_groupes.py
    """
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
# ETAPE 1 : ENTRAINEMENT DU MODELE V2 A LA VOLEE
# ============================================================

print("=" * 60)
print("SIMULATION PHASE ELIMINATOIRE - Avec vraie forme")
print("=" * 60)

print("\n[1/6] Chargement des donnees d'entrainement...")
df_train = pd.read_csv("data/matchs_entrainement.csv")
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['home_score'] = df_train['home_score'].astype(int)
df_train['away_score'] = df_train['away_score'].astype(int)
df_train = df_train.sort_values(by='date').reset_index(drop=True)
print(f"      {len(df_train)} matchs charges")

print("\n[2/6] Calcul des features V2 sur le training...")

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
print(f"      Modele V2 entraine sur {len(X_train)} matchs")


# ============================================================
# ETAPE 2 : FONCTION DE PREDICTION AVEC VRAIE FORME
# ============================================================

# Date de reference pour calculer la forme : juste avant la CdM 2026
DATE_REFERENCE = pd.to_datetime('2026-06-11')


def simuler_match_couperet(equipe_A, equipe_B):
    """
    Simule un match a elimination directe avec la VRAIE forme des equipes.
    Force un vainqueur en cas de prediction de match nul.
    """
    # Calcul de la VRAIE forme de chaque equipe
    forme_att_A = calculer_forme(df_train, equipe_A, DATE_REFERENCE, 'attaque', nb_matchs=5)
    forme_att_B = calculer_forme(df_train, equipe_B, DATE_REFERENCE, 'attaque', nb_matchs=5)
    forme_def_A = calculer_forme(df_train, equipe_A, DATE_REFERENCE, 'defense', nb_matchs=5)
    forme_def_B = calculer_forme(df_train, equipe_B, DATE_REFERENCE, 'defense', nb_matchs=5)

    forme_att_A_10 = calculer_forme(df_train, equipe_A, DATE_REFERENCE, 'attaque', nb_matchs=10)
    forme_att_B_10 = calculer_forme(df_train, equipe_B, DATE_REFERENCE, 'attaque', nb_matchs=10)
    forme_def_A_10 = calculer_forme(df_train, equipe_A, DATE_REFERENCE, 'defense', nb_matchs=10)
    forme_def_B_10 = calculer_forme(df_train, equipe_B, DATE_REFERENCE, 'defense', nb_matchs=10)

    # Points FIFA
    fifa_A = POINTS_FIFA.get(equipe_A, 1200)
    fifa_B = POINTS_FIFA.get(equipe_B, 1200)

    # Construction de la ligne de donnees
    donnees_match = pd.DataFrame([[
        forme_att_A, forme_att_B,
        forme_def_A, forme_def_B,
        forme_att_A_10, forme_att_B_10,
        forme_def_A_10, forme_def_B_10,
        fifa_A, fifa_B,
        1  # Terrain neutre (phase finale = stade unique)
    ]], columns=features)

    # Prediction
    proba = modele.predict_proba(donnees_match)[0]
    classes = encoder.inverse_transform(modele.classes_)
    probas_dict = dict(zip(classes, proba))

    proba_A = probas_dict.get('1', 0)
    proba_B = probas_dict.get('2', 0)

    # Tie-break : on ignore la proba de nul et on compare A vs B
    if proba_A >= proba_B:
        confiance = (proba_A / (proba_A + proba_B)) * 100
        return equipe_A, confiance
    else:
        confiance = (proba_B / (proba_A + proba_B)) * 100
        return equipe_B, confiance


# ============================================================
# ETAPE 3 : CONSTRUCTION DU TABLEAU DES 1/16 (regles FIFA)
# ============================================================

print("\n[4/6] Chargement des qualifies et construction du tableau...")

df_qualifies = pd.read_csv("data/qualifies_1_16.csv")

# Rangement des equipes dans des dictionnaires
premiers = df_qualifies[
    df_qualifies['qualification'].str.startswith('1er')
].set_index('groupe')['equipe'].to_dict()

deuxiemes = df_qualifies[
    df_qualifies['qualification'].str.startswith('2eme')
].set_index('groupe')['equipe'].to_dict()

troisiemes = df_qualifies[df_qualifies['mode'] == 'repechage']['equipe'].tolist()

print(f"      32 equipes pretes pour le bracket")

# Tableau des 16 matchs (structure FIFA officielle)
matchs_1_16 = [
    (deuxiemes['A'], deuxiemes['B']),
    (premiers['C'], deuxiemes['F']),
    (premiers['E'], troisiemes.pop(0)),
    (premiers['F'], deuxiemes['C']),
    (deuxiemes['E'], deuxiemes['I']),
    (premiers['I'], troisiemes.pop(0)),
    (premiers['A'], troisiemes.pop(0)),
    (premiers['L'], troisiemes.pop(0)),
    (premiers['G'], troisiemes.pop(0)),
    (premiers['D'], troisiemes.pop(0)),
    (premiers['H'], deuxiemes['J']),
    (deuxiemes['K'], deuxiemes['L']),
    (premiers['B'], troisiemes.pop(0)),
    (deuxiemes['D'], deuxiemes['G']),
    (premiers['J'], deuxiemes['H']),
    (premiers['K'], troisiemes.pop(0))
]


# ============================================================
# ETAPE 4 : SIMULATION TOUR PAR TOUR
# ============================================================

print("\n[5/6] Simulation tour par tour...\n")

noms_tours = ["1/16 DE FINALE", "1/8 DE FINALE", "QUARTS DE FINALE", "DEMI-FINALES", "FINALE"]
matchs_tour_actuel = matchs_1_16
champion_final = ""

# Pour sauvegarder tous les resultats
historique_complet = []

for tour in noms_tours:
    print("=" * 60)
    print(f"  {tour}")
    print("=" * 60)

    vainqueurs = []

    for eq1, eq2 in matchs_tour_actuel:
        vainqueur, confiance = simuler_match_couperet(eq1, eq2)
        vainqueurs.append(vainqueur)

        # On affiche aussi les points FIFA pour le contexte
        fifa1 = POINTS_FIFA.get(eq1, 1200)
        fifa2 = POINTS_FIFA.get(eq2, 1200)
        print(f"   {eq1:20} ({fifa1:.0f}) vs {eq2:20} ({fifa2:.0f}) -> {vainqueur} ({confiance:.1f}%)")

        # Sauvegarde
        historique_complet.append({
            'tour': tour,
            'equipe_1': eq1,
            'equipe_2': eq2,
            'vainqueur': vainqueur,
            'confiance': round(confiance, 1),
        })

    print()

    # Preparation du tour suivant
    if len(vainqueurs) > 1:
        matchs_tour_actuel = []
        for i in range(0, len(vainqueurs), 2):
            matchs_tour_actuel.append((vainqueurs[i], vainqueurs[i + 1]))
    else:
        champion_final = vainqueurs[0]


# ============================================================
# ETAPE 5 : LE VERDICT FINAL
# ============================================================

print("\n" + "=" * 60)
print("LE VERDICT DE L'IA")
print("=" * 60)
print(f"\n   CHAMPION DU MONDE 2026 : {champion_final.upper()}\n")

# Affichage du parcours du champion
print(f"   Parcours du champion :")
parcours = [h for h in historique_complet if h['vainqueur'] == champion_final]
for match in parcours:
    adversaire = match['equipe_1'] if match['equipe_2'] == champion_final else match['equipe_2']
    print(f"      {match['tour']:20} : bat {adversaire} ({match['confiance']:.1f}%)")


# ============================================================
# ETAPE 6 : SAUVEGARDE
# ============================================================

print("\n[6/6] Sauvegarde des resultats...")

# Sauvegarde de tous les matchs
df_historique = pd.DataFrame(historique_complet)
df_historique.to_csv("data/bracket_complet.csv", index=False)
print("      Bracket complet : data/bracket_complet.csv")

# Sauvegarde du champion
df_champion = pd.DataFrame([{
    'role': 'Champion',
    'equipe': champion_final,
    'version': 'V2 avec vraie forme'
}])
df_champion.to_csv("data/vainqueur_final.csv", index=False)
print(f"      Vainqueur : data/vainqueur_final.csv")