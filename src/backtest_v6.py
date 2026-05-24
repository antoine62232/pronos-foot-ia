# ============================================================
# ROLE    : V6 - V2 + features de probabilites Elo
# VERSION : V6 (14 features = 11 V2 + 3 Elo)
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
    """Forme attaque/defense sur N derniers matchs (V2 features)."""
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
# NOUVELLES FONCTIONS V6 - FEATURES ELO
# ============================================================

def calculer_proba_elo(points_fifa_a, points_fifa_b):
    """
    Calcule la probabilite que l'equipe A batte l'equipe B selon la formule Elo.

    Parametres :
        points_fifa_a : Points FIFA de l'equipe A
        points_fifa_b : Points FIFA de l'equipe B

    Retour :
        float : probabilite entre 0 et 1 que l'equipe A gagne
    """
    # Difference de points (positif = A est plus fort)
    diff = points_fifa_a - points_fifa_b

    # Formule Elo classique
    # Si diff = 0   : proba = 50% (equipes equilibrees)
    # Si diff > 0   : A est favoris (proba > 50%)
    # Si diff < 0   : A est outsider (proba < 50%)
    proba = 1 / (1 + 10 ** (-diff / 400))

    return round(proba, 4)


# ============================================================
# FONCTION PRINCIPALE DE BACKTEST V6
# ============================================================

def backtester_cdm_v6(annee_cdm, date_debut_cdm):
    """Backteste le modele V6 sur une Coupe du Monde donnee."""

    print(f"\n{'=' * 60}")
    print(f"BACKTEST V6 - CdM {annee_cdm}")
    print(f"{'=' * 60}")

    # 1. CHARGEMENT
    print(f"\n[1/6] Chargement des donnees...")
    df = pd.read_csv("data/matchs_entrainement.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    date_debut_cdm = pd.to_datetime(date_debut_cdm)

    # 2. SEPARATION TEMPORELLE
    print(f"[2/6] Separation temporelle avant le {date_debut_cdm.date()}...")
    df_train = df[df['date'] < date_debut_cdm].copy()
    df_train = df_train.sort_values(by='date').reset_index(drop=True)
    print(f"      Matchs d'entrainement : {len(df_train)}")

    df_test = df[
        (df['tournament'] == 'FIFA World Cup') &
        (df['date'].dt.year == annee_cdm)
    ].copy()
    print(f"      Matchs CdM {annee_cdm} a predire : {len(df_test)}")

    # 3. FEATURES V2
    print(f"[3/6] Calcul des features V2...")

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

    # 3.5 NOUVELLES FEATURES V6 - ELO
    print(f"[3.5/6] Calcul des probabilites Elo...")

    # Vectorisation : on calcule la proba Elo directement sur tout le DataFrame
    # C'est tres rapide car c'est une operation mathematique vectorielle
    df_train['proba_elo_dom'] = 1 / (
        1 + 10 ** (-(df_train['points_fifa_domicile'] - df_train['points_fifa_exterieur']) / 400)
    )
    df_train['proba_elo_ext'] = 1 - df_train['proba_elo_dom']
    df_train['diff_elo'] = df_train['points_fifa_domicile'] - df_train['points_fifa_exterieur']

    # Statistiques pour verification
    print(f"        Proba Elo domicile : min={df_train['proba_elo_dom'].min():.2f}, "
          f"max={df_train['proba_elo_dom'].max():.2f}, "
          f"moyenne={df_train['proba_elo_dom'].mean():.2f}")

    # Resultat
    df_train['resultat'] = df_train.apply(
        lambda row: determiner_resultat(row['home_score'], row['away_score']),
        axis=1
    )
    df_train = df_train.fillna(0)

    # 4. ENTRAINEMENT
    print(f"\n[4/6] Entrainement du modele V6...")

    features = [
        # V2 features (11)
        'forme_attaque_domicile', 'forme_attaque_exterieur',
        'forme_defense_domicile', 'forme_defense_exterieur',
        'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
        'forme_defense_domicile_10', 'forme_defense_exterieur_10',
        'points_fifa_domicile', 'points_fifa_exterieur',
        'match_neutre',
        # V6 NOUVELLES FEATURES ELO (3)
        'proba_elo_dom', 'proba_elo_ext', 'diff_elo',
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
    print(f"      Modele V6 entraine sur {len(X_train)} matchs ({len(features)} features).")

    # 5. PREDICTION
    print(f"[5/6] Prediction des matchs de la CdM {annee_cdm}...")

    predictions_list = []

    for _, match in df_test.iterrows():
        eq_dom = match['home_team']
        eq_ext = match['away_team']
        date_m = match['date']

        # Features V2
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

        # NOUVELLES FEATURES V6 - ELO
        proba_elo_dom = calculer_proba_elo(fifa_dom, fifa_ext)
        proba_elo_ext = 1 - proba_elo_dom
        diff_elo = fifa_dom - fifa_ext

        donnees_match = pd.DataFrame([[
            forme_att_dom, forme_att_ext,
            forme_def_dom, forme_def_ext,
            forme_att_dom_10, forme_att_ext_10,
            forme_def_dom_10, forme_def_ext_10,
            fifa_dom, fifa_ext,
            match_neutre,
            proba_elo_dom, proba_elo_ext, diff_elo,
        ]], columns=features)

        proba = modele.predict_proba(donnees_match)[0]
        classes = encoder.inverse_transform(modele.classes_)
        probas_dict = dict(zip(classes, proba))
        pronostic = max(probas_dict, key=probas_dict.get)
        vrai_resultat = determiner_resultat(match['home_score'], match['away_score'])

        predictions_list.append({
            'date': match['date'].strftime('%Y-%m-%d'),
            'match': f"{eq_dom} vs {eq_ext}",
            'score': f"{match['home_score']}-{match['away_score']}",
            'vrai_resultat': vrai_resultat,
            'pronostic': pronostic,
            'correct': pronostic == vrai_resultat,
            'proba_1': probas_dict.get('1', 0) * 100,
            'proba_N': probas_dict.get('N', 0) * 100,
            'proba_2': probas_dict.get('2', 0) * 100,
            'proba_elo_dom': proba_elo_dom * 100,  # Pour analyse posterieure
        })

    df_predictions = pd.DataFrame(predictions_list)

    # 6. ANALYSE
    print(f"[6/6] Analyse...\n")
    nb_correct = df_predictions['correct'].sum()
    nb_total = len(df_predictions)
    precision = (nb_correct / nb_total) * 100

    print(f"{'-' * 60}")
    print(f"RESULTAT V6 CdM {annee_cdm} : {nb_correct}/{nb_total} ({precision:.1f}%)")
    print(f"{'-' * 60}")

    print(f"\nDetail par type :")
    for r_type, label in [('1', 'Victoires domicile'), ('N', 'Matchs nuls'), ('2', 'Victoires exterieur')]:
        matchs_t = df_predictions[df_predictions['vrai_resultat'] == r_type]
        if len(matchs_t) > 0:
            c_t = matchs_t['correct'].sum()
            taux = (c_t / len(matchs_t)) * 100
            print(f"   {label:25} : {c_t}/{len(matchs_t)} ({taux:.0f}%)")

    fichier_sortie = f"data/backtest_v6_cdm_{annee_cdm}.csv"
    df_predictions.to_csv(fichier_sortie, index=False)
    print(f"\nResultats sauvegardes : {fichier_sortie}")

    return {
        'annee': annee_cdm,
        'precision': precision,
        'nb_correct': nb_correct,
        'nb_total': nb_total,
    }


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BACKTEST V6 - V2 + Probabilites Elo (14 features)")
    print("=" * 60)

    resultats_2018 = backtester_cdm_v6(2018, '2018-06-14')
    resultats_2022 = backtester_cdm_v6(2022, '2022-11-20')

    print(f"\n{'=' * 60}")
    print(f"RESUME FINAL V6 vs V2")
    print(f"{'=' * 60}")
    print(f"\n                       V2          V6")
    print(f"CdM 2018             51.6%       {resultats_2018['precision']:.1f}%")
    print(f"CdM 2022             57.8%       {resultats_2022['precision']:.1f}%")
    moyenne = (resultats_2018['precision'] + resultats_2022['precision']) / 2
    print(f"Moyenne              54.7%       {moyenne:.1f}%")