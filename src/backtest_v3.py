# ============================================================
# FICHIER : src/backtest_v3.py (CORRIGE)
# ROLE    : Backtest V3 - Features V3 calculees SUR TOUT le dataset
# VERSION : V3.1 - 21 features, approche vectorisee
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# POINTS FIFA
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
# MAPPING DES CONFEDERATIONS
# ============================================================

CONFEDERATION = {
    # UEFA = 1
    'France': 1, 'Spain': 1, 'England': 1, 'Portugal': 1, 'Netherlands': 1,
    'Belgium': 1, 'Germany': 1, 'Croatia': 1, 'Italy': 1, 'Switzerland': 1,
    'Denmark': 1, 'Turkey': 1, 'Austria': 1, 'Norway': 1, 'Ukraine': 1,
    'Poland': 1, 'Sweden': 1, 'Serbia': 1, 'Czech Republic': 1, 'Hungary': 1,
    'Scotland': 1, 'Greece': 1, 'Slovakia': 1, 'Slovenia': 1, 'Romania': 1,
    'Albania': 1, 'Bosnia and Herzegovina': 1, 'Wales': 1, 'Georgia': 1,
    'Finland': 1, 'Iceland': 1, 'Kosovo': 1, 'Montenegro': 1,
    # CONMEBOL = 2
    'Argentina': 2, 'Brazil': 2, 'Colombia': 2, 'Uruguay': 2, 'Ecuador': 2,
    'Peru': 2, 'Chile': 2, 'Paraguay': 2, 'Venezuela': 2, 'Bolivia': 2,
    # CAF = 3
    'Morocco': 3, 'Senegal': 3, 'Nigeria': 3, 'Algeria': 3, 'Egypt': 3,
    'Ivory Coast': 3, 'Tunisia': 3, 'Cameroon': 3, 'DR Congo': 3,
    'Mali': 3, 'South Africa': 3, 'Burkina Faso': 3, 'Cape Verde': 3,
    'Ghana': 3, 'Guinea': 3,
    # AFC = 4
    'Japan': 4, 'Iran': 4, 'South Korea': 4, 'Australia': 4, 'Qatar': 4,
    'Saudi Arabia': 4, 'Uzbekistan': 4, 'Iraq': 4, 'Jordan': 4,
    # CONCACAF = 5
    'Mexico': 5, 'United States': 5, 'Canada': 5, 'Panama': 5,
    'Costa Rica': 5, 'Honduras': 5, 'Jamaica': 5,
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


def preparer_donnees_equipe(df, equipe):
    """
    Prepare une vue unifiee de tous les matchs d'une equipe avec :
    - resultat (W/L/D)
    - buts marques / encaisses
    - clean sheet (0/1)
    """
    # Matchs ou l'equipe est a domicile
    home = df[df['home_team'] == equipe].copy()
    home['equipe'] = equipe
    home['buts_pour'] = home['home_score']
    home['buts_contre'] = home['away_score']

    # Matchs ou l'equipe est a l'exterieur
    away = df[df['away_team'] == equipe].copy()
    away['equipe'] = equipe
    away['buts_pour'] = away['away_score']
    away['buts_contre'] = away['home_score']

    # Combinaison
    matchs_eq = pd.concat([home, away]).sort_values('date').reset_index(drop=True)

    # Resultat
    matchs_eq['resultat'] = np.where(
        matchs_eq['buts_pour'] > matchs_eq['buts_contre'], 'W',
        np.where(matchs_eq['buts_pour'] < matchs_eq['buts_contre'], 'L', 'D')
    )

    # Clean sheet
    matchs_eq['clean_sheet'] = (matchs_eq['buts_contre'] == 0).astype(int)

    return matchs_eq[['date', 'equipe', 'resultat', 'buts_pour', 'buts_contre', 'clean_sheet']]


def calculer_features_v3_pour_equipe(matchs_eq):
    """
    Calcule streak, diff_buts, clean_sheets pour CHAQUE ligne (en respectant l'ordre temporel).
    Utilise des rolling windows vectorises pour aller vite.
    """
    # Rolling sur les 10 derniers matchs
    matchs_eq['diff_buts_10'] = (matchs_eq['buts_pour'] - matchs_eq['buts_contre']).rolling(
        10, min_periods=1
    ).sum().shift(1).fillna(0)

    matchs_eq['clean_sheets_10'] = matchs_eq['clean_sheet'].rolling(
        10, min_periods=1
    ).mean().shift(1).fillna(0).round(2)

    # Streak victoires (W consecutifs a la fin)
    # On cree un compteur qui se reset des qu'on rencontre autre chose que W
    matchs_eq['est_W'] = (matchs_eq['resultat'] == 'W').astype(int)
    matchs_eq['est_L'] = (matchs_eq['resultat'] == 'L').astype(int)

    # Cumul qui se reset quand on rencontre un 0
    matchs_eq['streak_W'] = matchs_eq['est_W'] * (
        matchs_eq['est_W'].groupby((matchs_eq['est_W'] == 0).cumsum()).cumcount() + 1
    )
    matchs_eq['streak_L'] = matchs_eq['est_L'] * (
        matchs_eq['est_L'].groupby((matchs_eq['est_L'] == 0).cumsum()).cumcount() + 1
    )

    # On shift pour ne pas inclure le match en cours
    matchs_eq['streak_W'] = matchs_eq['streak_W'].shift(1).fillna(0).astype(int)
    matchs_eq['streak_L'] = matchs_eq['streak_L'].shift(1).fillna(0).astype(int)

    return matchs_eq[['date', 'equipe', 'diff_buts_10', 'clean_sheets_10',
                       'streak_W', 'streak_L']]


def calculer_features_v3_globales(df):
    """
    Calcule les features V3 pour TOUTES les equipes en utilisant l'approche vectorisee.
    Retourne un DataFrame indexe par (date, equipe).
    """
    print("        Preparation des donnees par equipe...")

    # On recupere toutes les equipes uniques
    equipes_dom = df['home_team'].unique()
    equipes_ext = df['away_team'].unique()
    toutes_equipes = set(list(equipes_dom) + list(equipes_ext))

    print(f"        {len(toutes_equipes)} equipes a traiter...")

    # Pour chaque equipe, calculer ses features V3
    liste_features = []
    for i, equipe in enumerate(toutes_equipes):
        if (i + 1) % 50 == 0:
            print(f"        ... {i + 1}/{len(toutes_equipes)} equipes traitees")

        matchs_eq = preparer_donnees_equipe(df, equipe)
        if len(matchs_eq) == 0:
            continue

        features_eq = calculer_features_v3_pour_equipe(matchs_eq)
        liste_features.append(features_eq)

    # Concatener tout
    df_features = pd.concat(liste_features).reset_index(drop=True)
    return df_features


# ============================================================
# FONCTION PRINCIPALE DE BACKTEST V3
# ============================================================

def backtester_cdm(annee_cdm, date_debut_cdm):
    """Backteste le modele V3 (corrige) sur une Coupe du Monde donnee."""

    print(f"\n{'=' * 60}")
    print(f"BACKTEST V3.1 (CORRIGE) - CdM {annee_cdm}")
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

    # 3. FEATURES V2 (forme 5 et 10 matchs)
    print(f"[3/6] Calcul des features V2 (forme 5 et 10 matchs)...")

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

    # 3.5 FEATURES V3 - VECTORISE !
    print(f"[3.5/6] Calcul des features V3 (approche vectorisee)...")
    df_features_v3 = calculer_features_v3_globales(df_train)

    # Fusionner les features V3 avec df_train
    # Pour chaque match, on prend les stats du domicile ET de l'exterieur
    print(f"        Fusion des features V3 dans df_train...")
    df_features_v3_dom = df_features_v3.rename(columns={
        'equipe': 'home_team',
        'diff_buts_10': 'diff_buts_dom',
        'clean_sheets_10': 'clean_sheets_dom',
        'streak_W': 'streak_victoires_dom',
        'streak_L': 'streak_defaites_dom',
    })
    df_features_v3_ext = df_features_v3.rename(columns={
        'equipe': 'away_team',
        'diff_buts_10': 'diff_buts_ext',
        'clean_sheets_10': 'clean_sheets_ext',
        'streak_W': 'streak_victoires_ext',
        'streak_L': 'streak_defaites_ext',
    })

    df_train = df_train.merge(df_features_v3_dom, on=['date', 'home_team'], how='left')
    df_train = df_train.merge(df_features_v3_ext, on=['date', 'away_team'], how='left')

    # Confederations
    df_train['conf_dom'] = df_train['home_team'].map(CONFEDERATION).fillna(0)
    df_train['conf_ext'] = df_train['away_team'].map(CONFEDERATION).fillna(0)

    # Resultat
    df_train['resultat'] = df_train.apply(
        lambda row: determiner_resultat(row['home_score'], row['away_score']),
        axis=1
    )
    df_train = df_train.fillna(0)

    # 4. ENTRAINEMENT
    print(f"\n[4/6] Entrainement du modele XGBoost (V3 corrigee)...")

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
    print(f"      Modele V3 entraine sur {len(X_train)} matchs ({len(features)} features).")

    # 5. PREDICTION (on utilise les fonctions ponctuelles ici, pour les 64 matchs c'est ok)
    print(f"[5/6] Prediction des matchs de la CdM {annee_cdm}...")

    # Helper pour le streak / diff / cs en mode "ponctuel"
    def get_streak_diff_cs(equipe, date_match):
        matchs_eq = preparer_donnees_equipe(df_train, equipe)
        matchs_eq = matchs_eq[matchs_eq['date'] < date_match]
        if len(matchs_eq) == 0:
            return 0, 0, 0, 0.0
        # Diff buts sur 10 derniers
        derniers_10 = matchs_eq.tail(10)
        diff = (derniers_10['buts_pour'] - derniers_10['buts_contre']).sum()
        # Clean sheets ratio
        cs_ratio = round(derniers_10['clean_sheet'].mean(), 2)
        # Streaks
        resultats = matchs_eq['resultat'].tolist()
        streak_w = 0
        streak_l = 0
        for r in reversed(resultats):
            if r == 'W' and streak_l == 0:
                streak_w += 1
            elif r != 'W' and streak_w > 0:
                break
            elif r == 'L' and streak_w == 0:
                streak_l += 1
            elif r != 'L' and streak_l > 0:
                break
            else:
                break
        return streak_w, streak_l, diff, cs_ratio

    predictions_list = []

    for _, match in df_test.iterrows():
        eq_dom = match['home_team']
        eq_ext = match['away_team']
        date_m = match['date']

        # V2 features
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

        # V3 features
        sw_dom, sl_dom, diff_dom, cs_dom = get_streak_diff_cs(eq_dom, date_m)
        sw_ext, sl_ext, diff_ext, cs_ext = get_streak_diff_cs(eq_ext, date_m)
        conf_dom = CONFEDERATION.get(eq_dom, 0)
        conf_ext = CONFEDERATION.get(eq_ext, 0)

        donnees_match = pd.DataFrame([[
            forme_att_dom, forme_att_ext,
            forme_def_dom, forme_def_ext,
            forme_att_dom_10, forme_att_ext_10,
            forme_def_dom_10, forme_def_ext_10,
            fifa_dom, fifa_ext,
            match_neutre,
            sw_dom, sw_ext,
            sl_dom, sl_ext,
            diff_dom, diff_ext,
            cs_dom, cs_ext,
            conf_dom, conf_ext,
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
        })

    df_predictions = pd.DataFrame(predictions_list)

    # 6. ANALYSE
    print(f"[6/6] Analyse...\n")
    nb_correct = df_predictions['correct'].sum()
    nb_total = len(df_predictions)
    precision = (nb_correct / nb_total) * 100

    print(f"{'-' * 60}")
    print(f"RESULTAT V3.1 CdM {annee_cdm} : {nb_correct}/{nb_total} ({precision:.1f}%)")
    print(f"{'-' * 60}")

    print(f"\nDetail par type :")
    for r_type, label in [('1', 'Victoires domicile'), ('N', 'Matchs nuls'), ('2', 'Victoires exterieur')]:
        matchs_t = df_predictions[df_predictions['vrai_resultat'] == r_type]
        if len(matchs_t) > 0:
            c_t = matchs_t['correct'].sum()
            taux = (c_t / len(matchs_t)) * 100
            print(f"   {label:25} : {c_t}/{len(matchs_t)} ({taux:.0f}%)")

    fichier_sortie = f"data/backtest_v3_cdm_{annee_cdm}.csv"
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
    print("BACKTEST V3.1 (CORRIGE) - 21 FEATURES")
    print("=" * 60)

    resultats_2018 = backtester_cdm(2018, '2018-06-14')
    resultats_2022 = backtester_cdm(2022, '2022-11-20')

    print(f"\n{'=' * 60}")
    print(f"RESUME FINAL V3.1")
    print(f"{'=' * 60}")
    print(f"\nComparaison V2 vs V3.1 :")
    print(f"                       V2          V3.1")
    print(f"CdM 2018             51.6%       {resultats_2018['precision']:.1f}%")
    print(f"CdM 2022             57.8%       {resultats_2022['precision']:.1f}%")
    moyenne = (resultats_2018['precision'] + resultats_2022['precision']) / 2
    print(f"Moyenne              54.7%       {moyenne:.1f}%")