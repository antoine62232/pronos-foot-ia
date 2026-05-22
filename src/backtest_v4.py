# ============================================================
# ROLE    : Backtest V4 - Modele EPURE (14 features selectionnees)
# VERSION : V4 - Seules les features avec importance >= 3%
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# On reutilise les fonctions du backtest V3
import sys
sys.path.append('src')
from backtest_v3 import (
    POINTS_FIFA, CONFEDERATION,
    determiner_resultat, calculer_forme,
    calculer_features_v3_globales,
    preparer_donnees_equipe
)


def backtester_cdm_v4(annee_cdm, date_debut_cdm):
    """Backteste le modele V4 (epure) sur une Coupe du Monde donnee."""

    print(f"\n{'=' * 60}")
    print(f"BACKTEST V4 (EPURE) - CdM {annee_cdm}")
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

    # 3. FEATURES V2 - SEULEMENT LA FORME SUR 10 MATCHS (on supprime les 5 matchs)
    print(f"[3/6] Calcul des features (forme 10 matchs uniquement)...")

    # Forme sur 10 matchs UNIQUEMENT
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

    # 3.5 FEATURES V3 - SELECTIONNEES
    print(f"[3.5/6] Calcul des features V3 selectionnees...")
    df_features_v3 = calculer_features_v3_globales(df_train)

    df_v3_dom = df_features_v3.rename(columns={
        'equipe': 'home_team',
        'diff_buts_10': 'diff_buts_dom',
        'clean_sheets_10': 'clean_sheets_dom',
        'streak_L': 'streak_defaites_dom',
        'streak_W': 'streak_victoires_dom_drop',  # On garde pour merge, on supprime apres
    })
    df_v3_ext = df_features_v3.rename(columns={
        'equipe': 'away_team',
        'diff_buts_10': 'diff_buts_ext',
        'clean_sheets_10': 'clean_sheets_ext',
        'streak_L': 'streak_defaites_ext_drop',
        'streak_W': 'streak_victoires_ext_drop',
    })

    df_train = df_train.merge(df_v3_dom, on=['date', 'home_team'], how='left')
    df_train = df_train.merge(df_v3_ext, on=['date', 'away_team'], how='left')

    # Confederations
    df_train['conf_dom'] = df_train['home_team'].map(CONFEDERATION).fillna(0)
    df_train['conf_ext'] = df_train['away_team'].map(CONFEDERATION).fillna(0)

    # Resultat
    df_train['resultat'] = df_train.apply(
        lambda row: determiner_resultat(row['home_score'], row['away_score']),
        axis=1
    )
    df_train = df_train.fillna(0)

    # 4. ENTRAINEMENT - SEULEMENT 14 FEATURES SELECTIONNEES
    print(f"\n[4/6] Entrainement du modele V4 (14 features)...")

    features = [
        # Top FIFA (les + importantes)
        'points_fifa_domicile', 'points_fifa_exterieur',
        # Contexte
        'match_neutre',
        # Diff buts (excellentes)
        'diff_buts_dom', 'diff_buts_ext',
        # Forme sur 10 matchs (les seules de la forme qui valent le coup)
        'forme_attaque_domicile_10', 'forme_attaque_exterieur_10',
        'forme_defense_domicile_10', 'forme_defense_exterieur_10',
        # Confederations
        'conf_dom', 'conf_ext',
        # Clean sheets
        'clean_sheets_dom', 'clean_sheets_ext',
        # Streak defaites domicile (la seule streak utile)
        'streak_defaites_dom',
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
    print(f"      Modele V4 entraine sur {len(X_train)} matchs ({len(features)} features).")

    # 5. PREDICTION
    print(f"[5/6] Prediction des matchs de la CdM {annee_cdm}...")

    def get_features_v3(equipe, date_match):
        """Recupere les features V3 pour une equipe a une date donnee."""
        matchs_eq = preparer_donnees_equipe(df_train, equipe)
        matchs_eq = matchs_eq[matchs_eq['date'] < date_match]
        if len(matchs_eq) == 0:
            return 0, 0.0, 0  # diff, clean_sheets, streak_l

        derniers_10 = matchs_eq.tail(10)
        diff = (derniers_10['buts_pour'] - derniers_10['buts_contre']).sum()
        cs_ratio = round(derniers_10['clean_sheet'].mean(), 2)

        # Streak defaites en cours
        streak_l = 0
        for r in reversed(matchs_eq['resultat'].tolist()):
            if r == 'L':
                streak_l += 1
            else:
                break

        return diff, cs_ratio, streak_l

    predictions_list = []

    for _, match in df_test.iterrows():
        eq_dom = match['home_team']
        eq_ext = match['away_team']
        date_m = match['date']

        # Forme 10 matchs
        forme_att_dom_10 = calculer_forme(df_train, eq_dom, date_m, 'attaque', nb_matchs=10)
        forme_att_ext_10 = calculer_forme(df_train, eq_ext, date_m, 'attaque', nb_matchs=10)
        forme_def_dom_10 = calculer_forme(df_train, eq_dom, date_m, 'defense', nb_matchs=10)
        forme_def_ext_10 = calculer_forme(df_train, eq_ext, date_m, 'defense', nb_matchs=10)

        # FIFA et contexte
        fifa_dom = POINTS_FIFA.get(eq_dom, 1200)
        fifa_ext = POINTS_FIFA.get(eq_ext, 1200)
        match_neutre = int(match['neutral'])

        # V3 features
        diff_dom, cs_dom, streak_l_dom = get_features_v3(eq_dom, date_m)
        diff_ext, cs_ext, _ = get_features_v3(eq_ext, date_m)

        # Confederations
        conf_dom = CONFEDERATION.get(eq_dom, 0)
        conf_ext = CONFEDERATION.get(eq_ext, 0)

        donnees_match = pd.DataFrame([[
            fifa_dom, fifa_ext,
            match_neutre,
            diff_dom, diff_ext,
            forme_att_dom_10, forme_att_ext_10,
            forme_def_dom_10, forme_def_ext_10,
            conf_dom, conf_ext,
            cs_dom, cs_ext,
            streak_l_dom,
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
    print(f"RESULTAT V4 CdM {annee_cdm} : {nb_correct}/{nb_total} ({precision:.1f}%)")
    print(f"{'-' * 60}")

    print(f"\nDetail par type :")
    for r_type, label in [('1', 'Victoires domicile'), ('N', 'Matchs nuls'), ('2', 'Victoires exterieur')]:
        matchs_t = df_predictions[df_predictions['vrai_resultat'] == r_type]
        if len(matchs_t) > 0:
            c_t = matchs_t['correct'].sum()
            taux = (c_t / len(matchs_t)) * 100
            print(f"   {label:25} : {c_t}/{len(matchs_t)} ({taux:.0f}%)")

    fichier_sortie = f"data/backtest_v4_cdm_{annee_cdm}.csv"
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
    print("BACKTEST V4 (EPURE) - 14 FEATURES")
    print("=" * 60)

    resultats_2018 = backtester_cdm_v4(2018, '2018-06-14')
    resultats_2022 = backtester_cdm_v4(2022, '2022-11-20')

    print(f"\n{'=' * 60}")
    print(f"RESUME FINAL V4 vs V2 vs V3.1")
    print(f"{'=' * 60}")
    print(f"\n                       V2          V3.1         V4")
    print(f"CdM 2018             51.6%       53.1%       {resultats_2018['precision']:.1f}%")
    print(f"CdM 2022             57.8%       54.7%       {resultats_2022['precision']:.1f}%")
    moyenne = (resultats_2018['precision'] + resultats_2022['precision']) / 2
    print(f"Moyenne              54.7%       53.9%       {moyenne:.1f}%")