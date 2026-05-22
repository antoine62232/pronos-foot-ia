# ============================================================
# ROLE    : Identifier les meilleurs et pires pronostics
# ============================================================

import pandas as pd

print("=" * 70)
print("ANALYSE DETAILLEE DU BACKTEST CdM 2022")
print("=" * 70)

# Chargement des resultats du backtest
df = pd.read_csv("data/backtest_cdm_2022.csv")

# On calcule la probabilite max pour chaque match
df['proba_max'] = df[['proba_1', 'proba_N', 'proba_2']].max(axis=1)

# ============================================================
# CATEGORIE A : Pronostics haute confiance REUSSIS
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 - PRONOSTICS HAUTE CONFIANCE REUSSIS (>60%)")
print("=" * 70)

reussis_haute_conf = df[(df['correct'] == True) & (df['proba_max'] >= 60)]
reussis_haute_conf = reussis_haute_conf.sort_values('proba_max', ascending=False).head(5)

for _, ligne in reussis_haute_conf.iterrows():
    print(f"\n{ligne['date']} - {ligne['match']}")
    print(f"   Score reel : {ligne['score']}")
    print(f"   Probas IA  : Dom {ligne['proba_1']:.0f}% | Nul {ligne['proba_N']:.0f}% | Ext {ligne['proba_2']:.0f}%")
    print(f"   Verdict    : VRAI (confiance {ligne['proba_max']:.0f}%)")

# ============================================================
# CATEGORIE B : Pronostics haute confiance RATES
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 - PRONOSTICS HAUTE CONFIANCE RATES (>60% mais faux)")
print("=" * 70)

rates_haute_conf = df[(df['correct'] == False) & (df['proba_max'] >= 60)]
rates_haute_conf = rates_haute_conf.sort_values('proba_max', ascending=False).head(5)

if len(rates_haute_conf) > 0:
    for _, ligne in rates_haute_conf.iterrows():
        print(f"\n{ligne['date']} - {ligne['match']}")
        print(f"   Score reel : {ligne['score']}")
        print(f"   Probas IA  : Dom {ligne['proba_1']:.0f}% | Nul {ligne['proba_N']:.0f}% | Ext {ligne['proba_2']:.0f}%")
        print(f"   Verdict    : FAUX (etait sur de lui a {ligne['proba_max']:.0f}%)")
else:
    print("\nAucun raté haute confiance !")

# ============================================================
# STATISTIQUES PAR NIVEAU DE CONFIANCE
# ============================================================

print("\n" + "=" * 70)
print("PRECISION PAR NIVEAU DE CONFIANCE")
print("=" * 70)

niveaux = [
    ("Tres haute confiance (>70%)", df['proba_max'] >= 70),
    ("Haute confiance (60-70%)",    (df['proba_max'] >= 60) & (df['proba_max'] < 70)),
    ("Moyenne confiance (50-60%)",  (df['proba_max'] >= 50) & (df['proba_max'] < 60)),
    ("Faible confiance (40-50%)",   (df['proba_max'] >= 40) & (df['proba_max'] < 50)),
    ("Tres faible confiance (<40%)", df['proba_max'] < 40),
]

for nom, condition in niveaux:
    sous_ensemble = df[condition]
    if len(sous_ensemble) > 0:
        nb_correct = sous_ensemble['correct'].sum()
        nb_total = len(sous_ensemble)
        precision = (nb_correct / nb_total) * 100
        print(f"\n{nom:35} : {nb_correct}/{nb_total} ({precision:.0f}%)")