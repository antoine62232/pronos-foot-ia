import pandas as pd

df = pd.read_csv("data/matchs_entrainement.csv")

df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Forme offensive (buts MARQUÉS)

# Moyenne de buts marqués sur les 5 derniers matchs à domicile
df['forme_attaque_domicile'] = df.groupby('home_team')['home_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)

# Moyenne de buts marqués sur les 5 derniers matchs à l'extérieur
df['forme_attaque_exterieur'] = df.groupby('away_team')['away_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)

print("✅ Features offensives calculées.")

# Forme défensive (buts ENCAISSÉS)

# Plus ce chiffre est BAS, meilleure est la défense
# Ex : une équipe qui encaisse 0.5 but/match est solide défensivement

# Buts encaissés par l'équipe domicile = buts marqués par l'adversaire (away_score)
df['forme_defense_domicile'] = df.groupby('home_team')['away_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)

# Buts encaissés par l'équipe extérieure = buts marqués par l'équipe domicile
df['forme_defense_exterieur'] = df.groupby('away_team')['home_score'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)

print("✅ Features défensives calculées.")

# Forme sur 10 matchs 
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
print("✅ Features V2 (10 matchs) calculées.")


# Points FIFA officiels (source : FIFA.com)


points_fifa = {
    # Top mondial
    'France': 1877.32,
    'Spain': 1876.40,
    'Argentina': 1874.81,
    'England': 1825.97,
    'Portugal': 1763.83,
    'Brazil': 1761.16,
    'Netherlands': 1757.87,
    'Morocco': 1755.87,
    'Belgium': 1734.71,
    'Germany': 1730.37,
    'Croatia': 1717.07,
    'Italy': 1700.37,
    'Colombia': 1693.09,
    'Senegal': 1688.99,
    'Mexico': 1681.03,
    'United States': 1673.13,
    'Uruguay': 1673.07,
    'Japan': 1660.43,
    'Switzerland': 1649.40,
    'Denmark': 1620.81,
    'Iran': 1615.30,
    'Turkey': 1599.04,
    'Ecuador': 1594.78,
    'Austria': 1593.45,
    'South Korea': 1588.66,
    'Nigeria': 1585.09,
    'Australia': 1580.67,
    'Algeria': 1564.26,
    'Egypt': 1563.24,
    'Canada': 1556.48,
    'Norway': 1550.94,
    'Ukraine': 1546.88,
    'Panama': 1540.64,
    'Ivory Coast': 1532.98,
    'Poland': 1528.00,
    'Sweden': 1514.77,
    'Serbia': 1508.65,
    'Paraguay': 1503.50,
    'Czech Republic': 1501.38,
    'Hungary': 1500.58,
    'Scotland': 1498.35,
    'Tunisia': 1483.05,
    'Cameroon': 1481.24,
    'DR Congo': 1478.35,
    'Greece': 1475.82,
    'Slovakia': 1473.94,
    'Venezuela': 1468.05,
    'Uzbekistan': 1465.34,
    'Costa Rica': 1459.90,
    'Mali': 1459.13,
    'Peru': 1455.87,
    'Chile': 1455.28,
    'Qatar': 1454.96,
    'Romania': 1451.16,
    'Iraq': 1447.14,
    'Slovenia': 1446.44,
    'South Africa': 1429.73,
    'Saudi Arabia': 1421.43,
    'Burkina Faso': 1412.49,
    'Jordan': 1391.45,
    'Albania': 1388.06,
    'Bosnia and Herzegovina': 1385.84,
    'Honduras': 1380.27,
    'Cape Verde': 1366.13,
    'Jamaica': 1358.00,
    'Georgia': 1350.18,
    'Finland': 1346.41,
    'Ghana': 1346.31,
    'Iceland': 1345.07,
    'Bolivia': 1329.42,
    'Kosovo': 1318.83,
    'Guinea': 1300.01,
    'Montenegro': 1295.52,
    'Curaçao': 1294.65,
    'Haiti': 1291.71,
    'New Zealand': 1281.57,
    'New Caledonia': 1036.95,
}

# On applique les points FIFA à chaque ligne
# Si une équipe n'est pas dans notre dictionnaire → 1200 par défaut (niveau faible)
df['points_fifa_domicile'] = df['home_team'].map(points_fifa).fillna(1200)
df['points_fifa_exterieur'] = df['away_team'].map(points_fifa).fillna(1200)

print("✅ Points FIFA officiels appliqués.")

# Variable cible : Résultat du match

def determiner_resultat(ligne):
    if ligne['home_score'] > ligne['away_score']:
        return "1"
    elif ligne['home_score'] == ligne['away_score']:
        return "N"
    else:
        return "2"

df['resultat'] = df.apply(determiner_resultat, axis=1)

# Match sur terrain neutre :
#   - True  = match sur terrain neutre (Coupe du Monde, finales...)
#   - False = équipe à domicile joue chez elle
# On la convertit en 0/1 pour que XGBoost puisse l'utiliser

df['match_neutre'] = df['neutral'].astype(int)

print("✅ Feature 'match_neutre' créée.")
print(f"   Répartition : {df['match_neutre'].value_counts().to_dict()}")

# Sauvegarde du nouveau dataset avec les features créées
df = df.fillna(0)
df.to_csv("data/matchs_entrainement.csv", index=False)

print("\n✅ Toutes les features créées avec succès !")
print(f"   Nouvelles colonnes : {[c for c in df.columns if 'forme' in c or 'fifa' in c or 'resultat' in c]}")

# Vérification sur un exemple parlant : Brésil vs Maroc
print("\n🔍 Vérification FIFA — Brésil vs Maroc :")
print(f"   Brésil  : {points_fifa['Brazil']} points")
print(f"   Maroc   : {points_fifa['Morocco']} points")
print(f"   Écart   : {points_fifa['Brazil'] - points_fifa['Morocco']:.2f} points (très proches !)")

print("\n🔍 Vérification FIFA — Uruguay vs Arabie Saoudite :")
print(f"   Uruguay       : {points_fifa['Uruguay']} points")
print(f"   Arabie Soud.  : {points_fifa['Saudi Arabia']} points")
print(f"   Écart         : {points_fifa['Uruguay'] - points_fifa['Saudi Arabia']:.2f} points ✅")