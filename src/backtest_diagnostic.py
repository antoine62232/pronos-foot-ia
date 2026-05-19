import pandas as pd

# Chargement de l'historique
df = pd.read_csv("data/matchs_entrainement.csv")
df['date'] = pd.to_datetime(df['date'])

print("=" * 60)
print("DIAGNOSTIC : matchs Coupe du Monde dans notre historique")
print("=" * 60)

# La colonne 'tournament' indique le nom de la competition
# On filtre uniquement les matchs ou le tournoi est "FIFA World Cup"
matchs_cdm = df[df['tournament'] == 'FIFA World Cup']

print(f"\nNombre total de matchs de CdM dans la BDD : {len(matchs_cdm)}")
print(f"\nRepartition par annee :")
print(matchs_cdm['date'].dt.year.value_counts().sort_index())

# Focus sur 2018 et 2022
print("\n" + "=" * 60)
print("DETAIL CdM 2018")
print("=" * 60)
cdm_2018 = matchs_cdm[matchs_cdm['date'].dt.year == 2018]
print(f"Nombre de matchs : {len(cdm_2018)}")
print(f"Date premier match : {cdm_2018['date'].min()}")
print(f"Date dernier match : {cdm_2018['date'].max()}")
print(f"\nApercu :")
print(cdm_2018[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(5))

print("\n" + "=" * 60)
print("DETAIL CdM 2022")
print("=" * 60)
cdm_2022 = matchs_cdm[matchs_cdm['date'].dt.year == 2022]
print(f"Nombre de matchs : {len(cdm_2022)}")
print(f"Date premier match : {cdm_2022['date'].min()}")
print(f"Date dernier match : {cdm_2022['date'].max()}")
print(f"\nApercu :")
print(cdm_2022[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(5))