# ============================================================
# FICHIER : src/enrichir_groupes.py
# ROLE    : Ajouter la colonne 'group' au fichier matchs_a_predire.csv
# ============================================================

import pandas as pd


# ============================================================
# LES 12 GROUPES DE LA CDM 2026 (officiels)
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


def trouver_groupe(equipe):
    """
    Trouve a quel groupe appartient une equipe.

    Parametres :
        equipe : nom de l'equipe

    Retour :
        str : lettre du groupe (A a L) ou None si non trouve
    """
    for lettre_groupe, equipes in GROUPES.items():
        if equipe in equipes:
            return lettre_groupe
    return None


# ============================================================
# ENRICHISSEMENT DU FICHIER
# ============================================================

print("=" * 60)
print("ENRICHISSEMENT - Ajout colonne 'group' au CSV CdM 2026")
print("=" * 60)

# Chargement
print("\n[1/4] Chargement du fichier...")
df = pd.read_csv("data/matchs_a_predire.csv")
print(f"      {len(df)} matchs charges")

# Verification : toutes les equipes du CSV sont-elles dans nos groupes ?
print("\n[2/4] Verification de la coherence des equipes...")
equipes_csv = set(df['home_team'].tolist() + df['away_team'].tolist())
equipes_groupes = set()
for equipes in GROUPES.values():
    equipes_groupes.update(equipes)

# Equipes presentes dans CSV mais pas dans nos groupes
manquantes = equipes_csv - equipes_groupes
if manquantes:
    print(f"      [!!!] EQUIPES NON TROUVEES DANS NOS GROUPES :")
    for eq in sorted(manquantes):
        print(f"            - {eq}")
    print(f"      -> Verifier l'orthographe ou completer la liste GROUPES")
else:
    print(f"      [OK] Les 48 equipes du CSV correspondent a nos groupes")

# Equipes dans nos groupes mais pas dans CSV (anomalie)
en_trop = equipes_groupes - equipes_csv
if en_trop:
    print(f"\n      [!!!] EQUIPES DANS NOS GROUPES MAIS ABSENTES DU CSV :")
    for eq in sorted(en_trop):
        print(f"            - {eq}")

# Ajout de la colonne 'group'
print("\n[3/4] Ajout de la colonne 'group'...")
df['group'] = df['home_team'].apply(trouver_groupe)

# Verification que tous les matchs ont un groupe
matchs_sans_groupe = df[df['group'].isna()]
if len(matchs_sans_groupe) > 0:
    print(f"      [!!!] {len(matchs_sans_groupe)} matchs sans groupe :")
    print(matchs_sans_groupe[['home_team', 'away_team']].head())
else:
    print(f"      [OK] Tous les matchs ont un groupe assigne")

# Verification de coherence : les 2 equipes du match sont-elles dans le meme groupe ?
print("\n[4/4] Verification : 2 equipes dans le meme groupe...")
incoherences = []
for idx, row in df.iterrows():
    groupe_dom = trouver_groupe(row['home_team'])
    groupe_ext = trouver_groupe(row['away_team'])
    if groupe_dom != groupe_ext:
        incoherences.append({
            'match': f"{row['home_team']} vs {row['away_team']}",
            'groupe_dom': groupe_dom,
            'groupe_ext': groupe_ext
        })

if incoherences:
    print(f"      [!!!] {len(incoherences)} INCOHERENCES (equipes pas dans le meme groupe) :")
    for inc in incoherences[:5]:
        print(f"            - {inc['match']} : {inc['groupe_dom']} vs {inc['groupe_ext']}")
else:
    print(f"      [OK] Toutes les paires d'equipes sont dans le meme groupe")

# Sauvegarde
print("\n[Sauvegarde] Mise a jour du fichier matchs_a_predire.csv...")
df.to_csv("data/matchs_a_predire.csv", index=False)
print("      [OK] Fichier sauvegarde avec la nouvelle colonne 'group'")

# Resume
print("\n" + "=" * 60)
print("RESUME PAR GROUPE")
print("=" * 60)
for lettre in sorted(GROUPES.keys()):
    matchs_groupe = df[df['group'] == lettre]
    equipes = GROUPES[lettre]
    print(f"\nGROUPE {lettre} : {', '.join(equipes)}")
    print(f"  {len(matchs_groupe)} matchs prevus")