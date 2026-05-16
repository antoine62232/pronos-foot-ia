import streamlit as st
import pandas as pd
from joblib import load
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards

# CONFIGURATION DE LA PAGE

st.set_page_config(
    page_title="Pronos Foot IA — Coupe du Monde 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS PERSONNALISÉ

st.markdown("""
<style>
    /* Fond principal */
    .stApp {
        background-color: #0A0E1A;
    }

    /* Textes en blanc cassé */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #F8FAFC !important;
    }

    /* Onglets stylés */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #141B2D;
        padding: 4px;
        border-radius: 10px;
        gap: 4px;
        border: 0.5px solid #1E293B;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94A3B8;
        border-radius: 8px;
        padding: 8px 20px;
        font-size: 14px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(34,211,238,0.1)) !important;
        color: #A78BFA !important;
    }

    /* On cache les éléments Streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# CHARGEMENT DU MODÈLE ET DES DONNÉES

@st.cache_resource
def charger_modele():
    return load("models/modele_football.pkl")

@st.cache_resource
def charger_encoder():
    return load("models/label_encoder.pkl")

@st.cache_data
def charger_matchs():
    return pd.read_csv("data/matchs_a_predire.csv")

@st.cache_data
def charger_historique():
    return pd.read_csv("data/matchs_entrainement.csv")

modele = charger_modele()
encoder = charger_encoder()
matchs_futurs = charger_matchs()
historique = charger_historique()

# POINTS FIFA OFFICIELS

points_fifa = {
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
    'Honduras': 1380.27, 'Cape Verde': 1366.13, 'Jamaica': 1358.00,
    'Georgia': 1350.18, 'Finland': 1346.41, 'Ghana': 1346.31,
    'Iceland': 1345.07, 'Bolivia': 1329.42, 'Kosovo': 1318.83,
    'Guinea': 1300.01, 'Montenegro': 1295.52, 'Curaçao': 1294.65,
    'Haiti': 1291.71, 'New Zealand': 1281.57, 'New Caledonia': 1036.95,
}

# CODES ISO DES PAYS (pour les drapeaux via flagcdn.com)

# On associe chaque pays à son code ISO (2 lettres)
# flagcdn.com génère le drapeau à partir de ce code
codes_iso = {
    'France': 'fr', 'Spain': 'es', 'Argentina': 'ar', 'England': 'gb-eng',
    'Portugal': 'pt', 'Brazil': 'br', 'Netherlands': 'nl', 'Morocco': 'ma',
    'Belgium': 'be', 'Germany': 'de', 'Croatia': 'hr', 'Italy': 'it',
    'Colombia': 'co', 'Senegal': 'sn', 'Mexico': 'mx', 'United States': 'us',
    'Uruguay': 'uy', 'Japan': 'jp', 'Switzerland': 'ch', 'Denmark': 'dk',
    'Iran': 'ir', 'Turkey': 'tr', 'Ecuador': 'ec', 'Austria': 'at',
    'South Korea': 'kr', 'Nigeria': 'ng', 'Australia': 'au', 'Algeria': 'dz',
    'Egypt': 'eg', 'Canada': 'ca', 'Norway': 'no', 'Ukraine': 'ua',
    'Panama': 'pa', 'Ivory Coast': 'ci', 'Poland': 'pl', 'Sweden': 'se',
    'Serbia': 'rs', 'Paraguay': 'py', 'Czech Republic': 'cz', 'Hungary': 'hu',
    'Scotland': 'gb-sct', 'Tunisia': 'tn', 'Cameroon': 'cm', 'DR Congo': 'cd',
    'Greece': 'gr', 'Slovakia': 'sk', 'Venezuela': 've', 'Uzbekistan': 'uz',
    'Costa Rica': 'cr', 'Mali': 'ml', 'Peru': 'pe', 'Chile': 'cl',
    'Qatar': 'qa', 'Romania': 'ro', 'Iraq': 'iq', 'Slovenia': 'si',
    'South Africa': 'za', 'Saudi Arabia': 'sa', 'Burkina Faso': 'bf',
    'Jordan': 'jo', 'Albania': 'al', 'Bosnia and Herzegovina': 'ba',
    'Honduras': 'hn', 'Cape Verde': 'cv', 'Jamaica': 'jm', 'Georgia': 'ge',
    'Finland': 'fi', 'Ghana': 'gh', 'Iceland': 'is', 'Bolivia': 'bo',
    'Kosovo': 'xk', 'Guinea': 'gn', 'Montenegro': 'me', 'Curaçao': 'cw',
    'Haiti': 'ht', 'New Zealand': 'nz', 'New Caledonia': 'nc',
}

def get_url_drapeau(equipe):
    """Retourne l'URL du drapeau via flagcdn.com (gratuit, fiable)."""
    code = codes_iso.get(equipe, 'un')  # 'un' = drapeau ONU par défaut
    return f"https://flagcdn.com/w80/{code}.png"

# FONCTIONS UTILITAIRES

def get_forme_attaque(equipe, historique):
    """Moyenne de buts marqués sur les 5 derniers matchs."""
    matchs_dom = historique[historique['home_team'] == equipe]['home_score']
    matchs_ext = historique[historique['away_team'] == equipe]['away_score']
    tous_les_buts = pd.concat([matchs_dom, matchs_ext])
    if len(tous_les_buts) == 0:
        return 0.0
    return round(tous_les_buts.tail(5).mean(), 2)

def get_forme_defense(equipe, historique):
    """Moyenne de buts encaissés sur les 5 derniers matchs."""
    buts_encaisses_dom = historique[historique['home_team'] == equipe]['away_score']
    buts_encaisses_ext = historique[historique['away_team'] == equipe]['home_score']
    tous_buts_encaisses = pd.concat([buts_encaisses_dom, buts_encaisses_ext])
    if len(tous_buts_encaisses) == 0:
        return 1.0
    return round(tous_buts_encaisses.tail(5).mean(), 2)

def get_rang_fifa(equipe):
    """Retourne le rang FIFA d'une équipe (1 = meilleure)."""
    if equipe not in points_fifa:
        return None
    points_tries = sorted(points_fifa.values(), reverse=True)
    return points_tries.index(points_fifa[equipe]) + 1

# GÉNÉRATION DES PRÉDICTIONS

@st.cache_data
def generer_predictions():
    """Calcule les prédictions pour les 72 matchs de la Coupe du Monde."""
    resultats = []
    pays_hotes = ['United States', 'Canada', 'Mexico']

    for _, match in matchs_futurs.iterrows():
        equipe_dom = match['home_team']
        equipe_ext = match['away_team']

        # Calcul des 7 features
        forme_att_dom = get_forme_attaque(equipe_dom, historique)
        forme_att_ext = get_forme_attaque(equipe_ext, historique)
        forme_def_dom = get_forme_defense(equipe_dom, historique)
        forme_def_ext = get_forme_defense(equipe_ext, historique)
        fifa_dom = points_fifa.get(equipe_dom, 1200)
        fifa_ext = points_fifa.get(equipe_ext, 1200)

        match_neutre = 1
        if equipe_dom in pays_hotes:
            match_neutre = 0

        donnees_match = pd.DataFrame([[
            forme_att_dom, forme_att_ext,
            forme_def_dom, forme_def_ext,
            fifa_dom, fifa_ext,
            match_neutre
        ]], columns=[
            'forme_attaque_domicile', 'forme_attaque_exterieur',
            'forme_defense_domicile', 'forme_defense_exterieur',
            'points_fifa_domicile', 'points_fifa_exterieur',
            'match_neutre'
        ])

        probas = modele.predict_proba(donnees_match)[0]
        classes = encoder.inverse_transform(modele.classes_)
        probas_dict = dict(zip(classes, probas))
        pronostic = max(probas_dict, key=probas_dict.get)

        resultats.append({
            'date': match['date'],
            'equipe_dom': equipe_dom,
            'equipe_ext': equipe_ext,
            'pronostic': pronostic,
            'proba_1': probas_dict.get('1', 0) * 100,
            'proba_N': probas_dict.get('N', 0) * 100,
            'proba_2': probas_dict.get('2', 0) * 100,
        })

    return pd.DataFrame(resultats)

# Génération des prédictions
df_predictions = generer_predictions()

# Calcul des pronos haute confiance (proba max >= 50%)
df_predictions['proba_max'] = df_predictions[['proba_1', 'proba_N', 'proba_2']].max(axis=1)
pronos_haute_confiance = int((df_predictions['proba_max'] >= 50).sum())

# HEADER

st.title("⚽ Pronos Foot IA", anchor=False)
st.markdown(
    '<p style="color: #94A3B8; font-size: 14px; margin-top: -10px;">Tous les pronostics de la Coupe du Monde 2026</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<hr style="border: none; height: 2px; background: linear-gradient(90deg, #A78BFA, #22D3EE); margin: 16px 0;">',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# CALCULS POUR LES STATS

date_cdm = datetime(2026, 6, 11)
date_aujourd_hui = datetime.now()
jours_restants = max(0, (date_cdm - date_aujourd_hui).days)
nombre_matchs = len(matchs_futurs)

# CARTES DE STATS

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Matchs analysés", value=nombre_matchs)

with col2:
    st.metric(label="Pronos haute confiance", value=pronos_haute_confiance)

with col3:
    st.metric(label="Coup d'envoi", value=f"J - {jours_restants}")

style_metric_cards(
    background_color="#141B2D",
    border_left_color="#A78BFA",
    border_color="#1E293B",
    border_radius_px=10
)

st.markdown("<br>", unsafe_allow_html=True)

# ONGLETS

tab1, tab2, tab3 = st.tabs(["📅 Les matchs", "🎯 Mes pronos", "🧠 L'IA explique"])

# ONGLET 1 : LES MATCHS

with tab1:
    st.subheader("Tous les matchs de la Coupe du Monde 2026")
    st.markdown(
        '<p style="color: #94A3B8;">Les pronostics de l\'IA pour les 72 matchs.</p>',
        unsafe_allow_html=True
    )

    # Pour chaque match, on génère une carte stylée
    for _, match in df_predictions.iterrows():

        # Récupération des infos
        eq_dom = match['equipe_dom']
        eq_ext = match['equipe_ext']
        url_drap_dom = get_url_drapeau(eq_dom)
        url_drap_ext = get_url_drapeau(eq_ext)
        rang_dom = get_rang_fifa(eq_dom)
        rang_ext = get_rang_fifa(eq_ext)
        p1 = match['proba_1']
        pN = match['proba_N']
        p2 = match['proba_2']

        # Texte du pronostic
        if match['pronostic'] == '1':
            pronostic_texte = f"Victoire {eq_dom}"
        elif match['pronostic'] == '2':
            pronostic_texte = f"Victoire {eq_ext}"
        else:
            pronostic_texte = "Match nul"

        # Couleurs des barres
        couleur_1 = "linear-gradient(90deg, #A78BFA, #22D3EE)" if match['pronostic'] == '1' else "#334155"
        couleur_N = "#FBBF24" if match['pronostic'] == 'N' else "#334155"
        couleur_2 = "linear-gradient(90deg, #A78BFA, #22D3EE)" if match['pronostic'] == '2' else "#334155"

        # Texte des rangs FIFA
        rang_dom_txt = f"{rang_dom}ème mondial" if rang_dom else "Non classé"
        rang_ext_txt = f"{rang_ext}ème mondial" if rang_ext else "Non classé"

        # Affichage de la carte
        st.markdown(f"""
        <div style="background: #141B2D; padding: 20px; border-radius: 12px;
                    border: 0.5px solid #1E293B; margin-bottom: 16px;
                    position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px;
                        background: linear-gradient(90deg, #A78BFA, #22D3EE);"></div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <span style="color: #94A3B8; font-size: 12px;">📅 {match['date']}</span>
                <span style="background: rgba(34,211,238,0.15); color: #22D3EE; padding: 3px 10px;
                             border-radius: 6px; font-size: 10px; font-weight: 500;
                             border: 0.5px solid rgba(34,211,238,0.3);">COUPE DU MONDE 2026</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px;">
                <div style="display: flex; align-items: center; gap: 12px; flex: 1;">
                    <img src="{url_drap_dom}" style="width: 40px; height: auto; border-radius: 3px; border: 0.5px solid #1E293B;">
                    <div>
                        <div style="color: #F8FAFC; font-weight: 600; font-size: 16px;">{eq_dom}</div>
                        <div style="color: #94A3B8; font-size: 11px;">{rang_dom_txt}</div>
                    </div>
                </div>
                <div style="text-align: center; color: #475569; font-size: 12px; font-weight: 500;">VS</div>
                <div style="display: flex; align-items: center; gap: 12px; flex: 1; justify-content: flex-end;">
                    <div style="text-align: right;">
                        <div style="color: #F8FAFC; font-weight: 600; font-size: 16px;">{eq_ext}</div>
                        <div style="color: #94A3B8; font-size: 11px;">{rang_ext_txt}</div>
                    </div>
                    <img src="{url_drap_ext}" style="width: 40px; height: auto; border-radius: 3px; border: 0.5px solid #1E293B;">                </div>
            </div>
            <div style="background: #0A0E1A; padding: 14px; border-radius: 8px; border: 0.5px solid #1E293B;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <span style="color: #94A3B8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">🤖 L'IA prédit</span>
                    <span style="background: linear-gradient(135deg, #A78BFA, #22D3EE); color: white;
                                 padding: 3px 10px; border-radius: 6px; font-size: 11px;
                                 font-weight: 500; margin-left: auto;">{pronostic_texte}</span>
                </div>
                <div style="display: flex; gap: 4px; height: 8px;">
                    <div style="flex: {p1}; background: {couleur_1}; border-radius: 4px;"></div>
                    <div style="flex: {pN}; background: {couleur_N}; border-radius: 4px;"></div>
                    <div style="flex: {p2}; background: {couleur_2}; border-radius: 4px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px;">
                    <span style="color: #94A3B8;">{eq_dom} <strong style="color: #F8FAFC;">{p1:.0f}%</strong></span>
                    <span style="color: #FBBF24;">Nul <strong>{pN:.0f}%</strong></span>
                    <span style="color: #A78BFA;">{eq_ext} <strong>{p2:.0f}%</strong></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ONGLET 2 : MES PRONOS

with tab2:
    st.subheader("Mes pronostics")
    st.markdown(
        '<p style="color: #94A3B8;">Saisis tes propres pronostics et compare-toi à l\'IA.</p>',
        unsafe_allow_html=True
    )
    st.info("🚧 Cet onglet sera disponible bientôt !")

# ONGLET 3 : L'IA EXPLIQUE

with tab3:
    st.subheader("Comment l'IA fait ses prédictions")
    st.markdown(
        '<p style="color: #94A3B8;">Découvre les coulisses du modèle.</p>',
        unsafe_allow_html=True
    )
    st.info("🚧 Cet onglet sera disponible bientôt !")