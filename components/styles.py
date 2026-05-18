import streamlit as st

def appliquer_styles():
    """
    Applique le thème 'Data Sport Pro' (dark moderne).

    Cette fonction injecte le CSS personnalisé dans la page Streamlit.
    À appeler UNE FOIS au début de app.py, juste après st.set_page_config().

    Aucun paramètre, aucun retour. Effet de bord : applique le CSS.
    """

    # On utilise st.markdown avec unsafe_allow_html=True
    # car on injecte du HTML/CSS qui n'est pas du markdown standard

    st.markdown("""
    <style>
        /* Fond principal en bleu nuit profond */
        .stApp {
            background-color: #0A0E1A;
        }

        /* Tous les textes en blanc casse */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #F8FAFC !important;
        }

        /* Onglets stylés avec dégradé violet/cyan quand actifs */
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

        /* On cache les elements Streamlit par defaut */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* On cache le bouton "Manage app" de Streamlit Cloud */
        .viewerBadge_container__r5tak {
            display: none !important;
        }

        [data-testid="stToolbar"] {
            display: none !important;
        }

        .stAppDeployButton {
            display: none !important;
        }

        [class*="viewerBadge"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)


def afficher_header():
    """
    Affiche l'en-tête de l'application : titre + sous-titre + ligne dégradée.

    Cette fonction encapsule la création visuelle du header pour ne pas
    polluer app.py avec du HTML.

    Aucun paramètre, aucun retour. Effet de bord : affiche le header.
    """

    # Titre principal — anchor=False enlève l'icône de lien à côté du titre
    st.title("⚽ Pronos Foot IA", anchor=False)

    # Sous-titre en gris clair
    st.markdown(
        '<p style="color: #94A3B8; font-size: 14px; margin-top: -10px;">'
        'Tous les pronostics de la Coupe du Monde 2026</p>',
        unsafe_allow_html=True
    )

    # Ligne dégradée violet → cyan
    st.markdown(
        '<hr style="border: none; height: 2px; '
        'background: linear-gradient(90deg, #A78BFA, #22D3EE); '
        'margin: 16px 0;">',
        unsafe_allow_html=True
    )

    # Un peu d'espace avant le contenu
    st.markdown("<br>", unsafe_allow_html=True)