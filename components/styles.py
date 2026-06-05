import streamlit as st

def appliquer_styles():
    """
    Applique le thème 'Data Sport Pro' (dark moderne).

    Cette fonction injecte le CSS personnalisé dans la page Streamlit.
    À appeler UNE FOIS au début de app.py, juste après st.set_page_config().
    """

    st.markdown("""
    <style>
        /* @import DOIT etre la toute premiere ligne du CSS pour etre valide */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&family=Outfit:wght@600;700;800&family=Bricolage+Grotesque:wght@600;700;800&family=Playfair+Display:wght@700;800&display=swap');

        /* ===== FOND PRINCIPAL ===== */
        .stApp {
            background-color: #0A0E1A;
        }

        /* ===== TEXTES ===== */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #F8FAFC !important;
        }

        /* ===== POLICES ===== */
        /* Police generale de l'app */
        .stApp, .stApp p, .stApp label, .stApp button, .stApp div {
            font-family: 'Inter', sans-serif;
        }
        /* Police des grands titres */
        h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
        }

        /* ===== ONGLETS ===== */
        /* La barre qui contient tous les onglets */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #141B2D;
            padding: 4px;
            border-radius: 10px;
            gap: 4px;
            border: 0.5px solid #1E293B;
        }

        /* Chaque onglet individuel (etat normal) */
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #94A3B8;
            border-radius: 8px;
            padding: 8px 20px;
            font-size: 14px;
            font-weight: 500;
        }

        /* L'onglet selectionne (etat actif) */
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(34,211,238,0.1)) !important;
            color: #A78BFA !important;
        }

        /* ===== CORRECTION DU TRAIT D'ONGLET ===== */
        /* Le trait sous l'onglet actif : rouge par defaut -> violet (ton theme) */
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #A78BFA !important;
        }
        /* La fine ligne sous toute la rangee d'onglets */
        .stTabs [data-baseweb="tab-border"] {
            background-color: #1E293B !important;
        }

        /* ===== ELEMENTS STREAMLIT A CACHER ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
                
        /* ===== RESSERRER L'ESPACE EN HAUT ===== */
        .block-container {
            padding-top: 2rem !important;
        }
                
    </style>
    """, unsafe_allow_html=True)


def afficher_header():
    """Affiche l'en-tête : titre + sous-titre centres + ligne courte centree."""

    police_titre = "Bricolage Grotesque"

    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style="font-family: '{police_titre}', sans-serif; font-size: 2.9rem;
                       font-weight: 800; margin-bottom: 0;">
                ⚽ Pronos Foot IA
            </h1>
            <p style="color: #94A3B8; font-size: 15px; margin-top: 4px;">
                Tous les pronostics de la Coupe du Monde 2026
            </p>
            <!-- Ligne courte et centree sous le titre -->
            <div style="width: 90px; height: 3px; margin: 14px auto 0 auto;
                        border-radius: 2px;
                        background: linear-gradient(90deg, #A78BFA, #22D3EE);"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)