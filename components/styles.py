import streamlit as st
import streamlit.components.v1 as components

def appliquer_styles():
    """
    Applique le thème 'Data Sport Pro' et la logique de scroll dynamique.
    """

    st.markdown("""
    <style>
        /* @import DOIT etre la toute premiere ligne du CSS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&family=Bricolage+Grotesque:wght@600;700;800&display=swap');

        /* ===== FOND PRINCIPAL ===== */
        .stApp {
            background-color: #0A0E1A !important;
        }

        /* ===== TEXTES ===== */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #F8FAFC !important;
        }

        /* ===== POLICES ===== */
        .stApp, .stApp p, .stApp label, .stApp button, .stApp div {
            font-family: 'Inter', sans-serif;
        }
        
        h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
        }

        /* ===== ELEMENTS STREAMLIT A CACHER ===== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
                
        /* ===== RESSERRER L'ESPACE EN HAUT ===== */
        .block-container {
            padding-top: 2rem !important;
        }
        
        /* ===== NAVBAR STYLE ONGLETS (uniquement la navigation principale) ===== */
        .st-key-nav div[role="radiogroup"] {
            background-color: transparent !important;
            padding: 0 !important;
            border: none !important;
            border-bottom: 1px solid #1E293B !important;   /* la ligne de separation du bas */
            border-radius: 0 !important;
            box-shadow: none !important;
            display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
            gap: 8px !important; overflow-x: auto !important;
        }
        .st-key-nav div[role="radiogroup"]::-webkit-scrollbar { display: none !important; }

        .st-key-nav label[data-baseweb="radio"] {
            background-color: transparent !important;
            padding: 10px 18px !important;
            border-radius: 8px 8px 0 0 !important;          /* coins arrondis en haut seulement */
            cursor: pointer !important;
            flex: 1 1 auto !important;                       /* largeur selon le texte, pas etire */
            display: flex !important;
            justify-content: center !important; align-items: center !important;
            margin: 0 !important;
            border: none !important;
            border-bottom: 2px solid transparent !important; /* le futur soulignement, invisible par defaut */
            transition: all 0.2s ease !important;
        }
        .st-key-nav label[data-baseweb="radio"] > div:first-child:not([data-testid="stMarkdownContainer"]) {
            display: none !important;                         /* on masque le rond natif du radio */
        }
        .st-key-nav label[data-baseweb="radio"] p {
            color: #94A3B8 !important; font-size: 15px !important; font-weight: 500 !important;
            margin: 0 !important; white-space: nowrap !important;
        }
        .st-key-nav label[data-baseweb="radio"]:hover {
            background-color: rgba(167,139,250,0.05) !important;
        }
        .st-key-nav label[data-baseweb="radio"]:has(input:checked) {
            background: rgba(167,139,250,0.08) !important;    /* leger fond sur l'onglet actif */
            border-bottom: 2px solid #A78BFA !important;      /* le soulignement violet */
        }
        .st-key-nav label[data-baseweb="radio"]:has(input:checked) p {
            color: #F8FAFC !important; font-weight: 600 !important;  /* texte blanc et gras quand actif */
        }

        /* --- FLECHES INDICATRICES DE SCROLL (uniquement la navigation) --- */
        @media (max-width: 768px) {
            .st-key-nav { position: relative !important; }
            .st-key-nav div[role="radiogroup"] { position: relative !important; }

            .st-key-nav[data-scroll="start"]::after,
            .st-key-nav[data-scroll="middle"]::after {
                content: "»"; position: absolute; right: 0; bottom: 0; height: 44px;
                display: flex; align-items: center; padding: 0 10px 0 30px;
                background: linear-gradient(to right, rgba(20,27,45,0), rgba(20,27,45,1) 60%);
                color: #A78BFA; font-size: 20px; font-weight: bold; pointer-events: none;
                border-radius: 0 8px 8px 0; z-index: 10;
            }

            .st-key-nav[data-scroll="end"]::before,
            .st-key-nav[data-scroll="middle"]::before {
                content: "«"; position: absolute; left: 0; bottom: 0; height: 44px;
                display: flex; align-items: center; padding: 0 30px 0 10px;
                background: linear-gradient(to left, rgba(20,27,45,0), rgba(20,27,45,1) 60%);
                color: #A78BFA; font-size: 20px; font-weight: bold; pointer-events: none;
                border-radius: 8px 0 0 8px; z-index: 10;
            }
        }

        /* ========================================================= */
        /* ===== ONGLETS NATIFS (EX: PHASE ÉLIMINATOIRE) =========== */
        /* ========================================================= */
        
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            background-color: #141B2D !important;
            padding: 6px !important;
            border-radius: 12px !important;
            border: 1px solid #1E293B !important;
            display: flex !important;
            width: 100% !important;
            gap: 6px !important;
            overflow-x: auto !important; 
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab"] {
            background-color: transparent !important;
            color: #94A3B8 !important;
            border-radius: 8px !important;
            padding: 10px 5px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            flex: 1 1 0px !important; 
            min-width: 110px !important; 
            display: flex !important;
            justify-content: center !important;
            text-align: center !important;
            white-space: nowrap !important;
            border: 1px solid transparent !important;
            transition: all 0.3s ease !important;
        }

        div[data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(167, 139, 250, 0.15), rgba(34, 211, 238, 0.05)) !important;
            border: 1px solid rgba(167, 139, 250, 0.3) !important;
            box-shadow: 0 4px 12px rgba(167, 139, 250, 0.1) !important;
            color: #A78BFA !important;
            font-weight: 600 !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"], 
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Les fleches de scroll ne concernent que la navbar : on ne surveille que .st-key-nav.
    components.html("""
        <script>
            const doc = window.parent.document;
            function updateScrollIndicators() {
                const radiogroups = doc.querySelectorAll('.st-key-nav > div[role="radiogroup"]');
                radiogroups.forEach(rg => {
                    const parent = rg.parentElement;
                    const isScrollable = rg.scrollWidth > rg.clientWidth + 2; 
                    if (isScrollable) {
                        const isAtStart = rg.scrollLeft <= 2;
                        const isAtEnd = Math.ceil(rg.scrollLeft + rg.clientWidth) >= rg.scrollWidth - 2;
                        parent.setAttribute('data-scroll', isAtStart ? 'start' : (isAtEnd ? 'end' : 'middle'));
                    } else { parent.setAttribute('data-scroll', 'none'); }
                });
            }
            const observer = new MutationObserver(() => {
                const radiogroups = doc.querySelectorAll('.st-key-nav > div[role="radiogroup"]');
                radiogroups.forEach(rg => {
                    if (!rg.dataset.listened) {
                        rg.addEventListener('scroll', updateScrollIndicators);
                        rg.dataset.listened = 'true';
                        setTimeout(updateScrollIndicators, 100); 
                    }
                });
            });
            observer.observe(doc.body, {childList: true, subtree: true});
            window.parent.addEventListener('resize', updateScrollIndicators);
        </script>
    """, height=0, width=0)


def afficher_header():
    """Affiche l'en-tete : titre, sous-titre et trait degrade."""
    police_titre = "Bricolage Grotesque"

    st.markdown(
        f"""<div style="text-align: center;">
<h1 style="font-family: '{police_titre}', sans-serif; font-size: 2.9rem; font-weight: 800; margin-bottom: 0;">⚽ Prediktora : Coupe du Monde 2026</h1>
<p style="color: #94A3B8; font-size: 15px; margin-top: 4px; margin-bottom: 12px;">Prédictions 100% Data &amp; Machine Learning</p>
<div style="width: 90px; height: 3px; margin: 0 auto; border-radius: 2px; background: linear-gradient(90deg, #A78BFA, #22D3EE);"></div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)