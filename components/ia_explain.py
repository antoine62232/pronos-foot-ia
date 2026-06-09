import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.styles import afficher_description_onglet

# FONCTION PRINCIPALE

def afficher_onglet_ia_explique(df_predictions):
    """
    Affiche le contenu complet de l'onglet 'L'IA explique'.

    4 sections :
        1. Performance du modele
        2. Importance des features
        3. Distribution des pronostics
        4. Explication pedagogique

    Parametre :
        df_predictions (DataFrame) : Les 72 predictions generees
    """

    st.subheader("Comment l'IA fait ses prédictions")
    afficher_description_onglet("Sur quels critères l'IA s'appuie pour décider, et le poids de chacun.")

    _section_performance()
    _section_importance_features()
    _section_distribution_pronostics(df_predictions)
    _section_explication_pedagogique()

# SECTION 1 : Performance du modele

def _section_performance():
    """Affiche la precision du modele avec contexte explicatif."""

    st.markdown("### 📊 Performance du modèle")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Pur hasard",
            value="33%",
            help="Si on devinait au hasard entre 1, N, 2 → 1 chance sur 3"
        )

    with col2:
        st.metric(
            label="🤖 Notre modèle XGBoost",
            value="53%",
            delta="+20 points / hasard",
            help="Précision sur les matchs équilibrés (type Coupe du Monde), mesurée sur un backtest de 5 822 matchs jamais vus."
        )

    with col3:
        st.metric(
            label="Bookmakers pros",
            value="52-55%",
            help="Référence du secteur (sites de paris professionnels)"
        )

    st.info(
        "💡 **Ce que ça veut dire** : sur 100 matchs équilibrés (type Coupe du Monde), le modèle "
        "prédit correctement le résultat (victoire, nul ou défaite) **53 fois** — au niveau des bookmakers "
        "professionnels, qui ont pourtant accès à bien plus de données.\n\n"
        "Ce chiffre vient d'un **backtest rigoureux sur 5 822 matchs** : le modèle apprend uniquement sur le "
        "passé, puis on teste ses prédictions sur des matchs qu'il n'a jamais vus (aucune triche avec le futur). "
        "Sur des matchs plus déséquilibrés comme les qualifications, il monte à environ **58%**.\n\n"
        "Et ses probabilités sont **fiables** : quand il annonce 75% de confiance, il a raison environ 76% du temps."
    )

    st.markdown("<br>", unsafe_allow_html=True)

# SECTION 2 : Importance des features

def _section_importance_features():
    """Affiche le graphique d'importance des 11 features (modele Elo)."""

    st.markdown("### 🎯 Sur quoi se base l'IA pour décider")

    # Valeurs reelles du modele de production
    # (extraites via src/importance_features.py)
    features_data = pd.DataFrame({
        'Feature': [
            'Elo extérieur',
            'Elo domicile',
            'Terrain neutre',
            'Défense extérieur (10 matchs)',
            'Défense domicile (10 matchs)',
            'Attaque domicile (10 matchs)',
            'Attaque extérieur (10 matchs)',
            'Défense extérieur (5 matchs)',
            'Attaque extérieur (5 matchs)',
            'Défense domicile (5 matchs)',
            'Attaque domicile (5 matchs)',
        ],
        'Importance (%)': [20.6, 20.2, 14.6, 7.9, 7.5, 5.3, 5.2, 5.2, 4.7, 4.6, 4.4],
        'Catégorie': [
            'Force (Elo)', 'Force (Elo)',
            'Contexte',
            'Défense', 'Défense',
            'Attaque', 'Attaque',
            'Défense',
            'Attaque',
            'Défense',
            'Attaque',
        ]
    })

    fig = px.bar(
        features_data,
        x='Importance (%)',
        y='Feature',
        color='Catégorie',
        orientation='h',
        color_discrete_map={
            'Force (Elo)': '#A78BFA',
            'Contexte'   : '#22D3EE',
            'Défense'    : '#34D399',
            'Attaque'    : '#FBBF24'
        }
    )

    fig.update_layout(
        plot_bgcolor='#141B2D',
        paper_bgcolor='#141B2D',
        font=dict(color='#F8FAFC', family='sans-serif'),
        yaxis=dict(autorange='reversed', gridcolor='#1E293B'),
        xaxis=dict(gridcolor='#1E293B'),
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "🏆 **Ce que l'IA regarde le plus** :\n\n"
        "- La **force des équipes (Elo)** pèse **41%** du pouvoir prédictif — c'est le facteur n°1\n"
        "- La **défense** (25%) compte **plus que l'attaque** (20%) — le vieux dicton est statistiquement vrai !\n"
        "- Le **terrain neutre** (15%) est crucial pour un tournoi comme la Coupe du Monde, où presque toutes les équipes jouent loin de chez elles"
    )

    st.markdown("<br>", unsafe_allow_html=True)

# SECTION 3 : Distribution des pronostics

def _section_distribution_pronostics(df_predictions):
    """Affiche la repartition des 72 pronostics."""

    st.markdown("### 📈 Répartition des 72 pronostics")

    distribution = df_predictions['pronostic'].value_counts()

    labels_map = {'1': 'Victoire domicile', '2': 'Victoire extérieur', 'N': 'Match nul'}
    colors_map = {'1': '#A78BFA', '2': '#22D3EE', 'N': '#FBBF24'}

    labels  = [labels_map.get(p, p) for p in distribution.index]
    values  = distribution.values
    colors  = [colors_map.get(p, '#94A3B8') for p in distribution.index]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='#0A0E1A', width=2)),
        textinfo='label+percent',
        textfont=dict(color='#F8FAFC', size=13),
    )])

    fig.update_layout(
        plot_bgcolor='#141B2D',
        paper_bgcolor='#141B2D',
        font=dict(color='#F8FAFC'),
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[dict(
            text=f"<b>{len(df_predictions)}</b><br>matchs",
            x=0.5, y=0.5,
            font=dict(size=18, color='#F8FAFC'),
            showarrow=False
        )]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

# SECTION 4 : Explication pedagogique

def _section_explication_pedagogique():
    """Explique en termes simples comment fonctionne le modele."""

    st.markdown("### 🤖 Comment fonctionne notre IA ?")

    with st.expander("📚 Voir l'explication détaillée"):
        st.markdown("""
        **Notre modèle est un XGBoost** (eXtreme Gradient Boosting).

        Imagine **des centaines d'arbres de décision** qui votent ensemble pour prédire un match :

        - 🌳 **Arbre 1** : *"Si l'Elo de l'équipe domicile dépasse de 200 points celui de l'adversaire, alors victoire à 65%"*
        - 🌳 **Arbre 2** : *"Si la défense adverse encaisse plus de 2 buts en moyenne, alors victoire domicile à 70%"*
        - 🌳 **Arbre 3** : *"Si match sur terrain neutre et Elo proches, alors match nul à 40%"*
        - 🌳 ... et **environ 200 arbres** au total, chacun avec ses propres questions

        **Chaque arbre donne sa probabilité, et l'IA combine tout** pour produire le résultat final.

        ---

        **Le classement Elo, le cœur du modèle :**
        Plutôt qu'un classement figé, on calcule un **score de force pour chaque équipe**, qui évolue après
        chaque match : battre plus fort que soi rapporte beaucoup de points, battre plus faible en rapporte peu.
        C'est le même principe qu'aux échecs. Ce score est recalculé uniquement à partir du passé, sans jamais
        utiliser le futur.

        ---

        **Données utilisées :**
        - 📊 **25 196 matchs** internationaux depuis l'an 2000
        - 🎯 **11 variables** par match (la forme sur 5 et 10 matchs, l'Elo des deux équipes, le terrain neutre)
        - 🔄 Évaluation **par séparation dans le temps** : on apprend sur les matchs anciens, on teste sur les récents (jamais vus)

        **Ce que l'IA NE peut PAS prédire :**
        - Une blessure de dernière minute
        - Une décision tactique du sélectionneur
        - Un événement exceptionnel (carton rouge précoce, but de la main...)

        C'est pour ça que même les meilleurs modèles **plafonnent autour de 55%** de précision sur le football.
        **Le foot reste un sport profondément humain et imprévisible**, et c'est ce qui le rend passionnant.
        """)