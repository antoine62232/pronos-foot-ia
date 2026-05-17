# ============================================================
# FICHIER : components/ia_explain.py
# RÔLE    : Onglet "L'IA explique" — pédagogie & insights
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def afficher_onglet_ia_explique(df_predictions):
    """
    Affiche le contenu complet de l'onglet 'L'IA explique'.

    4 sections :
        1. Performance du modèle
        2. Importance des features
        3. Distribution des pronostics
        4. Explication pédagogique

    Paramètre :
        df_predictions (DataFrame) : Les 72 prédictions générées

    Aucun retour. Effet de bord : affiche le contenu dans Streamlit.
    """

    st.subheader("Comment l'IA fait ses prédictions")
    st.markdown(
        '<p style="color: #94A3B8;">Découvre les coulisses du modèle d\'intelligence artificielle.</p>',
        unsafe_allow_html=True
    )

    # On affiche chaque section une par une
    _section_performance()
    _section_importance_features()
    _section_distribution_pronostics(df_predictions)
    _section_explication_pedagogique()


# ============================================================
# SECTION 1 : Performance du modèle
# ============================================================

def _section_performance():
    """Affiche la précision du modèle avec contexte explicatif."""

    st.markdown("### 📊 Performance du modèle")

    # 3 cartes côte à côte pour comparer
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Pur hasard",
            value="33%",
            help="Si on devinait au hasard entre 1, N, 2 → 1 chance sur 3"
        )

    with col2:
        # On met en avant notre modèle
        st.metric(
            label="🤖 Notre modèle XGBoost",
            value="51%",
            delta="+18 points / hasard",
            help="Précision mesurée sur 5 040 matchs de test inconnus du modèle"
        )

    with col3:
        st.metric(
            label="Bookmakers pros",
            value="52-55%",
            help="Référence du secteur (sites de paris professionnels)"
        )

    # Note explicative
    st.info(
        "💡 **Ce que ça veut dire** : sur 100 matchs, notre modèle prédit correctement "
        "le bon résultat (victoire domicile, nul, ou victoire extérieur) **51 fois**. "
        "C'est presque le niveau des bookmakers professionnels qui ont accès à des "
        "données 10x plus complètes."
    )

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 2 : Importance des features
# ============================================================

def _section_importance_features():
    """Affiche le graphique d'importance des 7 features."""

    st.markdown("### 🎯 Sur quoi se base l'IA pour décider")

    # Données issues de l'entraînement réel du modèle
    # (extraites de notre exécution de modele.py)
    features_data = pd.DataFrame({
        'Feature': [
            'Points FIFA extérieur',
            'Points FIFA domicile',
            'Match neutre',
            'Forme défense extérieur',
            'Forme défense domicile',
            'Forme attaque extérieur',
            'Forme attaque domicile'
        ],
        'Importance (%)': [24.6, 24.1, 15.4, 10.7, 10.7, 7.5, 7.0],
        'Catégorie': [
            'Niveau FIFA', 'Niveau FIFA',
            'Contexte',
            'Défense', 'Défense',
            'Attaque', 'Attaque'
        ]
    })

    # Graphique en barres horizontales avec Plotly
    fig = px.bar(
        features_data,
        x='Importance (%)',
        y='Feature',
        color='Catégorie',
        orientation='h',
        color_discrete_map={
            'Niveau FIFA': '#A78BFA',
            'Contexte'   : '#22D3EE',
            'Défense'    : '#34D399',
            'Attaque'    : '#FBBF24'
        }
    )

    # On personnalise le graphique pour qu'il colle au thème dark
    fig.update_layout(
        plot_bgcolor='#141B2D',
        paper_bgcolor='#141B2D',
        font=dict(color='#F8FAFC', family='sans-serif'),
        yaxis=dict(autorange='reversed', gridcolor='#1E293B'),
        xaxis=dict(gridcolor='#1E293B'),
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # use_container_width fait que le graphique s'adapte à la largeur de la page
    st.plotly_chart(fig, use_container_width=True)

    # Insights derrière le graphique
    st.success(
        "🏆 **Insights clés** :\n\n"
        "- Les **points FIFA** représentent **49%** du pouvoir prédictif total\n"
        "- La **défense** (21%) compte **plus que l'attaque** (15%) — le vieux dicton est statistiquement vrai !\n"
        "- Le **terrain neutre** (15%) est crucial pour les tournois comme la Coupe du Monde"
    )

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 3 : Distribution des pronostics
# ============================================================

def _section_distribution_pronostics(df_predictions):
    """Affiche la répartition des 72 pronostics."""

    st.markdown("### 📈 Répartition des 72 pronostics")

    # On compte combien de fois chaque résultat est prédit
    distribution = df_predictions['pronostic'].value_counts()

    # On prépare les données pour le graphique
    labels_map = {'1': 'Victoire domicile', '2': 'Victoire extérieur', 'N': 'Match nul'}
    colors_map = {'1': '#A78BFA', '2': '#22D3EE', 'N': '#FBBF24'}

    labels  = [labels_map.get(p, p) for p in distribution.index]
    values  = distribution.values
    colors  = [colors_map.get(p, '#94A3B8') for p in distribution.index]

    # Graphique en donut (camembert avec trou au centre)
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
        # Texte au centre du donut
        annotations=[dict(
            text=f"<b>{len(df_predictions)}</b><br>matchs",
            x=0.5, y=0.5,
            font=dict(size=18, color='#F8FAFC'),
            showarrow=False
        )]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SECTION 4 : Explication pédagogique
# ============================================================

def _section_explication_pedagogique():
    """Explique en termes simples comment fonctionne le modèle."""

    st.markdown("### 🤖 Comment fonctionne notre IA ?")

    # On utilise un expander pour ne pas surcharger la page
    # L'utilisateur clique pour déplier le contenu

    with st.expander("📚 Voir l'explication détaillée"):
        st.markdown("""
        **Notre modèle est un XGBoost** (eXtreme Gradient Boosting).

        Imagine **des centaines d'arbres de décision** qui votent ensemble pour prédire un match :

        - 🌳 **Arbre 1** : *"Si les points FIFA de l'équipe domicile > 1700, alors victoire à 60%"*
        - 🌳 **Arbre 2** : *"Si la défense extérieur encaisse plus de 2 buts en moyenne, alors victoire domicile à 70%"*
        - 🌳 **Arbre 3** : *"Si match neutre + petite différence FIFA, alors nul à 40%"*
        - 🌳 ... et **197 autres arbres** qui posent leurs propres questions

        **Chaque arbre donne sa probabilité, et l'IA fait la moyenne** pour produire le résultat final.

        ---

        **Données utilisées :**
        - 📊 **25 196 matchs** internationaux depuis l'an 2000
        - 🎯 **7 variables** par match (forme, points FIFA, terrain neutre)
        - 🔄 **80%** des matchs pour l'apprentissage, **20%** pour tester la fiabilité

        **Ce que l'IA NE peut PAS prédire :**
        - Une blessure de dernière minute
        - Une décision tactique du sélectionneur
        - Un événement exceptionnel (carton rouge précoce, but de la main...)

        C'est pour ça qu'aucune IA ne dépassera jamais 60% de précision sur le foot.
        **Le football reste un sport profondément humain et imprévisible**, et c'est ce qui le rend passionnant.
        """)