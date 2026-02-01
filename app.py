import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="Credit Analytics Pro",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Charger le modèle
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_svm.pkl')
    except:
        return None

# Charger les données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("AER_credit_card_data.csv")
        # Encoder les colonnes binaires
        binary_cols = ['card', 'owner', 'selfemp']
        for col in binary_cols:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        return df
    except:
        return None

model = load_model()
df = load_data()

# Barre latérale
st.sidebar.markdown("""
<div class="sidebar-header">
    <h1>💳 Credit Analytics</h1>
    <p style="color: #b8860b; font-size: 14px; margin-top: -10px;">Plateforme d'Analyse</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "🔮 Prédiction", "📈 Analyse"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Description de l'application
st.sidebar.markdown("""
**📊 À propos**

Cette application analyse les données de crédit clients et prédit leurs dépenses mensuelles à l'aide d'un modèle SVM optimisé.
""")

# Page d'accueil
if page == "🏠 Accueil":
    st.markdown("""
    <div class="hero-section">
        <div style="display: flex; align-items: center; justify-content: center; gap: 30px;">
            <div style="font-size: 80px;">🏦</div>
            <div class="hero-content">
                <h1 class="hero-title">Credit Analytics Pro</h1>
                <p class="hero-subtitle">Plateforme d'analyse et de prédiction des dépenses clients par Magne Dassi</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistiques clés
    st.markdown("""
    <div class="section-header">
        <h2>📊 Statistiques Clés</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="metric-icon">👥</div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Clients Total</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_exp = df['expenditure'].mean()
            st.markdown(f"""
            <div class="metric-card metric-card-green">
                <div class="metric-icon">💵</div>
                <div class="metric-value">${avg_exp:.0f}</div>
                <div class="metric-label">Dépenses Moyennes</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            approval_rate = (df['card'].sum() / len(df)) * 100
            st.markdown(f"""
            <div class="metric-card metric-card-gold">
                <div class="metric-icon">✅</div>
                <div class="metric-value">{approval_rate:.1f}%</div>
                <div class="metric-label">Taux d'Approbation</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_age = df['age'].mean()
            st.markdown(f"""
            <div class="metric-card metric-card-purple">
                <div class="metric-icon">🎂</div>
                <div class="metric-value">{avg_age:.0f} ans</div>
                <div class="metric-label">Âge Moyen</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Graphiques
        st.markdown("""
        <div class="section-header">
            <h2>📈 Visualisations</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des dépenses
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['expenditure'], bins=50, color='#1a365d', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution des Dépenses Mensuelles', fontsize=16, fontweight='bold')
            ax.set_xlabel('Dépenses ($)', fontsize=12)
            ax.set_ylabel('Nombre de clients', fontsize=12)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Revenus vs dépenses
            sample_df = df.sample(300).sort_values('income')
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(sample_df['income'], sample_df['expenditure'], 
                               c=sample_df['expenditure'], cmap='viridis', 
                               alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            ax.plot(sample_df['income'], sample_df['expenditure'], 
                   color='#4299e1', linewidth=2, alpha=0.8)
            ax.set_title('Relation Revenu vs Dépenses', fontsize=16, fontweight='bold')
            ax.set_xlabel('Revenu (10k $)', fontsize=12)
            ax.set_ylabel('Dépenses ($)', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Dépenses')
            st.pyplot(fig)
        
        # Profils de dépenses
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Catégoriser les dépenses
        df['categorie'] = pd.cut(df['expenditure'], 
                                  bins=[0, 50, 200, 500, 5000], 
                                  labels=['Faible', 'Modéré', 'Élevé', 'Très élevé'])
        
        cat_counts = df['categorie'].value_counts()
        categories_ordered = ['Faible', 'Modéré', 'Élevé', 'Très élevé']
        counts_ordered = [cat_counts.get(cat, 0) for cat in categories_ordered]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(categories_ordered, counts_ordered, 
                     color=['#48bb78', '#4299e1', '#ed8936', '#f56565'], 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title('Répartition des Profils de Dépenses', fontsize=16, fontweight='bold')
        ax.set_xlabel('Catégorie de Dépenses', fontsize=12)
        ax.set_ylabel('Nombre de Clients', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les étiquettes de valeur sur les barres
        for bar, count in zip(bars, counts_ordered):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        
    else:
        st.error("⚠️ Impossible de charger les données. Vérifiez que 'AER_credit_card_data.csv' est présent.")

# Page de prédiction
elif page == "🔮 Prédiction":
    st.markdown("""
    <div class="page-header">
        <h1>🔮 Prédiction des Dépenses</h1>
        <p>Estimez les dépenses mensuelles d'un client grâce à notre modèle SVM optimisé</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Modèle non trouvé. Veuillez exécuter le notebook d'entraînement et placer 'model_svm.pkl' dans le dossier.")
        st.stop()
    
    # Formulaire de prédiction
    with st.form("prediction_form"):
        st.markdown("### 📝 Informations du Client")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💰 Informations Financières**")
            income = st.number_input("Revenu Annuel (en 10k $)", 0.5, 20.0, 4.5, step=0.1, 
                                    help="Revenu annuel du client en dizaines de milliers de dollars")
            share = st.number_input("Part du Revenu (Ratio)", 0.0, 1.0, 0.05, format="%.4f",
                                   help="Ratio de partage du revenu")
            majorcards = st.number_input("Nombre de Cartes Majeures", 0, 5, 1,
                                        help="Nombre de cartes de crédit majeures détenues")
        
        with col2:
            st.markdown("**👤 Informations Personnelles**")
            age = st.number_input("Âge", 18, 100, 35,
                                 help="Âge du client en années")
            dependents = st.number_input("Personnes à Charge", 0, 10, 1,
                                        help="Nombre de personnes à charge")
            months = st.number_input("Mois à l'Adresse Actuelle", 0, 500, 24,
                                    help="Durée de résidence à l'adresse actuelle")
        
        with col3:
            st.markdown("**🏦 Profil Crédit**")
            reports = st.number_input("Rapports Négatifs", 0, 20, 0,
                                     help="Nombre de rapports de crédit négatifs")
            active = st.number_input("Comptes Actifs", 0, 30, 5,
                                    help="Nombre de comptes de crédit actifs")
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Statuts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            card_input = st.selectbox("💳 Demande de Carte Acceptée ?", ["Oui", "Non"], 
                                     help="La demande de carte de crédit a-t-elle été approuvée ?")
        with col2:
            owner_input = st.selectbox("🏠 Propriétaire ?", ["Oui", "Non"],
                                      help="Le client est-il propriétaire de sa résidence ?")
        with col3:
            selfemp_input = st.selectbox("💼 Indépendant ?", ["Oui", "Non"],
                                        help="Le client est-il travailleur indépendant ?")
        
        # Convertir les entrées
        card = 1 if card_input == "Oui" else 0
        owner = 1 if owner_input == "Oui" else 0
        selfemp = 1 if selfemp_input == "Oui" else 0
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Lancer la Prédiction", use_container_width=True)
    
    # Résultat
    if submitted:
        # Préparer les caractéristiques
        features = np.array([[card, reports, age, income, share, owner, selfemp, dependents, months, majorcards, active]])
        
        # Prédire
        prediction = model.predict(features)[0]
        
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
            <h2>✨ Résultat de l'Analyse</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Affichage du résultat
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Déterminer la catégorie
            if prediction < 50:
                category = "Faibles"
                color = "#48bb78"
                icon = "📉"
                message = "Ce client présente un profil de **faibles dépenses**. Idéal pour des produits d'entrée de gamme."
            elif prediction < 200:
                category = "Modérées"
                color = "#4299e1"
                icon = "📊"
                message = "Ce client a un profil de dépenses **modérées**. Bon candidat pour des produits standards."
            elif prediction < 500:
                category = "Élevées"
                color = "#ed8936"
                icon = "📈"
                message = "Ce client présente des dépenses **élevées**. Excellent candidat pour des produits premium."
            else:
                category = "Très Élevées"
                color = "#f56565"
                icon = "🚀"
                message = "Ce client a des dépenses **très élevées**. Profil VIP pour produits de luxe."
            
            st.markdown(f"""
            <div class="prediction-result" style="border-left: 5px solid {color};">
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 60px;">{icon}</div>
                </div>
                <div class="prediction-value" style="color: {color};">
                    ${prediction:.2f}
                </div>
                <div class="prediction-label">
                    Dépenses Mensuelles Estimées
                </div>
                <div style="margin-top: 20px; padding: 15px; background: #f7fafc; border-radius: 8px; text-align: left;">
                    <p style="margin: 0; color: #2d3748; font-size: 14px; line-height: 1.6;">
                        <b style="color: {color};">Catégorie: {category}</b><br><br>
                        {message}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Graphique de comparaison
        if df is not None:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("### 📊 Positionnement par Rapport aux Autres Clients")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df['expenditure'], bins=50, alpha=0.7, color='#cbd5e0', edgecolor='black', label='Tous les clients')
            ax.axvline(x=prediction, color=color, linestyle='--', linewidth=3, label=f'Prédiction: ${prediction:.2f}')
            ax.set_title('Position de la Prédiction dans la Distribution Globale', fontsize=16, fontweight='bold')
            ax.set_xlabel('Dépenses ($)', fontsize=12)
            ax.set_ylabel('Nombre de clients', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

# Page d'analyse
elif page == "📈 Analyse":
    st.markdown("""
    <div class="page-header">
        <h1>📈 Analyse Approfondie</h1>
        <p>Explorez les données et les insights de notre modèle</p>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        # Analyse par catégorie
        st.markdown("### 📊 Analyse par Catégorie")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dépenses par statut de propriétaire
            owner_exp = df.groupby('owner')['expenditure'].mean()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(['Locataire', 'Propriétaire'], [owner_exp[0], owner_exp[1]], 
                         color=['#4299e1', '#1a365d'], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_title('Dépenses Moyennes: Locataire vs Propriétaire', fontsize=16, fontweight='bold')
            ax.set_ylabel('Dépenses Moyennes ($)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Ajouter les étiquettes de valeur
            for bar, value in zip(bars, [owner_exp[0], owner_exp[1]]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                       f'${value:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
        
        with col2:
            # Dépenses par tranche d'âge
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], 
                                     labels=['<30', '30-40', '40-50', '50+'])
            age_exp = df.groupby('age_group', observed=True)['expenditure'].mean()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(['<30', '30-40', '40-50', '50+'], 
                         [age_exp.get('<30', 0), age_exp.get('30-40', 0), 
                          age_exp.get('40-50', 0), age_exp.get('50+', 0)], 
                         color=['#48bb78', '#4299e1', '#ed8936', '#f56565'], 
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_title('Dépenses Moyennes par Tranche d\'Âge', fontsize=16, fontweight='bold')
            ax.set_xlabel('Tranche d\'Âge', fontsize=12)
            ax.set_ylabel('Dépenses Moyennes ($)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Ajouter les étiquettes de valeur
            values = [age_exp.get('<30', 0), age_exp.get('30-40', 0), 
                     age_exp.get('40-50', 0), age_exp.get('50+', 0)]
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                       f'${value:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
    else:
        st.error("⚠️ Impossible de charger les données pour l'analyse.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 20px;">
    <p style="margin: 0; font-size: 14px;">
        💳 <b>Credit Analytics Pro</b> - Powered by Magne Dassi Christ Laure | 
        © 2026 | Modèle: SVM Optimisé
    </p>
</div>
""", unsafe_allow_html=True)