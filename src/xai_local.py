import matplotlib
matplotlib.use('Agg')  # Backend non interactif pour l'export d'images
import matplotlib.pyplot as plt
import shap
from xai_commun import CAS, Xte, m

# 1. Initialisation de l'explainer SHAP
explainer = shap.TreeExplainer(m)

# 2. Identification de la transaction de test la plus suspecte
probabilities = m.predict_proba(Xte)[:, 1]
most_suspicious_idx = Xte.index[probabilities.argmax()]

# 3. Définition des cibles à analyser (cas 'benali' et le cas le plus suspect)
targets = [
    ('benali', CAS),
    (f'test_{most_suspicious_idx}', Xte.loc[[most_suspicious_idx]])
]

# 4. Boucle d'analyse et génération des graphiques SHAP
for name, sample in targets:
    # Calcul des valeurs SHAP pour la 1ère instance et la classe fraude (index 1)
    shap_values = explainer(sample)[0, :, 1]
    
    # Affichage des détails de la transaction
    print(f'--- {name} :', sample.iloc[0].to_dict())
    print('Valeur de base :', round(float(shap_values.base_values), 4))
    
    # Affichage des contributions des caractéristiques triées par impact décroissant
    for col, val in sorted(zip(sample.columns, shap_values.values), key=lambda t: -abs(t[1])):
        print(f'  {col:11s} {val:+.4f}')
        
    # Vérification de cohérence (Base + Somme vs Prédiction du modèle)
    base_sum = round(float(shap_values.base_values + shap_values.values.sum()), 4)
    prediction = round(m.predict_proba(sample)[0, 1], 4)
    print(f'Base + Somme   : {base_sum} | Prédiction : {prediction}')
    
    # Génération et sauvegarde du graphique waterfall
    plt.figure()
    shap.plots.waterfall(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(f'reports/shap_{name}.png', dpi=120)
    plt.close('all')