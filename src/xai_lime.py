from lime.lime_tabular import LimeTabularExplainer
from xai_commun import CAS, X, Xtr, m


def expliquer(graine, n):
    """Crée un explicateur LIME et retourne l'explication pour l'instance cible."""
    explainer = LimeTabularExplainer(
        Xtr.values,
        feature_names=list(X.columns),
        class_names=['normal', 'fraude'],
        mode='classification',
        random_state=graine
    )
    return explainer.explain_instance(
        CAS.values[0],
        m.predict_proba,
        num_features=4,
        num_samples=n
    )


# 1. Explication de référence (graine=0, n=2000)
e = expliquer(0, 2000)
print('Règles :', [(r, round(w, 3)) for r, w in e.as_list()])

# Sauvegarde des rapports (HTML interactif et image PNG)
e.save_to_file('reports/lime_benali.html')
fig = e.as_pyplot_figure()
fig.tight_layout()
fig.savefig('reports/lime_benali.png', dpi=120)
fig.clear()

# 2. Analyse de la stabilité (test sur différentes tailles de voisinage et graines)
for n in [500, 2000]:
    print(f'\n--- {n} voisins      montant    nuit   anciennete   freq_24h')
    for graine in range(5):
        d = dict(expliquer(graine, n).as_list())
        # Extraction des poids associés à chaque caractéristique de X
        w = [next(v for k, v in d.items() if col in k) for col in X.columns]
        print(f'graine {graine}        {w[0]:+.3f}  {w[3]:+.3f}     {w[1]:+.3f}    {w[2]:+.3f}')