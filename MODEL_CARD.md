# Model Card - fraude (version servie par l'alias production)

## Modèle

- **Nom et version dans le registre MLflow :** `fraude`, version 1, alias `production`.
- **Famille et réglages :** RandomForestClassifier, avec `n_estimators = 500` et `max_depth = 16`.
- **Run d'origine (MLflow) :** `22f24b788d0e472e91dcf570a107784a`.

## Objectif et usage prévu

Détecter les transactions bancaires potentiellement frauduleuses à partir du montant, de l'ancienneté du client, de la fréquence des transactions sur 24 heures et de l'indicateur de transaction nocturne. Aider à identifier les opérations nécessitant une vérification.

## Usages exclus

Le modèle ne doit pas être utilisé comme unique fondement pour bloquer définitivement un compte ou accuser un client de fraude. Une vérification humaine reste nécessaire avant toute décision importante.

## Données

- **Fichier :** `data/transactions.csv`.
- **Nombre de lignes :** 5 000.
- **Part de fraudes :** à calculer à partir de la colonne cible `fraude`.
- **Empreinte (tag donnees_md5) :** à renseigner après l'activité 6.
- **Origine des données :** à confirmer à partir de la documentation du jeu de données.

## Performances (jeu de test, 1 250 transactions)

- **F1 :** 0.8669
- **Précision :** 0.8837
- **Rappel :** 0.8507
- **AUC :** 0.9872

## Méthodes d'explicabilité

- **Globale :** importance par permutation et SHAP TreeExplainer.
- **Locale :** à compléter après l'activité 3 avec les explications SHAP des transactions étudiées.
- **Variables les plus influentes :** le montant arrive en première position selon les résultats de permutation (0.593), suivi de `freq_24h` (0.269), `anciennete` (0.268) et `nuit` (0.196). Les résultats SHAP globaux restent à confirmer.

## Comparaison avec un modèle interprétable

À compléter après l'activité 5, à partir des résultats de la régression logistique et de l'arbre de décision de profondeur limitée. Comparer leurs performances avec celles de la forêt aléatoire et leur facilité d'interprétation.

## Limites et risques connus

Les performances mesurées sur le jeu de test ne garantissent pas les mêmes résultats sur des transactions réelles. Les explications SHAP et LIME décrivent le comportement du modèle, mais ne démontrent pas de relation causale. Les erreurs de détection et les cas rares doivent être surveillés.

## Recommandations d'utilisation et de surveillance

Surveiller régulièrement les performances, notamment le rappel et la précision, ainsi que l'évolution des données et des variables utilisées. Prévoir une vérification humaine des transactions signalées comme frauduleuses et réévaluer le modèle après chaque réentraînement.

## Traçabilité

- **Run des explications (expérience `tp7-explicabilite`) :** à renseigner après l'activité 6.
- **Auteurs :** Meriem Baddi.
- **Date :** 13 novembre 2026.
- **Prochaine revue :** à définir selon la fréquence de surveillance et les réentraînements du modèle.