# Introduction au Machine Learning avec Scikit-Learn

Vous trouverez un exemple de code pour apprendre simplement et rapidement un modèle avec __Scikit-Learn__.
Le fichier `src/prepare.py` charge les données et les arrange de manière à pouvoir être traitées.
Le fichier `src/main.py` montre comment séparer les données de départ en deux sous ensembles:

* Le `train set` pour apprendre le modèle avec la fonction `fit()`
* Le `test set` pour réaliser des prédictions et comaprer le résultat avec la réalité, via la fonction `score()`

## Exécuter le programme

Installer les dépendances:
```console
pip install -U requirements.txt
```

Exécuter:
```console
python src/main.py
```
