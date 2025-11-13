# simulation-ppo-repo

Ce dépôt contient `simulationPPO.py` comme fichier central : un environnement Gym pour simuler un véhicule électrique et un script d'entraînement/évaluation avec Stable-Baselines3 (PPO).

Structure proposée

- `simulationPPO.py` : fichier principal (environnement, classes véhicule/piste, fonctions d'évaluation et sauvegarde de traces).
- `requirements.txt` : dépendances Python minimales.
- `scripts/` : scripts d'aide (exécution / visualisation) — optionnel.

Prérequis

- Python 3.8+ (recommandé)
- Un environnement virtuel (venv ou conda)
- ffmpeg si vous voulez sauvegarder des animations

Installation rapide

```bash
# depuis la racine du dépôt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exemples d'utilisation

- Lancer un court run d'évaluation (si vous avez adapté un modèle ou voulez utiliser la partie `run_episode_and_record`) :

```bash
python3 simulationPPO.py
```

- Visualiser / tracer les résultats : j'ai inclus des scripts utiles dans `scripts/` dans le dépôt principal (si présents). Voir `scripts/plot_short_eval.py`.

Remarques

- Ce dépôt est un conteneur minimal ; si tu veux que je crée un dépôt Git (git init, add, commit) je peux le faire (dis-le moi).
- Les versions de dépendances dans `requirements.txt` sont larges par défaut ; je peux pinner des versions précises si tu veux.

