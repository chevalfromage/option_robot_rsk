# Lancement et test du simulateur sur windows:

## Activer l'environnement virtuel python: 

```bash
./venv/Scripts/Activate.ps1
```

## Activer le simulatuer ou le game controler:

```bash
cd robot-soccer-kit
```

### 1: Installer les dépendances via le .toml

```bash
pip install -e .[gc]
```

### 2: Lancer le simulateur ou le game controler
 
 ```bash
python -m rsk.game_controller --simulated
```

ou

```bash
python -m rsk.game_controller
```