*Lancement et test du simulateur sur windows:

**Activer l'environnement virtuel python: 

./rsk_neural_simulator/venv/Scripts/Activate.ps1

**Activer le simulatuer ou le game controler:

cd robot-soccer-kit

***1: Installer les dépendances via le .toml

pip install -e .[gc]

***2: Lancer le simulateur ou le game controler
 
python -m rsk.game_controller --simulated

ou

python -m rsk.game_controller
