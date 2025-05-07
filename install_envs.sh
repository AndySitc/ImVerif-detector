#!/bin/bash

# Déclaration des environnements avec leur version de Python et leurs fichiers requirements
declare -A envs
envs["Andy"]="python=3.10 requirements_andy.txt"
envs["Abdel"]="python=3.10 requirements_abdel.txt"
envs["Alex"]="python=3.10 requirements_alex.txt"
envs["Sameer"]="python=3.10 requirements_sameer.txt"
envs["Amine"]="python=3.10 requirements_amine.txt"
envs["CoDE"]="python=3.10 requirements_code.txt"
envs["Sahar"]="python=3.10 requirements_sahar.txt"

# Création et installation des environnements
for env in "${!envs[@]}"; do
    python_version=$(echo "${envs[$env]}" | cut -d' ' -f1)
    requirements_file=$(echo "${envs[$env]}" | cut -d' ' -f2)

    echo "Création de l'environnement $env avec $python_version..."
    conda create -y -n $env $python_version  # Création de l'environnement avec la version de Python

    echo "Activation de l'environnement $env..."
    source activate $env  # ou `conda activate $env` selon ta configuration

    echo "Installation des dépendances depuis $requirements_file..."
    pip install -r $requirements_file  # Installation des packages

    echo "Désactivation de l'environnement $env..."
    conda deactivate
done

echo "Tous les environnements ont été installés avec succès."
