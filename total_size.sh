#!/bin/bash

base_path="/medias/db/ImagingSecurity_misc/sitcharn"
declare -a env_names=("Andy" "Abdel" "Alex" "Sameer" "Amine" "CoDE" "Sahar" "Ensemble")

echo "Espace utilisé par chaque environnement :"
total=0
for env in "${env_names[@]}"; do
    env_path="$base_path/$env"
    if [ -d "$env_path" ]; then
        size=$(du -sh "$env_path" | cut -f1)
        echo "$env : $size"
    else
        echo "$env : dossier introuvable"
    fi
done

echo ""
echo "Espace total utilisé :"
du -sh ${env_names[@]/#/$base_path/} | tail -n 1
