#!/bin/bash

# Check argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <env_base_path>"
    exit 1
fi

env_base=$1  # Example: /medias/db/ImagingSecurity_misc/sitcharn

# Declare environment names and requirement files
declare -A envs
envs["Andy"]="requirements/requirements_andy.txt python=3.10"
envs["Abdel"]="requirements/requirements_abdel.txt python=3.10"
envs["Alex"]="requirements/requirements_alex.txt python=3.10"
envs["Sameer"]="requirements/requirements_sameer.txt python=3.10"
envs["Amine"]="requirements/requirements_amine.txt python=3.10"
envs["CoDE"]="requirements/requirements_code.txt python=3.10"
envs["Ensemble"]="requirements/requirements_ensemble.txt python=3.10"
envs["Sahar"]="requirements/requirements_sahar.yml"

# Loop through each environment and install it
for env_name in "${!envs[@]}"; do
    env_path="$env_base/$env_name"
    req_file_and_version="${envs[$env_name]}"

    echo "Processing environment: $env_name at $env_path"

    if [[ "$req_file_and_version" == *.yml ]]; then
        echo "Installing using conda YAML file..."
        if [ -d "$env_path" ]; then
            echo "Environment path $env_path already exists. Removing it..."
            rm -rf "$env_path"
        fi
        conda env create -y -f "$req_file_and_version" -p "$env_path"
    else
        python_version=$(echo "$req_file_and_version" | cut -d' ' -f2)
        requirements_file=$(echo "$req_file_and_version" | cut -d' ' -f1)

        echo "Creating conda environment with $python_version..."
        # conda create -y -p "$env_path" python="$python_version"
        conda create -y -p $env_path python=3.10

        echo "Activating environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env_path"
         # Ajout de l'export PATH ici
        export PATH="$env_path/bin:$PATH"

        echo "Installing dependencies from $requirements_file..."
        pip install -r "$requirements_file" --no-cache-dir

        echo "Deactivating environment..."
        conda deactivate
    fi

    echo "Finished setting up $env_name"
    echo "---------------------------------------------"
done

echo "All environments have been installed successfully."
