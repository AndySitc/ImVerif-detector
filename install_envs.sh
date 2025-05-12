#!/bin/bash

# Declare environments with their associated requirement files
declare -A envs
# Uncomment and configure these as needed
envs["/medias/db/ImagingSecurity_misc/sitcharn/Andy"]="requirements/requirements_andy.txt python=3.10"
envs["/medias/db/ImagingSecurity_misc/sitcharn/Abdel"]="requirements/requirements_abdel.txt python=3.10"
envs["/medias/db/ImagingSecurity_misc/sitcharn/Alex"]="requirements/requirements_alex.txt python=3.10"
envs["/medias/db/ImagingSecurity_misc/sitcharn/Sameer"]="requirements/requirements_sameer.txt python=3.10"
envs["/medias/db/ImagingSecurity_misc/sitcharn/Amine"]="requirements/requirements_amine.txt python=3.10"
envs["/medias/db/ImagingSecurity_misc/sitcharn/CoDE"]="requirements/requirements_code.txt python=3.10"
envs["/medias/db/ImagingSecurity_misc/sitcharn/Ensemble"]="requirements/requirements_ensemble.txt python=3.10"
envs["/medias/db/ImagingSecurity_misc/sitcharn/Sahar"]="requirements/requirements_sahar.yml"

# Create and install each environment
for env_path in "${!envs[@]}"; do
    req_file="${envs[$env_path]}"

    echo "Processing environment at $env_path with file $req_file..."

    if [[ "$req_file" == *.yml ]]; then
        echo "Installing using conda YAML file..."

        # Remove previous env if it exists
        if [ -d "$env_path" ]; then
            echo "Environment path $env_path already exists. Removing it..."
            rm -rf "$env_path"
        fi

        conda env create -y -f "$req_file" -p "$env_path"
    else
        # python_version=$(echo "$req_file" | awk '{print $2}')
        # requirements_file=$(echo "$req_file" | awk '{print $1}')
        python_version=$(echo "${envs[$env_path]}" | cut -d' ' -f2)
        requirements_file=$(echo "${envs[$env_path]}" | cut -d' ' -f1)

        echo "Creating conda environment with $python_version..."
        conda create -y -p $env_path python=3.10
        export PATH="$env_path/bin:$PATH"
        # conda create -p $env $python_version 

        echo "Activating environment..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env_path"

        echo "Installing dependencies from $requirements_file..."
        pip install -r "$requirements_file"

        echo "Deactivating environment..."
        conda deactivate
    fi

    echo "Finished setting up $env_path"
    echo "---------------------------------------------"
done

echo "All environments have been installed successfully."
