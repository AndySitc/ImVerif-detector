#!/bin/bash

# Vérifie la mémoire libre sur chaque GPU
get_free_gpu() {
    for i in 0 1; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i)
        total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $i)
        ratio=$(echo "$free / $total" | bc -l)
        if (( $(echo "$ratio > 0.05" | bc -l) )); then
            echo $i
            return
        fi
    done
    echo "none"
}

# Trouve un GPU dispo
GPU=$(get_free_gpu)
if [ "$GPU" == "none" ]; then
    echo "❌ Aucun GPU disponible avec suffisamment de mémoire."
    exit 1
fi

# Définit CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU
echo "🟢 Utilisation du GPU $GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

# Vérifier si un argument (chemin de l'image) est fourni
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_image>"
    exit 1
fi

# IMAGE_PATH="$1"
multiple=$1  # true ou false
input=$2     # soit une image, soit un fichier contenant la liste d'images

# Déclaration des environnements et scripts Python associés
declare -A envs_scripts
declare -A envs_args

# Scripts
envs_scripts["Andy"]="photoshopped_detector/get_pred_andy_test_v2.py"
envs_scripts["abdel"]="mixed/abdel__v2.py"
envs_scripts["Alex"]="HiFic/high-fidelity-generative-compression-master/high-fidelity-generative-compression-master/test-df-det_test__v2.py"
envs_scripts["Sameer"]="CLIP/keywords_extract.py"
envs_scripts["Amine"]="freqNet/run.py"
envs_scripts["CoDE"]="CoDE/CoDE_model/CoDE_run.py"
envs_scripts["Sahar"]="face_falcification/evaluation_text_file_input.py"

# Arguments spécifiques pour chaque environnement
envs_args["Andy"]=""
envs_args["abdel"]=""
envs_args["Alex"]="--model_type compression_gan --regime low"
envs_args["Sameer"]=""
envs_args["Amine"]=""
envs_args["CoDE"]=""
envs_args["Sahar"]=""

# INPUT IMAGE
root="face_falcification/"  # si besoin

for env in "${!envs_scripts[@]}"; do
    echo "Activation de l'environnement $env..."
    echo "---------------------------------------------------------------------------------"
    source activate "$env"

    if [[ "$env" == "Sahar" ]]; then
        echo "Cas spécial Sahar : 3 exécutions avec des poids différents"

        python "face_falcification/evaluation_text_file_input.py" \
            --detector_path "face_falcification/config/detector/clip.yaml" \
            --weights_path "models/face_falcification/pretrained/train_on_df40/clip_large.pth" \
            --input "$input" \
            --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

        python "${root}evaluation_text_file_input.py" \
            --detector_path "face_falcification/config/detector/clip.yaml" \
            --weights_path "models/face_falcification/pretrained/trained_on_fr/clip_large.pth" \
            --input "$input" \
            --face_detector_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

        python "${root}evaluation_text_file_input.py" \
            --detector_path "face_falcification/config/detector/clip.yaml" \
            --weights_path "models/face_falcification/pretrained/train_on_fs/clip_large.pth" \
            --input "$input" \
            --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

    else
        echo "Exécution du modèle dans $env... sur l'image $input..."
        args="${envs_args[$env]}"
        if [[ "$multiple" == "true" ]]; then
            python "${envs_scripts[$env]}" $args --input "$input" --multiple
        else
            python "${envs_scripts[$env]}" "$input"
        fi
    fi

    echo "Désactivation de l'environnement $env..."
    conda deactivate
    echo "---------------------------------------------------------------------------------"
done



echo "Toutes les prédictions ont été effectuées."
