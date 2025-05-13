#!/bin/bash

# -------------------- Argument Parsing --------------------
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <multiple:true|false> <image_or_list_path> <envs_path>"
    exit 1
fi

multiple=$1       # true or false
input=$2          # image path or file containing a list of images
envs_path=$3      # base path where environments are located

# -------------------- Define Environments --------------------
# Just the names of the subfolders (envs)
# declare -a env_names=("Andy" "Abdel" "Alex" "Sameer" "Amine" "CoDE" "Sahar")
declare -a env_names=("Andy" "Abdel" "Alex" "Amine" "CoDE" "Sahar")
# declare -a env_names=("Andy")

# Scripts associated with each environment
declare -A envs_scripts
declare -A envs_args

envs_scripts["Andy"]="photoshopped_detector/get_pred_andy_test_v2.py"
envs_scripts["Abdel"]="mixed/abdel__v2.py"
envs_scripts["Alex"]="fully_generated/high-fidelity-generative-compression-master/high-fidelity-generative-compression-master/test-df-det_test__v2.py"
# envs_scripts["Sameer"]="CLIP/keywords_extract.py"
envs_scripts["Amine"]="freqNet/run.py"
envs_scripts["CoDE"]="CoDE_folder/CoDE_model/CoDE_run.py"
envs_scripts["Sahar"]="face_falcification/evaluation_text_file_input_test.py"

envs_args["Andy"]=""
envs_args["Abdel"]=""
envs_args["Alex"]="--model_type compression_gan --regime low"
envs_args["Sameer"]=""
envs_args["Amine"]=""
envs_args["CoDE"]=""
envs_args["Sahar"]=""

root="face_falcification/"  # Path prefix if needed

# -------------------- Execution Loop --------------------
for name in "${env_names[@]}"; do
    env="${envs_path}/${name}"
    script="${envs_scripts[$name]}"
    args="${envs_args[$name]}"

    echo "Activating environment: $env"
    echo "---------------------------------------------------------------------------------"
    source activate "$env"

    if [[ "$name" == "Sahar" ]]; then
        echo "Special case for Sahar: running with three different weight files"
        export TRANSFORMERS_CACHE=/medias/db/ImagingSecurity_misc/huggingface_cache
        export HF_HOME=/medias/db/ImagingSecurity_misc/huggingface_cache

        weights=("train_on_df40" "trained_on_fr" "train_on_fs")
        for weight in "${weights[@]}"; do
            cmd="python ${root}evaluation_text_file_input_test.py \
                --detector_path face_falcification/config/detector/clip.yaml \
                --weights_path models/face_falcification/pretrained/${weight}/clip_large.pth \
                --input \"$input\" \
                --face_detector_path face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

            if [[ "$multiple" == "true" ]]; then
                cmd+=" --multiple"
            fi

            eval "$cmd"
        done

    else
        echo "Running model for $env on input: $input"
        export PATH="$env/bin:$PATH"

        if [[ "$multiple" == "true" ]]; then
            python "$script" $args --input "$input" --multiple
        else
            python "$script" $args --input "$input"
        fi
    fi

    echo "Deactivating environment: $env"
    conda deactivate
    echo "---------------------------------------------------------------------------------"
done

# -------------------- Final Assembly --------------------
echo "Running model for ensemble on input: $input"
source activate "${envs_path}/Ensemble"
export PATH="${envs_path}/Ensemble/bin:$PATH"
python temporary.py
python Assembly/test.py --model_checkpoint models/ensemble/meta_classifier.pth --data_norm

echo "All predictions completed."
