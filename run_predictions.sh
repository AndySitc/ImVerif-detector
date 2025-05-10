#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <multiple:true|false> <image_or_list_path>"
    exit 1
fi

multiple=$1  # true or false
input=$2     # image path or file containing a list of images

# Declare environments and associated Python scripts
declare -A envs_scripts
declare -A envs_args

# Scripts
envs_scripts["/medias/db/ImagingSecurity_misc/sitcharn/Andy"]="photoshopped_detector/get_pred_andy_test_v2.py"
envs_scripts["/medias/db/ImagingSecurity_misc/sitcharn/Abdel"]="mixed/abdel__v2.py"
envs_scripts["/medias/db/ImagingSecurity_misc/sitcharn/Alex"]="fully_generated/high-fidelity-generative-compression-master/high-fidelity-generative-compression-master/test-df-det_test__v2.py"
envs_scripts["/medias/db/ImagingSecurity_misc/sitcharn/Sameer"]="CLIP/keywords_extract.py"
envs_scripts["/medias/db/ImagingSecurity_misc/sitcharn/Amine"]="freqNet/run.py"
envs_scripts["/medias/db/ImagingSecurity_misc/sitcharn/CoDE"]="CoDE/CoDE_model/CoDE_run.py"
envs_scripts["/medias/db/ImagingSecurity_misc/sitcharn/Sahar"]="face_falcification/evaluation_text_file_input_test.py"

# Script arguments
envs_args["/medias/db/ImagingSecurity_misc/sitcharn/Andy"]=""
envs_args["/medias/db/ImagingSecurity_misc/sitcharn/Abdel"]=""
envs_args["/medias/db/ImagingSecurity_misc/sitcharn/Alex"]="--model_type compression_gan --regime low"
envs_args["/medias/db/ImagingSecurity_misc/sitcharn/Sameer"]=""
envs_args["/medias/db/ImagingSecurity_misc/sitcharn/Amine"]=""
envs_args["/medias/db/ImagingSecurity_misc/sitcharn/CoDE"]=""
envs_args["/medias/db/ImagingSecurity_misc/sitcharn/Sahar"]=""

root="face_falcification/"  # path prefix if needed

for env in "${!envs_scripts[@]}"; do
    echo "Activating environment: $env"
    echo "---------------------------------------------------------------------------------"
    source activate "$env"

    if [[ "$env" == "/medias/db/ImagingSecurity_misc/sitcharn/Sahar" ]]; then
        echo "Special case for Sahar: running with three different weight files"
        export TRANSFORMERS_CACHE=/medias/db/ImagingSecurity_misc/huggingface_cache
        export HF_HOME=/medias/db/ImagingSecurity_misc/huggingface_cache

        if [[ "$multiple" == "true" ]]; then
            python "${root}evaluation_text_file_input_test.py" \
                --detector_path "face_falcification/config/detector/clip.yaml" \
                --weights_path "models/face_falcification/pretrained/train_on_df40/clip_large.pth" \
                --input "$input" \
                --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat" \
                --multiple

            python "${root}evaluation_text_file_input_test.py" \
                --detector_path "face_falcification/config/detector/clip.yaml" \
                --weights_path "models/face_falcification/pretrained/trained_on_fr/clip_large.pth" \
                --input "$input" \
                --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat" \
                --multiple

            python "${root}evaluation_text_file_input_test.py" \
                --detector_path "face_falcification/config/detector/clip.yaml" \
                --weights_path "models/face_falcification/pretrained/train_on_fs/clip_large.pth" \
                --input "$input" \
                --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat" \
                --multiple
        else
            python "${root}evaluation_text_file_input_test.py" \
                --detector_path "face_falcification/config/detector/clip.yaml" \
                --weights_path "models/face_falcification/pretrained/train_on_df40/clip_large.pth" \
                --input "$input" \
                --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

            python "${root}evaluation_text_file_input_test.py" \
                --detector_path "face_falcification/config/detector/clip.yaml" \
                --weights_path "models/face_falcification/pretrained/trained_on_fr/clip_large.pth" \
                --input "$input" \
                --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

            python "${root}evaluation_text_file_input_test.py" \
                --detector_path "face_falcification/config/detector/clip.yaml" \
                --weights_path "models/face_falcification/pretrained/train_on_fs/clip_large.pth" \
                --input "$input" \
                --face_detector_path "face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"
        fi

    else
        echo "Running model for $env on input: $input"
        args="${envs_args[$env]}"
        export PATH="$env_path/bin:$PATH"
        if [[ "$multiple" == "true" ]]; then
            python "${envs_scripts[$env]}" $args --input "$input" --multiple
        else
            python "${envs_scripts[$env]}" $args --input "$input"
        fi

    fi

    echo "Deactivating environment: $env"
    conda deactivate
    echo "---------------------------------------------------------------------------------"
done

# python Assembly/run_assemble.py
source activate /medias/db/ImagingSecurity_misc/sitcharn/Ensemble
python temporary.py
python Assembly/test.py --model_checkpoint models/ensemble/meta_classifier.pth --data_norm

echo "All predictions completed."
