#!/bin/bash

# In this case, you need to have a text file containing image paths
export CUDA_VISIBLE_DEVICES=1

export TRANSFORMERS_CACHE=/medias/db/ImagingSecurity_misc/huggingface_cache
export HF_HOME=/medias/db/ImagingSecurity_misc/huggingface_cache

# Define the root directory
root='/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/'

# trained on all dataset
python "${root}evaluation_text_file_input.py" \
  --detector_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/config/detector/clip.yaml" \
  --weights_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/models/face_falcification/pretrained/train_on_df40/clip_large.pth" \
  --input "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg" \
  --face_detector_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

# trained on FR dataset
python "${root}evaluation_text_file_input.py" \
  --detector_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/config/detector/clip.yaml" \
  --weights_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/models/face_falcification/pretrained/trained_on_fr/clip_large.pth" \
  --input "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg" \
  --face_detector_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

# trained on FS dataset
python "${root}evaluation_text_file_input.py" \
  --detector_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/config/detector/clip.yaml" \
  --weights_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/models/face_falcification/pretrained/train_on_fs/clip_large.pth"  \
  --input "/medias/db/ImagingSecurity_misc/Collaborations/Hermes deepfake challenge/data/dataset_deepfake/dataset_deepfake_2/fake/generation/0.jpg" \
  --face_detector_path "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif-detector/face_falcification/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

# trained on FR dataset
# python "${root}training/evaluation_text_file_input.py" \
#   --detector_path "${root}training/config/detector/clip.yaml" \
#   --weights_path "${root}training/pretrained/trained_on_fr/clip_large.pth" \
#   --image_dir "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif_Detector 2/Sahar/output/FRAll_train" \
#   --face_detector_path "${root}preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

# # trained on FS dataset
# python "${root}training/evaluation_text_file_input.py" \
#   --detector_path "${root}training/config/detector/clip.yaml" \
#   --weights_path "${root}training/pretrained/train_on_fs/clip_large.pth" \
#   --image_dir "/medias/db/ImagingSecurity_misc/Collaborations/ImVerif_Detector 2/Sahar/output/FSAll_train" \
#   --face_detector_path "${root}preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"

  
