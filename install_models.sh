#!/bin/bash

FILE_ID="1iLhhgf5YJQVd60i4MqdnBOQA1y9s57ap"
FILE_NAME="models.zip"

echo "Downloading ZIP file from Google Drive..."

# Get the confirmation token and download file (handles large files)
CONFIRM=$(curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" | \
         grep -o 'confirm=[^&]*' | sed 's/confirm=//')

curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "${FILE_NAME}"
rm ./cookie

# Extract ZIP file
echo "Extracting contents..."
unzip -o "${FILE_NAME}" -d .

# Clean up ZIP
echo "Cleaning up..."
rm "${FILE_NAME}"

echo "Installation completed successfully."
