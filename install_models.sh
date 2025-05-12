#!/bin/bash

# FILE_ID="1iLhhgf5YJQVd60i4MqdnBOQA1y9s57ap"
FILE_NAME="models.zip"

echo "Downloading ZIP file from Google Drive..."

# Get the confirmation token and download file (handles large files)
wget "https://drive.usercontent.google.com/download?id=1iLhhgf5YJQVd60i4MqdnBOQA1y9s57ap&export=download&authuser=0&confirm=t" -O ${FILE_NAME}

# Extract ZIP file
echo "Extracting contents..."

# unzip -o "${FILE_NAME}" -d .
unzip ${FILE_NAME} -d .

# Clean up ZIP
echo "Cleaning up..."
rm "${FILE_NAME}"

echo "Installation completed successfully."


