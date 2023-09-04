#!/bin/bash

# Check if a file containing links exists
if [ ! -f $1 ]; then
  echo "File '$1' not found."
  exit 1
fi

# Create a directory to store downloaded files
mkdir -p data

# Loop through each line in the text file and download the files
while IFS= read -r url; do
  # Use 'wget' to download the file
  wget --content-disposition "$url" -P data

  # Check if the download was successful
  if [ $? -eq 0 ]; then
    echo "Downloaded: $filename"
  else
    echo "Failed to download: $filename"
  fi
done < $1

echo "Download process completed."


