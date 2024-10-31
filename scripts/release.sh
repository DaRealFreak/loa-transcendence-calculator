#!/bin/bash

# Define the name of the archive
archive_name="transcendence_calculator_$(date +'%Y%m%d').zip"

base_dir=$(dirname "$0")
cd "$base_dir/../" || exit

# Create a temporary directory for structuring the archive content
temp_dir=$(mktemp -d)

# Copy files and folders to the temp directory with specified exclusions
rsync -av --exclude='.git' --exclude='.venv' --exclude='.idea' --exclude='__pycache__' --exclude='screenshots' \
      --exclude='*.tar.gz' --exclude='dataset/*' --exclude='bin/*' ./ "$temp_dir/"

# Recreate the folder structure for dataset and bin without files
find dataset -type d -exec mkdir -p "$temp_dir/{}" \;
find bin -type d -exec mkdir -p "$temp_dir/{}" \;

# Move into the temporary directory and create the archive from its contents
cd "$temp_dir" || exit
#tar -cvzf "$archive_name" *
zip -r "${archive_name}" *

# Move the archive back to the original directory
mv "$archive_name" "$OLDPWD"

# Clean up the temporary directory
cd "$OLDPWD" || exit
rm -rf "$temp_dir"

# Provide feedback
echo "Archive $archive_name created successfully in the main directory."
