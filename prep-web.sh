# Bash script that will copy generated_profiles/ and gold-standards/ into the web directory while removing the previously existing files

# Remove existing files
rm -rf web-dashboard/generated_profiles/*
rm -rf web-dashboard/gold-standards/*

# Copy new files
cp -r generated_profiles web-dashboard/
cp -r gold-standards web-dashboard/
