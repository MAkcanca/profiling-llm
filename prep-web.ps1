# Powershell script that will copy generated_profiles/ and gold-standards/ into the web directory while removing the previously existing files

# Remove existing files
Remove-Item -Recurse -Force web-dashboard/generated_profiles/*
Remove-Item -Recurse -Force web-dashboard/gold-standards/*

# Copy new files
Copy-Item -Recurse -Force generated_profiles web-dashboard/
Copy-Item -Recurse -Force gold-standards web-dashboard/
