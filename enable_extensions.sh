#!/bin/bash

# Specify the path to the file containing the Jupyter Notebook extensions
extensions_file="jupyter_extensions.txt"

# Read the file line by line and enable each extension
while IFS= read -r line; do
    jupyter nbextension enable --user "$line"
done < "$extensions_file"

