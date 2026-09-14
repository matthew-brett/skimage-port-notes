#!/bin/bash
if [ -z "$1" ]; then
    echo Specify notebook file.
    exit 1
fi
nb_file=$1
jupytext "$nb_file" --to ipynb --execute
nb_base="${nb_file%.*}"
jupyter nbconvert "${nb_base}.ipynb" --to pdf
