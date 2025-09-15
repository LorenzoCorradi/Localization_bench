#!/bin/bash

# cartella sorgente con le immagini
SRC_DIR="./"
# cartella di output
OUT_DIR="out"

# crea la cartella di output
mkdir -p "$OUT_DIR"

# contatore
i=0
for img in "$SRC_DIR"/*; do
    if [ -f "$img" ]; then
        # crea la cartella numerata
        mkdir -p "$OUT_DIR/$i"
        # copia (o sposta) l'immagine dentro
        cp "$img" "$OUT_DIR/$i/"
        # incrementa contatore
        i=$((i+1))
    fi
done

echo "Fatto! Ho messo $i immagini in cartelle numerate in $OUT_DIR"
