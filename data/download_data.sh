#!/usr/bin/env bash
# Download datasets required by the empirical analyses.
#
# Horesh et al. (2021) — E. coli pangenome
#   Figshare: https://microbiology.figshare.com/articles/dataset/13270073
#   Reference: doi:10.1099/mgen.0.000499
#
# Douglas & Shapiro (2024) — cross-species pangenome metrics
#   Already included in the repo (92 KB compressed).
#
# Usage:
#   cd data/
#   bash download_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HORESH_DIR="$SCRIPT_DIR/horesh"

mkdir -p "$HORESH_DIR"

echo "Downloading Horesh et al. (2021) data from Figshare..."
echo ""

# F1: genome metadata (isolation source, phylogroup, genome ID)
if [ -f "$HORESH_DIR/F1_genome_metadata.csv" ]; then
    echo "  F1_genome_metadata.csv already exists, skipping."
else
    echo "  Downloading F1_genome_metadata.csv..."
    curl -L -o "$HORESH_DIR/F1_genome_metadata.csv" \
        "https://figshare.com/ndownloader/files/25426846"
    echo "  Done ($(du -h "$HORESH_DIR/F1_genome_metadata.csv" | cut -f1))."
fi

# F4: complete gene presence/absence matrix (~790 MB)
if [ -f "$HORESH_DIR/F4_complete_presence_absence.csv" ]; then
    echo "  F4_complete_presence_absence.csv already exists, skipping."
else
    echo "  Downloading F4_complete_presence_absence.csv (~790 MB, may take a few minutes)..."
    curl -L -o "$HORESH_DIR/F4_complete_presence_absence.csv" \
        "https://figshare.com/ndownloader/files/25426852"
    echo "  Done ($(du -h "$HORESH_DIR/F4_complete_presence_absence.csv" | cut -f1))."
fi

echo ""
echo "All data downloaded to $HORESH_DIR/"
echo "Douglas & Shapiro data is already at $SCRIPT_DIR/douglas_shapiro/"
