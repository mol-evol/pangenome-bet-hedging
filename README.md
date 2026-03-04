# Prokaryotic Pangenomes as Bet-Hedging Devices

Code for reproducing all figures in the manuscript. Figures follow the v40 main text numbering (5 main figures, 10 supplementary figures).

## Quick start

```bash
pip install -r requirements.txt
cd data && bash download_data.sh     # download Horesh et al. dataset (~790 MB)
cd ..
python run_figures.py --all          # regenerate everything
python run_figures.py --figures 1 3  # specific main figures only
python run_figures.py --quick        # fast mode (fewer replicates)
python run_figures.py --dry-run      # show commands without running
```

Output goes to `output/` by default. Override with `--output-dir`.

## Directory layout

```
code/
  run_figures.py            Batch runner — entry point for all figures
  requirements.txt          Python dependencies (numpy, matplotlib, scipy, etc.)

  shared/                   Shared utilities
    params.py               Model parameters (THEORY_PARAMS, EMPIRICAL_PARAMS)
    plotting.py             Colour palette, figure setup, save/export, panel labels

  figures/                  Theory/simulation figure scripts
    figure1_merged.py       Figure 1 (A–E): variance cost + HGT mechanism
    figure2_hgt.py          Figure 2 (A–D): HGT-mediated bet-hedging equilibrium
    figure3_merged.py       Figure 3 (A–F): U-shaped distribution + complexity threshold
    figure1_variance.py     Component module imported by figure1_merged (panels A–B)
    figure3_ushape.py       Component module imported by figure3_merged (panels A–C)
    figure4_complexity.py   Component module imported by figure3_merged (panels D–F)
    figure7_mechanism.py    Component module imported by figure1_merged (panels C–E)
    supplementary_figure_s1_frequency.py          Figure S1: optimal carrier frequency
    supplementary_figure_s2_parameter_sensitivity.py  Figure S2: parameter sensitivity
    supplementary_figure_s3_hgt_optimization.py   Figure S3: HGT rate optimisation

  empirical/                Empirical analyses (E. coli + cross-species)
    figure5_coupling.py     Figure 4 (A–D): environment coupling + gene classification
    figure6_insurance.py    Figure 5 (A–F): niche insurance + model discrimination + fitness
    supplementary_figures_s6_s10.py   Figures S6–S10: supplementary empirical panels
    decoupling_analysis.py  Jaccard distance / PERMANOVA analysis (panels for Fig 4 & S6)
    gene_classification.py  Chi-squared classification of genes (panels for Fig 4 & S7)
    niche_insurance_analysis.py  Away-niche retention analysis (panels for Fig 5 & S8)
    niche_simulation.py     Migration vs bet-hedging simulation (panels for Fig 5 & S9)
    fitness_landscape.py    Strategy fitness comparison (panels for Fig 5 & S10)
    selection_analysis.py   Figure S4: cross-species purifying selection (Douglas & Shapiro)
    variance_analysis.py    Figure S5: cross-species variance constraint (Douglas & Shapiro)
    test_*.py               Unit tests for empirical analyses

  data/                     Input datasets
    download_data.sh        Script to download Horesh et al. data from Figshare
    horesh/                 Horesh et al. (2021) E. coli pangenome (git-ignored; run download_data.sh)
    douglas_shapiro/        Douglas & Shapiro (2024) cross-species metrics (included, 92 KB)
      pangenome_and_related_metrics.tsv.gz

  output/                   Generated figures (git-ignored; regenerate with run_figures.py)
```

## Figure-to-script mapping

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 1 (A–E) | `figure1_merged.py` | Variance cost (A–B) + HGT mechanism (C–E) |
| Fig 2 (A–D) | `figure2_hgt.py` | Gene frequency equilibrium under HGT |
| Fig 3 (A–F) | `figure3_merged.py` | U-shaped distribution (A–C) + complexity threshold (D–F) |
| Fig 4 (A–D) | `figure5_coupling.py` | E. coli environment coupling + gene classification |
| Fig 5 (A–F) | `figure6_insurance.py` | Niche insurance + simulation + fitness landscape |
| Fig S1 | `supplementary_figure_s1_frequency.py` | Optimal population diversity |
| Fig S2 | `supplementary_figure_s2_parameter_sensitivity.py` | Parameter sensitivity |
| Fig S3 | `supplementary_figure_s3_hgt_optimization.py` | HGT rate optimisation |
| Fig S4 | `selection_analysis.py` | Cross-species purifying selection signatures |
| Fig S5 | `variance_analysis.py` | Cross-species variance constraint |
| Figs S6–S10 | `supplementary_figures_s6_s10.py` | Empirical supplementary panels |

## Architecture note

The merged figure scripts (`figure1_merged.py`, `figure3_merged.py`) import simulation functions from their component modules (`figure1_variance.py` + `figure7_mechanism.py`, and `figure3_ushape.py` + `figure4_complexity.py` respectively). The component scripts remain as importable modules and can also be run standalone for debugging.

Similarly, the main-text empirical figures (`figure5_coupling.py`, `figure6_insurance.py`) import panel-drawing functions from the individual analysis scripts. The supplementary empirical figures (`supplementary_figures_s6_s10.py`) draw the complementary panels from the same analysis scripts.

## Data sources

- **Horesh et al. (2021)**: 10,146 *E. coli* genomes with gene presence/absence and isolation source metadata. Figshare: [doi:10.6084/m9.figshare.13270073](https://doi.org/10.6084/m9.figshare.13270073). Reference: [doi:10.1099/mgen.0.000499](https://doi.org/10.1099/mgen.0.000499).
- **Douglas & Shapiro (2024)**: Pangenome metrics for 670 prokaryotic species. Zenodo dataset accompanying Douglas & Shapiro (2024).

## Tests

```bash
cd code
python -m pytest empirical/test_*.py -v
```
