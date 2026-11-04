# Practice II - Machine Learning (P2-P5)

Team repository for the cumulative Machine Learning project, Practices 2 to 5.
The internal role guide is available in [`docs/Guia_Roles_P2.pdf`](docs/Guia_Roles_P2.pdf).
The course assignment remains the authoritative specification.

## Repository structure

This repository corresponds to the `Proyecto/` directory in the assignment.
Commands described as running from `Proyecto/` must run from the repository root.

| Location | Contents | Responsibility |
|---|---|---|
| `datos/cyber/`, `datos/storage/` | Course data, provenance, grouping metadata and the canonical fold CSV files, which are already available | Preserve course-provided files unchanged |
| `esquemas/` | Submission schemas | Read-only reference |
| `herramientas/` | Course submission-format validator | Read-only reference |
| `final_test/` | Course README; final-test features are released only after the P5 model lock | No P2 notebook reads or checks final-test files |
| `plantillas/` | Existing course document files, completed in place for this team | Feature register, AI disclosure, roles and protocol memo |
| `folds/` | Course canonical fold files, read by the Practice 3 notebooks | Read-only reference |
| `notebooks/` | Notebooks of the current practice | Each role, its own notebook; R5 the final ones |
| `entrega_p2/` | Files handed in for Practice 2 | Archive, do not change |
| `entregas_prueba/` | Development-only dry-run predictions and corresponding feature subsets | R2 and R3; integration checked by R5 |
| Repository root | Cumulative `comparativa.csv`; the document files remain in `plantillas/` | R4 owns the comparison table; R5 checks delivery completeness |

## Practice 3 notebooks

Each role works in its own notebook inside `notebooks/`, so data paths start with `../datos` and fold paths with `../folds`.
R5 joins them into `notebooks/p3_cyber.ipynb` and `notebooks/p3_storage.ipynb`, the files that are handed in.
Check a notebook with `python revisor_notebook.py notebooks/my_notebook.ipynb`. It only reports and never changes the notebook.

The Practice 2 role drafts are no longer in `main`; they remain in the Git history (commit `69bd821`).

Coordinate substantive model or interpretation changes with the relevant role. Integration
may adjust paths and separate mixed-challenge cells while preserving the model configurations.

## Working with Git

Clone the repository and install the project dependencies:

```bash
git clone https://github.com/lxmfran/Practice_II_ML.git
cd Practice_II_ML
pip install -r requirements.txt
```

Work on the branch assigned to your role. Before integrating others' work, fetch the
remote state and merge `origin/main` into your branch. Preserve uncommitted changes
before switching branches or resolving conflicts. Another team member reviews the
pull request before it is merged into `main`.

| Branch | P2 role |
|---|---|
| `r1-datos` | R1 - Data and preprocessing |
| `r2-cyber` | R2 - Classification |
| `r3-storage` | R3 - Regression |
| `r4-evaluacion` | R4 - Evaluation and validation |
| `r5-informe` | R5 - Reporting and reproducibility |

Draft outputs may be cleared to reduce merge conflicts. This does not apply to the
two final delivery notebooks: restart their kernels, execute all cells, and save outputs.

## P2 checks before delivery

- Print `PROJECT_RANDOM_STATE = 2627` in the first cell of each final notebook. Pass it only to operations that consume randomness; do not search over seeds.
- Never use `row_id`, `feature_group`, `config_group` or `fold_id` as estimator inputs. Never reintroduce excluded predictors such as `iops`.
- Fit learned preprocessing only on each fold's training portion, inside a pipeline.
- After the explicit P2.2 protocol demonstration, use the course canonical fold assignments for scored development results.
- Model storage latency as `z = log10(latency_ns)` after P2.3. Invert predictions with `10 ** z_hat` before computing physical errors or writing submission latencies.
- Include the completed cumulative registers, comparison rows, the OneR explanation and the two-page protocol memo.
- Keep all assessed material in English, including comments, Markdown interpretations, figure labels and captions. The Spanish role guide is internal team documentation.
- Record the actual Python and library versions used for the final run; the verified numeric and kernel versions are pinned in `requirements.txt`; the interactive frontend requirement is retained separately.
- Document AI assistance in `AI_USAGE.md`, including purpose, affected sections and verification. No generative AI is allowed during individual defence questioning.
- Follow the data attribution and use requirements recorded in both `PROVENANCE.md` files. The team's internal guide requires a private repository for the course data.

## Submission-format dry runs

Run these commands from the repository root, using the same Python environment as the notebooks:

```bash
python herramientas/validate_submission.py cyber entregas_prueba/cyber_dry_run.csv entregas_prueba/cyber_dry_run_features.csv datos/cyber/cyber_class_codes.json
python herramientas/validate_submission.py storage entregas_prueba/storage_dry_run.csv entregas_prueba/storage_dry_run_features.csv
```

The validator checks format and identifiers, not predictive performance. For storage,
verify the conversion to nanoseconds and the output magnitude as well: a file containing
positive log-scale predictions can pass the format checks while still being incorrect.

## Verified P2 execution

On 2026-10-25, both integrated notebooks were run from beginning to end in fresh
Jupyter kernels, with outputs saved and no error outputs. The verification environment
was Python 3.12.14, NumPy 2.5.4, pandas 3.0.1, scikit-learn 1.9.1 and matplotlib 3.11.2.
The verified numeric stack is now pinned in `requirements.txt`. An earlier setup comment
referred to Python 3.14; the verified run instead used Python 3.12.14. Both dry-run validators returned
`valid`. Notebook metadata also records the environment and the source-cell mapping.

Numerical tie handling and floating-point differences can affect a small number of KNN
outputs across environments. Report the results actually obtained in the recorded
environment, and retain the grouped canonical protocol, grids and frozen seed.

## P2 requirements and verification (25 October 2026)

The integration includes the contributions available in `main` at commit `5190583`,
with paths adjusted for execution from `notebooks/`. The protocol discussion reports
the observed score differences and distinguishes paired model comparisons on the
same folds from comparisons between different split protocols. The team study notes
are internal material.

| Requirement | Evidence in the delivery | Result |
|---|---|---|
| P2.1 | Both final notebooks: schemas, supports, 34,883 cyber groups, 2,438 storage configurations and five replicates | Checked |
| P2.2 (a)-(g) | Own grouped folds, invariant outputs, fixed five-neighbour comparisons, canonical discrepancy tables and adoption | Checked |
| P2.3 | Storage: bounded raw/log comparison and justification repeated in the memo | Checked |
| P2.4 | Dummy/KNN, independent grids, full curves and dispersion, metric contracts, semantic confusion and paired model differences | Checked |
| P2.5 | OOF probabilities, ransomware support, ROC/AUC/Brier, five prescribed thresholds, separate argmax and operational costs | Checked |
| P2.6 | Both valid development dry runs, OneR explanation, four comparison rows, frozen basis and two-page English memo | Checked |
| Reproducibility | 2627 printed first, learned preprocessing inside each fold, canonical sidecars, recorded versions and fresh-kernel execution | Checked |
| Cumulative records | Completed P2 files in `plantillas/`: features, roles, AI usage and memo | Checked |

The final verification executed R4 (12 code cells), cyber (34) and storage (45)
without error outputs. Original course data were unchanged; schema documentation
and validator source only differ from the initial package in Windows line endings.
Both dry-run format checks returned `valid`. Comparison-table rows retain their
original measurement snapshots, including fit times; reruns can change timing and
do not rewrite historical rows.

For assessment, use **only `P2_cyber.ipynb` and `P2_storage.ipynb` as the two P2
challenge notebooks**, together with `comparativa.csv`, the completed records in
`plantillas/`, the memo PDF, the dry-run files, and the course data/dependencies needed
to reproduce them. R1-R4 source notebooks, `NOTAS_R4_explicacion.md` and the internal
Spanish role guide are supporting working material, not extra assessed challenge
notebooks. Later milestone template sections and `model_lock.md` are intentionally
not filled in for P2. The `features.md` references in historical comparison rows
identify the register stored at `plantillas/features.md`.

Before submission, each member confirms their AI declaration, reads the final
interpretations and figures, and practises explaining the work without assistance.
The team also confirms the submission format and repository visibility required by
its course instructions. The reviewed final files must be merged into `main` before
submission.
