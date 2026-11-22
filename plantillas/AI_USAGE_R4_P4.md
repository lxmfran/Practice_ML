# AI usage - Practice 4, R4

R4's entry for the Practice 4 section of `plantillas/AI_USAGE.md`. R5 copies it there when joining the declarations.

## R4 - Evaluation and validation

**Member:** Francisco Manuel Moya Sánchez

| Field | Entry |
|---|---|
| Tool or service | Claude Code (Anthropic), model Opus 5.5 |
| Purpose | Used as an assistant: reading the statement and planning the R4 tasks; explaining the concepts (canonical folds passed as `cv=folds`, the grouping invariant, paired per-fold differences, pipeline versus family comparisons); hints while I wrote the imports, the fold list, the group check and the metric functions myself; drafting the remaining cells (row-order checks, storage pieces, the tree comparisons, the two P4 rows of `comparativa.csv`) and the Markdown readings and penalty table, which I reviewed; reviewing the R2 and R3 notebooks; help with Git/GitHub (Practice 4 branches and role templates). |
| Files / sections affected | `notebooks/R4_p4.ipynb` (all sections) and `comparativa.csv` (the two Practice 4 rows). |
| How the output was checked | I ran the notebook top to bottom on the repository data, with R1's data cells copied unchanged, without errors. The assertions pass: the fold and group files have the same row order as `dev`, and no group is shared between training and validation. The method was checked by reproducing the Practice 3 − Practice 2 differences already reported (cyber −0.1074, storage −0.0181). The five scores of the tree are the ones printed in R2's and R3's notebooks. I checked that `comparativa.csv` only gained two lines at the end and that the P2 and P3 lines remained the same, and I checked the R2 and R3 notebooks against the list of automatic penalties. |
