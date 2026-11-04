# `features.md` — the typed feature register

**Team:** _Team 1_ · **Cumulative:** add to this file at every milestone and
**date each addition**. Never delete an earlier entry.

This is a **typed register, not a free list**. Every entry carries its definition, its
motivation, the milestone at which it was introduced, its **kind** and its **route**.

## The three kinds

| Kind | Meaning |
|---|---|
| **source feature** | a column distributed by the course, used as it is |
| **representation transformation** | scaling, encoding, or a **monotone** change of an existing numeric predictor — preprocessing, adding **no new predictive information** |
| **derived predictive feature** | a genuinely **new** predictor built from two or more source columns, or from one in a non-monotone way |

## The two routes

| Route | Meaning |
|---|---|
| **primary** | uses only the predictive-information basis frozen at the end of Practice 2. Only `primary` rows may be read as a model-family comparison. |
| **extension** | an engineering extension. Welcome, reported in its own comparison rows, and never a family comparison. |

**Nothing may be derived from a column that is not distributed, and nothing may be derived
from `row_id`.**

## What the frozen basis is

| Challenge | Frozen predictive-information basis | Frozen representation |
|---|---|---|
| A — cyber | the **eleven** distributed source predictors | `log1p` on the eight numeric predictors, declared as a representation transformation |
| B — storage | the **nine** distributed source predictors | none beyond method-required scaling; the **target** representation `z = log10(latency_ns)` is frozen in P2.3 |

Method-required predictor representations, including scaling and categorical encoding, may
differ between model families within the declared information basis. The target representation
`z = log10(latency_ns)` is fixed from P2.3 through P5; it is not a family-specific choice.
Unless the predictor representation is held constant, the evidence is reported as a
**pipeline comparison**, not an isolated comparison of estimator families.

## Register

| Date | Milestone | Challenge | Feature | Definition | Motivation | Kind | Route |
|---|---|---|---|---|---|---|---|
| 2026-10-19 | P2 | cyber | `proto` | distributed transport protocol | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `service` | distributed application-layer service | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `conn_state` | distributed connection state | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `duration` | distributed connection duration | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `src_bytes` | distributed payload bytes sent by the originator | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `dst_bytes` | distributed payload bytes sent by the responder | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `missed_bytes` | distributed bytes missed in content gaps | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `src_pkts` | distributed packets sent by the originator | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `src_ip_bytes` | distributed IP-level bytes sent by the originator; count, >= 0 | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `dst_pkts` | distributed packets sent by the responder | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `dst_ip_bytes` | distributed IP-level bytes sent by the responder | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `block_size` | distributed I/O block size; integer grid 4–1024 | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `n_jobs` | distributed number of parallel jobs; integer 1–64 | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `iodepth` | distributed I/O queue depth; integer 1–32 | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `read_fraction` | distributed percentage of read operations in the workload; integer 0–100 | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `n_disks` | distributed number of disks; integer 3–24 | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `load_type` | distributed access pattern | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `io_type` | distributed operation type | source column | source feature | primary |
| 2026-10-19 | P2 | storage | `raid` | distributed redundancy scheme | source column; a label, not a quantity, so it is one-hot encoded, never read as a number | source feature | primary |
| 2026-10-19 | P2 | storage | `device_type` | distributed device type | source column | source feature | primary |
| 2026-10-19 | P2 | cyber | `log1p(...)` on the eight numeric predictors | monotone transform of existing predictors | brings nine orders of magnitude into a range a distance can see | representation transformation | primary |
| 2026-10-19 | P2 | storage | `z = log10(latency_ns)` | target representation | positive target over four-plus orders of magnitude, log-ratio rather than absolute-nanosecond error currency, raw skew reduced; see the P2.3 justification and the grouped raw-versus-log demonstration | representation transformation (target) | primary |
| 2026-10-25 | P2 | cyber | One-hot encoding of `proto`, `service`, `conn_state` | `OneHotEncoder(handle_unknown="ignore")`, fitted on the training portion of each fold | represents nominal categories without imposing an ordinal distance; unseen categories are encoded as zeros for that predictor | representation transformation | primary |
| 2026-10-25 | P2 | cyber | Standardisation of the eight numeric predictors after `log1p` | `StandardScaler` fitted on training-fold log-transformed values, then applied unchanged to validation | prevents units and remaining scale differences from dominating KNN distance | representation transformation | primary |
| 2026-10-25 | P2 | storage | One-hot encoding of `load_type`, `io_type`, `raid`, `device_type` | `OneHotEncoder(handle_unknown="ignore")`, fitted on the training portion of each fold | represents nominal categories without imposing a numerical order; `raid` remains a nominal scheme | representation transformation | primary |
| 2026-10-25 | P2 | storage | Standardisation of `block_size`, `n_jobs`, `iodepth`, `read_fraction`, `n_disks` | `StandardScaler` fitted on the training portion of each fold, then applied unchanged to validation | prevents different numeric ranges from dominating KNN distance | representation transformation | primary |

## P2 scope and implementation checks

The source-feature entries retain the supplied date of 2026-10-19. The explicit encoding
and scaling entries were added during the register review on 2026-10-25; they document
transformations already used in the P2 notebooks, not a new predictive-information basis.

- Cyber uses exactly the eleven source predictors listed above; storage uses exactly the nine.
- `row_id`, `feature_group`, `config_group` and `fold_id` are identifiers or evaluation metadata, never estimator inputs. No feature is derived from them.
- `attack_code` is the cyber target. `latency_ns` is the distributed storage target; neither is a predictor. After the bounded raw-target demonstration in P2.3, storage models use `z = log10(latency_ns)`.
- No derived predictive feature or engineering-extension route is introduced in P2. One-hot output columns are representations of source predictors, not additional source information.
- Encoders and scalers are fitted only on the training portion of each fold through a `Pipeline` and `ColumnTransformer`. The `log1p` and `log10` functions have no learned parameters.
- Physical predictions are obtained with `10 ** z_hat`; MAE in nanoseconds is calculated against the original `latency_ns` after that inversion. `10 ** MAE(z)` is a multiplicative factor, never an error in nanoseconds.

## If you want a derived predictive feature in your final deployment candidate

You must do **exactly one** of these two things, and say which:

1. **rerun the relevant earlier comparison models on the same new feature basis**, so the
   families are again compared on equal information; or
2. **state explicitly**, in the notebook and in the model lock, that the resulting score is
   **not** an isolated model-family comparison.

Neither option is penalised. Silently doing neither is.
