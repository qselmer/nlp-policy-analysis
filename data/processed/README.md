# Processed and validated data

This folder is reserved for analysis-ready tables after human validation.

Expected outputs include:

- `documents_validated.parquet`: one row per official instrument;
- `screening_validated.parquet`: inclusion and exclusion decisions;
- `mechanisms_validated.parquet`: one row per evidence-backed regulatory mechanism;
- `transferability_validated.parquet`: component scores and justification;
- `validation_audit.csv`: corrections, reviewer, date, and reason.

Only records with an accepted or corrected human-validation status should enter the final evidence synthesis.
