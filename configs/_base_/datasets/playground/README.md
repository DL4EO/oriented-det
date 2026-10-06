# Airbus Playground CSVs

Default output of `odet playground-csv` (`tools/generate_airbus_playground_csv.py`).

Each run writes a dated pair here, `annotations_YYYYMMDD.csv` and `split_YYYYMMDD.csv` (UTC), so the files stay in the project instead of inside the Playground export. Commit the pair you train with: the split CSV is the validation holdout, and keeping it in git lets you rerun the same folds later.

Paste the paths the tool prints into the training config:

```json
"annotations_file": "configs/_base_/datasets/playground/annotations_YYYYMMDD.csv",
"split_file": "configs/_base_/datasets/playground/split_YYYYMMDD.csv"
```

A path that includes a directory is resolved from the project root (`ORIENTED_DET_PROJECT_ROOT`, else the current working directory). A bare filename is still resolved under `dataset.data_root`. Pass `--output-dir` to write the CSVs somewhere else.
