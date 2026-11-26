`prepare_data.py` writes one shared version of the labelled arrays, chronological split manifest, feature scaler, and BMI scaler here. Both model comparisons load these exact artifacts.

Do not manually edit split files between model runs. Re-run data preparation only when the input features, BMI labels, or split configuration change.
