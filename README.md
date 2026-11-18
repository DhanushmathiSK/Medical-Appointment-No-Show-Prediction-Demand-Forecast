# Medical-Appointment-No-Show-Prediction-Demand-Forecast

No-show model: ROC-AUC 0.786 (target met), F1 about 0.64 (target of 0.70 not met).
Forecast model: R² 0.29 and open-day MAPE 51% (targets not met). Say why: bursts of near-zero days, only 12 months of training data, and a series cut at 2021-04-02 because later days looked incomplete.
Weather features: I never saw your permutation importance result, so I can’t say whether they help or just identify dates. Mention it as a limitation, or run that cell and tell me what it shows.
No specialty filter in the forecaster. List it under “future work”.
