# Exploratory Data Analysis Summary

The supplied data contains **62,184 rows** and **11 raw columns**, spanning source-local dates **2015-08-14** to **2017-03-18**. No rows were dropped or altered.

## Key findings and modelling implications

1. **Target distribution:** skew=0.565; zero readings=8,775 (14.1%); above upper 1.5-IQR fence=355. Keep zero and high counts; use MAE and investigate large errors.
2. **Daily rhythm:** weekday peak=18:00 (52.03); weekend peak=17:00 (39.27). Encode hour cyclically and include weekend.
3. **Day-hour interaction:** busiest heatmap cell=Tuesday, 18:00, mean=54.72. Represent joint calendar effects.
4. **Semester association:** during-semester mean=34.53; outside-semester mean=18.47. Retain calendar flags and validate on later dates.
5. **Sparse holiday indicator:** 160 readings across 2 dates. Retain as context but interpret cautiously.
6. **Weather and time variation:** temperature Pearson r=0.373; lowest 28D rolling daily mean=5.72 on 2017-01-15. Combine weather with calendar context and use chronological validation.

## Interpretation limits

These are descriptive associations, not causal claims. The data cover less than two full years, holiday support is sparse, and the structural-break chart is not a formal change-point test. Correlation does not capture cyclical hour effects. Figures are saved under `reports/figures/` at 200 dpi.
