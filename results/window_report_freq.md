# Time-Window + Frequency-Band Report (Standalone)

## Summary
### Best per phase (RF)
| phase | win_s | overlap | model | bacc_mean | macro_f1_mean |
| --- | --- | --- | --- | --- | --- |
| gait_full | 5.000 | 0.250 | LR | 0.757 | 0.710 |
| post_uturn | 5.000 | 0.250 | LR | 0.784 | 0.722 |
| pre_uturn | 4.000 | 0.250 | LR | 0.738 | 0.680 |
| uturn | 6.000 | 0.500 | LR | 0.932 | 0.907 |


### RF: Time vs Time+Frequency (best per phase)
| phase | time_bacc | time_win_s | time_overlap | timefreq_bacc | timefreq_win_s | timefreq_overlap |
| --- | --- | --- | --- | --- | --- | --- |
| pre_uturn | 0.900 | 4.000 | 0.250 | 0.738 | 4.000 | 0.250 |
| post_uturn | 0.890 | 4.000 | 0.250 | 0.784 | 5.000 | 0.250 |
| gait_full | 0.891 | 3.000 | 0.500 | 0.757 | 5.000 | 0.250 |
| uturn | 0.600 | 3.000 | 0.250 | 0.932 | 6.000 | 0.500 |


![Time vs Time+Frequency](figures/phase_time_vs_timefreq_rf.png)

## Full Summary (first 40 rows)
| phase | sensor | win_s | overlap | model | bacc_mean | macro_f1_mean |
| --- | --- | --- | --- | --- | --- | --- |
| pre_uturn | RF | 3.000 | 0.250 | LR | 0.723 | 0.676 |
| pre_uturn | RF | 4.000 | 0.250 | LR | 0.738 | 0.680 |
| post_uturn | RF | 5.000 | 0.250 | LR | 0.784 | 0.722 |
| post_uturn | ALL | 5.000 | 0.250 | LR | 0.806 | 0.778 |
| post_uturn | RF | 6.000 | 0.250 | LR | 0.745 | 0.708 |
| post_uturn | ALL | 6.000 | 0.250 | RF | 0.791 | 0.802 |
| post_uturn | RF | 5.000 | 0.500 | LR | 0.737 | 0.683 |
| post_uturn | ALL | 5.000 | 0.500 | RF | 0.779 | 0.788 |
| post_uturn | RF | 6.000 | 0.500 | LR | 0.748 | 0.689 |
| post_uturn | ALL | 6.000 | 0.500 | SVM | 0.799 | 0.804 |
| gait_full | RF | 5.000 | 0.250 | LR | 0.757 | 0.710 |
| gait_full | ALL | 5.000 | 0.250 | SVM | 0.787 | 0.790 |
| gait_full | RF | 6.000 | 0.250 | LR | 0.704 | 0.667 |
| gait_full | ALL | 6.000 | 0.250 | RF | 0.785 | 0.798 |
| gait_full | RF | 5.000 | 0.500 | LR | 0.751 | 0.697 |
| gait_full | ALL | 5.000 | 0.500 | RF | 0.803 | 0.809 |
| gait_full | RF | 6.000 | 0.500 | LR | 0.719 | 0.671 |
| gait_full | ALL | 6.000 | 0.500 | RF | 0.794 | 0.801 |
| uturn | RF | 5.000 | 0.250 | LR | 0.917 | 0.882 |
| uturn | ALL | 5.000 | 0.250 | LR | 0.819 | 0.784 |
| uturn | RF | 6.000 | 0.250 | LR | 0.884 | 0.843 |
| uturn | ALL | 6.000 | 0.250 | LR | 0.964 | 0.938 |
| uturn | RF | 5.000 | 0.500 | LR | 0.844 | 0.773 |
| uturn | ALL | 5.000 | 0.500 | LR | 0.837 | 0.794 |
| uturn | RF | 6.000 | 0.500 | LR | 0.932 | 0.907 |
| uturn | ALL | 6.000 | 0.500 | LR | 0.932 | 0.907 |
