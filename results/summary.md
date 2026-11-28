| Experiment | Variant | Seeds | test acc | test sensitivity | test specificity | test auroc | test ece | val acc | id_val acc |
|---|---|---|---|---|---|---|---|---|---|
| phase1_erm_densenet121 | best | 0,1,2,3,4 | 71.5 ± 9.1 | 50.7 ± 20.4 | 92.3 ± 2.9 | 0.823 ± 0.045 | 0.184 ± 0.105 | 85.7 ± 2.1 | 87.7 ± 14.6 |
| phase1_erm_densenet121 | last | 0,1,2,3,4 | 74.4 ± 8.7 | 59.3 ± 17.6 | 89.5 ± 4.2 | 0.814 ± 0.113 | 0.150 ± 0.080 | 82.7 ± 2.5 | 97.6 ± 0.3 |
| phase1_erm_densenet121 | best+adabn | 0,1,2,3,4 | 89.8 ± 0.7 | 87.1 ± 2.8 | 92.6 ± 2.4 | 0.953 ± 0.011 | 0.039 ± 0.009 | 89.2 ± 1.1 | 97.3 ± 1.0 |
| phase1_erm_densenet121 | last+adabn | 0,1,2,3,4 | 90.2 ± 2.9 | 89.9 ± 0.6 | 90.6 ± 6.3 | 0.963 ± 0.016 | 0.041 ± 0.025 | 88.9 ± 1.5 | 97.9 ± 0.1 |
| phase1_erm_densenet121_pretrained | best | 0,1,2,3,4 | 85.3 ± 6.0 | 75.0 ± 15.3 | 95.7 ± 4.0 | 0.956 ± 0.019 | 0.078 ± 0.060 | 90.3 ± 0.9 | 98.5 ± 0.5 |
| phase1_erm_densenet121_pretrained | last | 0,1,2,3,4 | 78.6 ± 8.6 | 70.3 ± 24.2 | 87.0 ± 12.8 | 0.881 ± 0.093 | 0.112 ± 0.098 | 86.4 ± 2.7 | 96.9 ± 1.3 |
| phase1_erm_densenet121_pretrained | best+adabn | 0,1,2,3,4 | 92.3 ± 1.1 | 91.0 ± 2.5 | 93.5 ± 3.8 | 0.972 ± 0.006 | 0.033 ± 0.009 | 92.4 ± 1.2 | 98.5 ± 0.5 |
| phase1_erm_densenet121_pretrained | last+adabn | 0,1,2,3,4 | 91.9 ± 1.1 | 90.3 ± 1.5 | 93.6 ± 1.5 | 0.972 ± 0.004 | 0.024 ± 0.007 | 89.5 ± 1.4 | 97.9 ± 0.2 |
| phase2_stain_densenet121 | best | 0,1,2,3,4 | 69.7 ± 8.5 | 53.4 ± 21.3 | 86.0 ± 17.9 | 0.803 ± 0.137 | 0.193 ± 0.069 | 90.1 ± 0.7 | 85.2 ± 12.0 |
| phase2_stain_densenet121 | last | 0,1,2,3,4 | 80.5 ± 9.6 | 67.6 ± 22.4 | 93.5 ± 3.7 | 0.892 ± 0.077 | 0.109 ± 0.088 | 87.0 ± 2.8 | 96.9 ± 0.8 |
| phase2_stain_densenet121 | best+adabn | 0,1,2,3,4 | 92.8 ± 1.2 | 89.4 ± 1.9 | 96.2 ± 0.8 | 0.969 ± 0.012 | 0.020 ± 0.006 | 90.6 ± 0.8 | 97.2 ± 0.4 |
| phase2_stain_densenet121 | last+adabn | 0,1,2,3,4 | 93.9 ± 1.1 | 92.5 ± 1.5 | 95.2 ± 2.8 | 0.983 ± 0.005 | 0.013 ± 0.007 | 90.2 ± 1.8 | 97.3 ± 0.6 |

Accuracy, sensitivity and specificity in %; AUROC and ECE as fractions. Mean ± sample std across seeds. test = hospital 2, val = hospital 1, id_val = held-out patches from training hospitals 0, 3, 4.
