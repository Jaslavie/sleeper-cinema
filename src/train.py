"""
TODO
As per the paper: "The goal of semi-supervised GAD is to learn an anomaly scoring function."
Validation: model can label anomalies by finding scores that are greater than a threshold without ground truth labels for anomalies.
Training: subset of labeled normal and outlier movies are fed into the model and trained to find the optimal scoring function.
"""