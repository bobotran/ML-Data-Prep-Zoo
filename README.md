# ML Data Prep Zoo

A zoo of labelled datasets and ML models for data prep tasks.

## Task: ML Feature Type Inference

### Leaderboard

By submitting results, you acknowledge that your holdout test results (data_test.csv) are obtained purely by training on the training set (data_train.csv).

|                                                 Approaches                                               |     9-class      Accuracy    |   |      Numeric     |               |   |     Categorical    |               |   |      Datetime    |               |   |      Sentence    |               |   |        URL       |               |   |     Embedded   Number    |               |   |        List      |               |   |     Not-Generalizable    |               |   |     Context-Specific    |               |   |
|:--------------------------------------------------------------------------------------------------------:|:----------------------------:|---|:----------------:|:-------------:|---|:------------------:|:-------------:|---|:----------------:|:-------------:|---|:----------------:|:-------------:|---|:----------------:|:-------------:|---|:------------------------:|:-------------:|---|:----------------:|:-------------:|---|:------------------------:|:-------------:|---|:-----------------------:|:-------------:|---|
|                                                                                                          |                              |   |     Precision    |     Recall    |   |      Precision     |     Recall    |   |     Precision    |     Recall    |   |     Precision    |     Recall    |   |     Precision    |     Recall    |   |         Precision        |     Recall    |   |     Precision    |     Recall    |   |         Precision        |     Recall    |   |         Precision       |     Recall    |   |
|     Random Forest       |             0.9259           |   |       0.934      |      0.984    |   |        0.913       |      0.943    |   |       0.945      |      0.972    |   |       0.865      |      0.902    |   |       0.968      |      0.938    |   |           0.929          |      0.929    |   |         1        |      0.827    |   |           0.934          |      0.86     |   |           0.859         |      0.705    |   |
|     k-NN                |             0.8796           |   |       0.946      |      0.94     |   |        0.874       |      0.884    |   |       0.914      |      0.952    |   |       0.841      |      0.796    |   |         1        |      0.909    |   |           0.842          |      0.885    |   |        0.87      |      0.769    |   |           0.838          |      0.801    |   |           0.681         |      0.722    |   |
|     CNN                 |             0.8788           |   |       0.929      |      0.941    |   |        0.846       |      0.928    |   |       0.925      |      0.965    |   |       0.725      |      0.804    |   |       0.828      |      0.75     |   |           0.747          |      0.717    |   |       0.732      |      0.577    |   |            0.81          |      0.693    |   |           0.741         |      0.663    |   |
|     RBF-SVM             |             0.8761           |   |       0.921      |      0.944    |   |        0.855       |      0.885    |   |         1        |      0.963    |   |       0.879      |      0.624    |   |       0.967      |      0.879    |   |           0.955          |      0.972    |   |       0.542      |      0.907    |   |           0.832          |      0.796    |   |           0.768         |      0.676    |   |
|     Logistic Regression |             0.8643           |   |       0.909      |      0.943    |   |        0.808       |      0.884    |   |       0.951      |      0.972    |   |       0.913      |      0.793    |   |       0.939      |      0.969    |   |           0.919          |      0.919    |   |        0.93      |      0.769    |   |           0.732          |      0.66     |   |           0.747         |      0.621    |   |
