# Spam-Email-detection
spam email detection using R

Model to Identify spam mails.
Data set source :  https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/

1) Examined how each predictor differs between spam and non spam emails by comparing spam class average and non spam class average. 
Identified top 10 predictors with highest difference between spam and non spam average

2) Partitioned data into 80% training and 20% validation set. Performed Linear discriminant analysis including only the top 10 predictors identified in previous step.

3) Evaluated effectiveness of model using confusion matrix, lift chart and decile chart. 
