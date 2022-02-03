# LSTMVsOtherRegression--Kaggle--AllStateInsuranceClaims
Comparing results of using deep learning algorithm, LSTMs, for a regression problem versus simple Neural Networks and ensemble algorithms like Random Forest and XGBoost

This article is comparing the results of using a deep learning algorithm, LSTMs, for a regression problem versus simple Neural Networks and ensemble algorithms like Random Forest and XGBoost. The objective is to show that it is important to choose the right algorithm for the problem you are solving. 

Brief explanation of the concepts of ANN, RNN and LSTMs,relevant blogs and research papers are given.

Kaggle competition problem – Allstate Insurance Claims Severity. Objective is to predict the cost (severity) of insurance claims using a large data set of factors (over 130 factors and 180,000 rows of data). This is a regression problem requiring values to be predicted on a continuous scale. The data can be downloaded from here - https://www.kaggle.com/c/allstate-claims-severity/overview.

Sample code for comparing different regression algorithms for this problem is given in github – 
The code follows the standard data science steps (load, wrangle, encode, scale, train/test split, train, predict, metrics). It compares 4 different regression algorithms on accuracy and time taken to run without hyperparameter tuning, except for LSTM.
1.	LSTM network
2.	ANN
3.	XGBoost
4.	Random Forest

The results show that 
•	LSTMs gave the worst results (high RMSE) and also take a long time to run, over 2 hrs
•	XGBoost and a simple 4 layer ANN gave better results (low RMSE) without any hyperparameter tuning. They only took 3-10 minutes to run
•	Random Forest takes over 1 hr and gives results similar to XGBoost
•	Note – running time is on a basic MacBook Pro, will be faster on better CPU/GPU cloud

Submitting and comparing the results on Kaggle against global leaderboard, the overall results are not good
•	No 1 rank score is 1109.7
•	XGBoost score of 1184.5 has rank 2088

The results can be improved by adding more layers to LSTM or ANN networks or hyperparameter tuning for other algorithms, but relatively LSTM will still underperform. That was the objective of this article. 
Hope it helps in further understanding of deep learning algorithms and their applications.
