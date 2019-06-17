# AIForSEA_TrafficManagement
## Environments
```
python 3.6.5
```

To install require libraries, run:
```
pip install -r requirements.txt
```

## Solution to the problem
To handle the time series problem, my solution is to use the previous day information to create new feature, and transform the problem from time series forecasting to a normal supervised learning problem

Features are evaluated by using Light Gradient Boosting machine to inpect the importance of each feature on the demand, and combine with RFE to ensure the result.

Model of choice was a StackingRegressor of multiple linear model (Linear Regression, LASSO, Ridge, SVM linear, LGBM) and use SVM linear to control the effect of each model

Feature selection can be view more carefully in ```exploratory_data_analysis.ipynb```

## Result 
After training 50 day (from 1 to 51), I recreate the test environment by using day 43 to 57 and predict on day 58 to 61

The result I achieved: ```0.1688```

## Improvements
* The result is still need improvement
* Haven't implement validation to monitor overfitting
* Code still too messy 
* The result still in float value, however, the main goal of the problem is to monitor traffic, therefore it would be better if we visualize the density of Traffic after prediction. This is the ultimate goal.
