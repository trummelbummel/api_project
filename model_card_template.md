# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Responsible Developer: Theresa Fruhwuerth
Date of training: 5.5.2023
Model version: 1.0.0
Model type: Random forest Classifier from SKLearn library.
Training Details:
1. Parameters: estimators=100, max_depth=2

For questions contact: complain.aboutmymodel@gmail.com


## Intended Use

This model should be used to predict the salary for a given sample of demographic data.
Primary users would be anyone interested in earnings potential based on demographic variables
such as education, age, sex, origin, occupation and marital status e.g. policy makers to understand systematic
disadvantages of certain communities.
Out of scope would be the determination of exact salary as ranges (above 50 k and below 50 k)
are used.

## Training Data

20% (3135) of the total dataset of 15678 samples (After balancing) where used. 
Steps to preprocess the data:
2. Features: cat_features in train_model.py are one hot encoded 
3. Cleaning steps: duplicates dropped from the data, rebalancing the data with respect to the target
   variable salary.

Training data contains the following variables (for more information look into : ./data/census.csv)
* age
* workclass 
* fnlgt 
* education
* education-num 
* marital-status 
* occupation, 
* relationship 
* race 
* sex 
* capital-gain 
* capital-loss 
* hours-per-week 
* native-country 
* salary

## Evaluation Data

Since there is an imbalance in the target with only 25% being in the below 50k salary, 
we rebalanced the data.

The model is evaluated with precision, recall and f score, on the original distribution of the test dataset
hence  the evaluation might not be completely representative when the data is imbalanced in reality. 
We used 20% of the original dataset i.e. 6508 of the original 32537 datapoints as evaluation data.


## Metrics
_
Please include the metrics used and your model's performance on those metrics._
Factors for evaluation:
Evaluation was done checking for features such as race.

Overall metrics are:

precision:0.8307405102675793
recall: 0.4769560557341908
fbeta:0.6059918293236496
  

## Ethical Considerations

The model is clearly biased and the performance on Other group.
Furthermore the Asian population seems to be easiest to predict.

## Caveats and Recommendations

Best would be to collect more data for underrepresented groups
and retrain the model. Predictions for these two classes should not be 
used for policy decisions.


