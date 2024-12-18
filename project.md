---
title: Predicting Survival for Patients with Severe Illness
---

I applied machine learning techniques clinical data from patients with severe illnesses to predict whether or not they will pass away in the next 6 months. 

## Introduction 

For patients with severe illness, one of the major decisions that individuals, healthcare workers, loved ones may have to make is whether or not to prolong the care they are recieving, especially for those on life-support. According to a report by KFF, about 71% of people in the U.S. believe that helping people die without pain, stress, and discomfort is more important than prolonging someone's life as much as possible [^1]. Furthermore, the SUPPORT Principal Investigators ran a controlled trial to observe and document aspects of hospital deaths. They found that "for 50% of conscious patients who died in the hospital, family members reported moderate to severe pain at least half the time" and "only 47% of physicians knew when their patients preferred to avoid CPR". This study also found no improvement to communication between patients and physicians despite an intervention that included a specialized nurse to contact the patient and family.[^2] Therefore, being able to predict survival for patients who need end-of-life care can help those around them make informed decisions about their care, such as whether or not to continue life-support, sign a Do Not Resuscitate (DNR) form, or start hospice care.

This project uses a Linear Support Vector Classifier (LinearSVC) on the data collected by the SUPPORT Principal Investigators, which includes information about the patients' type of disease, age, vital signs, and other information, to predict whether or not a patient will pass away within the next six months. We then compared the accuracy of my model to the accuracy of LinearSVCs that only used combinations of the SUPPORT Model (as developed by the SUPPORT Principal Investiagors) and doctor's prognoses for six month survival. We concluded that we got the best results from the LinearSVC that used as many featurs as possible and that the SUPPORT Model was slightly more accurate than just doctors' prognoses.

## Data and Preprocessing

That data used in this project is from the UCI Machine Learning Repository. It was originally collected from 1989-1994, however, it was only put on the UCI Machine Learning Repository in late 2023. The dataset contains information about patients' age, race, sex, disease type, vital signs, if/when they signed a DNR, presence of other diseases such as diabetes, dementia, or cancer, etc (See [^3] for a comprehensive description of each of the 48 total features). The target feature is 'death', which is a 1 if the patient passed away within 180 days following the study, or 0 if they survived.

To pre-process this data, first I filled in some missing values with the [suggested values](https://archive.ics.uci.edu/dataset/880/support2#:~:text=Baseline%20Variable%09Normal%20Fill%2Din%20Value%0A%2D%20Serum%20albumin%20(alb)%093.5%0A%2D%20PaO2/FiO2%20ratio%20(pafi)%20%09333.3%0A%2D%20Bilirubin%20(bili)%091.01%0A%2D%20Creatinine%20(crea)%091.01%0A%2D%20bun%096.51%0A%2D%20White%20blood%20count%20(wblc)%099%20(thousands)%0A%2D%20Urine%20output%20(urine)%092502) [^3].

```python
#these are vital statistcs
df=df.fillna({'alb':3.5,'pafi':333.3,'bili':1.01,'crea':1.01,'bun':6.51,'wblc':9,'urine':2502})
```
Next, I dropped the columns for the target data we are not looking to model, as well as columns that had a significant number of missing values or shouldn't directly affect a patient's survival rate.

```python
#drop columns with not enough data
df=df.drop(columns=['adlp', 'adls', 'edu', 'income','glucose','ph','totcst','totmcst'])
#in order these columns represent: activity level as reported by the patient
                                   activity level as reported by family/others
                                   years of education
                                   glucose level
                                   blood ph level
                                   total cost of hospital stay
                                   total cost estimate (micro cost)

#drop columns that don't (intuitively) directly affect survival rates
df=df.drop(columns=['dzclass'])

#'dzclass' is the generalization of 'dzgroup' (the type of disease), so it is repeat data
```

Finally, since many of the columns were categorical data, I used the pandas method `pd.get_dummies(df)` to convert categorical data into numerical data. When we convert the data, we also make sure to drop the first column to avoid multicolinearity[^4]

```python
df=pd.get_dummies(df,drop_first=True)
```
After dropping any rows with missing data, we were left with 7237 samples.

## Initial Observations
To get a better idea of if this dataset would work for a machine learning model, I made some plots to examine the relationship between a few of the features and the target (predicting patient survival):

The description for this dataset notes that daily activity level and low number of comorbidites(other diseases also present in the patient) are correlated with higher survival rates. To investigate this, I made some plots to compare the proportion of patients that do not survive by their activity level or number of comorbidities.

![](assets/IMG/Comorbidities.png) 

*Figure 1: Each column represents a number of comorbidites and the value of the column represents the proportion of people with that number of comorbidities that died within 6 months of the conclusion of the study.*

Notice that of the patients with 0 comorbidities, about 49% of them died within 6 months. However, the rates of death for the other numbers of comorbidities were fairly similar, albeit higher. This suggests that having any comorbidities makes you more likely to pass away soon. Both of the people who had 8 comorbidities died, so the rate of death for patients with 8 comorbidities is 100%. This matches our expectation that patients with more comorbidities do not survive as long.

The next graph compares the rates of death based on Actvities of Daily Life.

![](assets/IMG/ADL.png)

*Figure 2: Each column represents a range of ADL corresponding to the value on the x axis. Higher ADL suggests a more active lifestyle.The value of the column represents the proportion of people in that range of ADL that died within 6 months of the conclusion of the study.*

![](assets/IMG/ADL.png)

*Figure 3: Each column represents a range of ADL corresponding to the value on the x axis. Higher ADL suggests a more active lifestyle.The value of the column represents the total number of people in that range of ADL that died within 6 months of the conclusion of the study.*

In Figure 2, we notice that it seems like more active people are less likely to survive. This is counter-intuitive because we expect that more active people would be on average healthier. I believe that this discrepancy is because people who are more active are less likely to have a severe illness in the first place, which means that they wouldn't even be included in the study. This is supported by how few patients there are with high ADL (Figure 3). Therefore, those that have a higher ADL might be more likely to die within 6 months because their illnesses have to be worse to affect them to the same degree as their peers who have less active lifestyles.

Finally, I compared the 2 month and 6 month estimates from doctors and the SUPPORT model to the target. The 

![](assets/IMG/2m.png)

*Figure 4: Doctor and SUPPORT model probabilities for survival after 2 months compared with whether or not the patient survived after 6 months.*

![](assets/IMG/6m.png)

*Figure 5: Doctor and SUPPORT model probabilities for survival after 6 months compared with whether or not the patient survived after 6 months.*

From Figure 4 and Figure 5, we see that the estimates from doctors and the SUPPORT model correlate loosely to the survival of a patient, which we expect, because the numbers are the estimated percent of survival after the specified number of months. I will be comparing the accuracy of a model created using just these estimates to one that uses as many features as possible. 

## Modeling

Given that we are predicting a discrete value (1 or 0 for 'died within 6 months' and 'survived'), and we were able to convert the categorical data into numerical data, is appropriate to use a support vector classifier(SVC) (I tested both linear and non-linear classifiers), logistic regressor, or random forest classifier. Each of these are appropriate for this problem because they are able to take many features and produce a model that can predict values of 1 and 0. In particular, the nonlinear SVCs and random forest classifier are able to create decision boundaries that allow for non-linear relationships between the features and the target. This is important because the factors that determine people's health aren't always linear. For example, a blood pressure that is too high or too low both lead to poor health.

Because there are so many samples (7237), I chose to use a test-train split of 20% test and 80% training data with 5-fold cross-validation. This helps prevent overfitting and leads to a model that performs better when asked to make a prediction on new data. Additionally, I applied the standard scaler to the support vector models because many of the features use units unique to that quantity, especially for the vital signs. For the Random Forest Regressor, normalization is not necessary, so that step was skipped.

I tested each classifier and foundt they all performed similarly when trained on the data with as many features as possible, which makes sense because in some cases, such as logistic regressor and LinearSVC, the two models can be the same. 

| Model        | RMSE    | Accuracy |
| :----------- | :-----: | :------: |
| LinearSVC    | 0.45525 | 0.76934  |
| RBF SVC      | 0.45809 | 0.75829  |
| Logit        | 0.45335 | 0.76934  |
| RFC          | 0.45240 | 0.76588  |

Because the LinearSVC trains relatively quickly and had a slightly better accuracy, I will proceed with just the LinearSVC model. I will also be using this model on the features that list the SUPPORT model and doctors' estimates on how likely a patient is to survive in the next 6 months. To ensure that each of these LinearSVCs(the one with as many features as possible, only doctor estimates, and only SUPPORT model estimates) is trained on the same set of patients, I fixed the random state for the test-train-split and K-fold validation.

 ```python
X=df.drop(columns=['death']) #overall features
y=df['death'] #target, stays the same for all models

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=r) #r is fixed for repeatability

X1_train=X_train.drop(columns=['prg2m','prg6m','surv2m','surv6m']) #for creating the model with as many features as possible
X2_train=X_train[['surv6m']] #features for testing against SUPPORT model
X3_train=X_train[['prg6m']] #features for testing against Doctors' estimates

X1_test=X_test.drop(columns=['prg2m','prg6m','surv2m','surv6m'])
X2_test=X_test[['surv6m']]
X3_test=X_test[['prg6m']]
```

## Results

With the LinearSVC trained on as many features as possible (excluding Support model and Doctors' estimates), I achieved a test accuracy of 76.934%. I will refer to this model as the "All Features SVC". The Support model and Doctors' estimates, when used to train their own Linear SVCs, achieved test accuracies of 71.616% and 72.928%, respectively. I will refer to these as the "Support Model SVC" and "Doctor SVC", respectively as well.

| Features Used                | RMSE    | Accuracy    |
| :----------------------------| :-----: | :---------: |
| As many as possible          | 0.45525 | 0.76933701  |
| Support Model Estimates      | 0.52238 | 0.71616022  |
| Doctor Estimates             | 0.53000 | 0.72928176  |

![](assets/IMG/ConfusionMatrix.png)

*Figure 6: Confusion Matrix for the All Features SVC*

Plotting the confusion matrix, we see that we have a slightly higher rate of false positives(190) than false negatives(144). Due to there being about 40 features in the All Features SVC, I decided to investigate which features are the most important and compare this model to one that only uses the most significant features. Below is a plot of the top 10 features and their coefficient from the model.

![](assets/IMG/Significance.png)

*Figure 7: Graph of the most signficant features from the All Features SVC. Bars with the same color or that have a label inside the bar indicates that the feature originally came from the categorical feature indicated by the label in the bar before being converted into dummy variables.*

From the graph of the most significant features, we see that the most important features are the disease group, DNR status, presence of cancer, age, and average cost of ICU treatments. From these, I created two more LinearSVC models, one that uses *only* the top 10 features, and one that uses the top 10 features and any features that came from their original category. 

| Features Used                         | RMSE    | Accuracy    |
| :-------------------------------------| :-----: | :---------: |
| Top 10 only                           | 0.48643 | 0.73342541  |
| Top 10 and most signficant categories | 0.47928 | 0.74240331  |
| As many as possible                   | 0.45525 | 0.76933701  |
| Support Model Estimates               | 0.52238 | 0.71616022  |
| Doctor Estimates                      | 0.53000 | 0.72928176  |

Compared to the model that uses as many features as possible, the model with only the top 10 features is about 3.59% less accurate and the model that uses the top 10 features and any features that came from their original category is about 2.69% less accurate.

![](assets/IMG/ROCcurve.png)
*Figure 8: The ROC curve for each of the Linear SVCs using the indicated features. Using only the top 10 features does not significantly reduce the model's accuracy.*

From the ROC curves and Accuracy measurements, it is clear that while the model that predicts most accurately is the one that uses all available features, it is sufficient to consider the main categories of disease group, DNR status, presence of cancer, age, and average cost of ICU treatments. Additionally, the ROC curve supports the observation that using only the most signficant features still yields better results than the SUPPORT Model SVC and Doctor SVC.

## Discussion

Despite testing various models, none of them achieved particularly high accuracy. Even though the LinearSVC that used as many features as possible was more accurate, it only about 5% more accurate than predicting based on doctors' and the SUPPORT Model's estimates. Thankfully, this means that we can still trust doctors to provide decent estimates of 6 month survival rates. Additionally, becuase this difference is so small, I do not believe that using a LinearSVC, as I have done here, will prove to be any more effective than current methods of predicting patient survival will be.

I believe that one of the reasons the tested models did not produce better results is that the target data is simple binary classification. As a result, each of the model's methods converged on similar features and numbers, producing models with nearly identical accuracy. I believe that the accuracy was also hindered by considering every type of disease. it is possible that limiting the dataset to patients with a specific disease will produce more accurate results. In addition, this data was originally collected in the 1990s, so I believe that with the improvements to technology today, we would see a higher percentage of patients surviving than as indicated from this model. Furthermore, this data was collected from a study that took special care to attempt to facilitate better patient-doctor communication, so survival rates in a real-world applications of this model could be lower than predicted.

One surprising result is that features associated with higher survival, those being a lack of comorbidities and higher activity levels, were not significant features in the All Features model. This supports the initial observation that the nubmer of comorbidities is not very relevant to survival unless the patient has 0 comorbidities, in which case they would have a slightly better chance of survival. Similarly, there are few patients with high daily activity levels, so that feature becomes less significant in the model.

Another interesting observation is that race is a significant factor in predicting patient survival. This is reflective of the racial inequality in healtcare between white and non-white populations in the U.s.

## Conclusion

Based on an analysis of the accuracy of various models to predict patient survival within 6 months, we conclude that
* race, presence of cancer, age, and type of disease are some of the most significant factors affecting patient survival.
* LinearSVC and similar classifiers for binary classification do not perform much better than existing models, specifically the SUPPORT Model developed alongside this dataset.
  
In the future, the results of this LinearSVC model can be compared to those of a to a neural net. In addition, developing a model to predict survival that is usable by patients, that is, only uses features that patients would have reasable access to measuring, such as weight, age, and presence of specific diseases, could be helpful for assisting in end-of-life decision making or encouraging patients to reach out to their doctor.

## References

[^1]: [B. Wu, “Views and Experiences with End-of-Life Medical Care in the U.S.,” KFF, Apr. 27, 2017. https://www.kff.org/report-section/views-and-experiences-with-end-of-life-medical-care-in-the-us-findings/](https://www.kff.org/report-section/views-and-experiences-with-end-of-life-medical-care-in-the-us-findings/)

[^2]: [A. F. Connors, et.al, “A Controlled Trial to Improve Care for Seriously III Hospitalized Patients,” JAMA, vol. 274, no. 20, p. 1591, Nov. 1995, doi: https://doi.org/10.1001/jama.1995.03530200027032.] (https://jamanetwork.com/journals/jama/article-abstract/391724)

[^3]: [F. Harrel, “SUPPORT2,” UCI Machine Learning Repository, Sep. 14, 2023. https://archive.ics.uci.edu/dataset/880/support2](https://archive.ics.uci.edu/dataset/880/support2)

[^4]: [SandhyaKrishnan02, “Multicollinearity, how to handle it to avoid dummy variable trap?,” www.kaggle.com, Jan. 2021. https://www.kaggle.com/discussions/general/294096](https://www.kaggle.com/discussions/general/294096)
[back](./)

