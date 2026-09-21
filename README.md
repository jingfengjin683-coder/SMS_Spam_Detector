SMS Spam Detector
is a machine learning project that classfies SMS messages as spam or ham
with a simple Streamlit interface. This project uses the UCI SMS Spam Collection. It
also evaluate predictions with precision, recall. F1-score and a confussion matrix. Logistic
regression overall have a better performance compare to Naive Bayes. I have
add the influence of word pairs to the model and there is an increase in precision, recall, F1-score.

Result:
Naives Bayes
Spam precision: 1.000
Spam recall:    0.603
Spam F1:        0.751

Logistic regression
Spam precision: 0.900
Spam recall:    0.931
Spam F1:        0.915

Logistic regression + word pairs
Spam precision: 0.919
Spam recall:    0.939
Spam F1:        0.929

Confusion matrix:
Predicted ham actually ham 892,
predicted spam actually ham 12,
predicted spam actually ham 12,
predicted spam actually spam 119

The app display the word and pairs that influence each prediction.
For each recognized feature, its contributions is calculated as: 
TF-IDF value * learned logistic-regression weight. Positive contributions
push toward spam negative contributions push toward ham. The final score includes
akk contributions plus the model's learned starting offset.
