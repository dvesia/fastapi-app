# Model Card

For additional information, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
## Model Details

This model, created by Domenico Vesia, is a Gradient Boosting Classifier 
that utilizes the default hyperparameters in scikit-learn.
## Intended Use

The model is intended for predicting a person's salary based on specific financial attributes.
## Training Data

The training data is sourced from https://archive.ics.uci.edu/ml/datasets/census+income, 
with 80% of the data being used for training.
## Evaluation Data


The evaluation data is also sourced from https://archive.ics.uci.edu/ml/datasets/census+income, 
with 20% of the data being used for evaluation.
## Metrics

The model's performance was evaluated using the accuracy score, achieving a value of approximately 0.834.
## Ethical Considerations

The dataset contains information related to race, gender, and origin country, which may lead to a model 
that potentially discriminates against certain individuals. 
Further investigation is recommended before using this model.

## Caveats and Recommendations

The dataset's gender classes are binary (male/not male), which we have interpreted as male/female. 
Further work is needed to evaluate the model across a spectrum of genders.