import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as Lr

# Multinomial Logistic regression on numeric attributes. Multinomial is used
# since we have to minmize logloss.

class Model:
    """Multinomial logistic regression model that can be trained on flexible
    attribute list.
    """
    def __init__(self, data, attributes):
        self.attributes = attributes
        self.features = data[attributes]
        self.output = data["interest_level"]

    def train(self, num_iter=100):
        self.model = Lr(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=num_iter,
                random_state=200
            ).fit(self.features, self.output)

    def predict(self, data):
        features = data[self.attributes]
        prob = self.model.predict_proba(features)

        df = pd.DataFrame(prob, columns=self.model.classes_)
        return df.assign(listing_id=data['listing_id'].values)

    def score(self, data):
        features = data[self.attributes]
        output = data["interest_level"]

        return self.model.score(features, output)


class Transform:
    """Used to transform DataFrame to sanitize data. All operations transform
    DataFrame.
    """
    def __init__(self, data):
        self.data = data

    def bound_correction(self, attr, bound, upper_bound=True):
        """Any row having 'attr' value greater than 'bound' is reassigned to
        'bound'.
        """
        data = self.data
        if upper_bound:
            index = data[data[attr] > bound].index
        else:
            index = data[data[attr] < bound].index

        data.loc[index, attr] = bound

        return data

    def bound_drop(self, attr, bound, upper_bound=True):
        """Any row having 'attr' value greater than 'bound' is dropped."""
        data = self.data
        if upper_bound:
            index = data[data[attr] >= bound].index
        else:
            index = data[data[attr] <= bound].index

        data.drop(index, inplace=True)

        return data


def sanitize_data(data):
    tr = Transform(data)

    # As seen in EDA notebook, all houses with bathrooms greater than 4 had
    # low interest rate
    tr.bound_correction('bathrooms', 4)

    # Removing outliers in prices
    tr.bound_drop('price', 15000)

    # Removing latitude outliers
    lbound = np.percentile(data.latitude.values, 1)
    ubound = np.percentile(data.latitude.values, 99)
    tr.bound_drop('latitude', lbound, False)
    tr.bound_drop('latitude', ubound)

    # Removing longitude outliers
    lbound = np.percentile(data.longitude.values, 1)
    ubound = np.percentile(data.longitude.values, 99)
    tr.bound_drop('longitude', lbound, False)
    tr.bound_drop('longitude', ubound)

def train_models():
    # Read from data files
    train = pd.read_json('../input/train.json')
    print("Number of rows in training data:", train.shape[0])

    sanitize_data(train)
    print("Number of rows after sanitization:", train.shape[0])

    models = pd.Series()

    # Model 1 - All numeric data
    all_numeric = Model(train, ['bathrooms', 'bedrooms', 'price', 'latitude', 'longitude'])
    all_numeric.train()
    models['all_numeric'] = all_numeric

    # Since 'bathrooms' and 'bedrooms' have high co-relation we use only one of
    # them now

    # Model 2 - Exclude 'bathrooms'
    ex_bathrooms = Model(train, ['bedrooms', 'price', 'latitude', 'longitude'])
    ex_bathrooms.train()
    models['ex_bathrooms'] = ex_bathrooms

    # Model 3 - Exclude 'bedrooms'
    ex_bedrooms = Model(train, ['bathrooms', 'price', 'latitude', 'longitude'])
    ex_bedrooms.train()
    models['ex_bedrooms'] = ex_bedrooms

    # Model 4 - From EDA, 'bathrooms' and 'bedrooms' didn't have much relation with
    # 'interest_level'. So exclude both now.
    ex_bath_bed = Model(train, ['price', 'latitude', 'longitude'])
    ex_bath_bed.train()
    models['ex_bath_bed'] = ex_bath_bed

    score_series = models.apply(lambda x: x.score(train))
    print("Scores on training set:")
    print(score_series)

    return models

def create_submission(data, model, filename):
    output = model.predict(data)
    print("File:", filename)
    print(output.head())

    output.to_csv(filename, index=False)

def save_all_predictions(model_series):
    submission = pd.read_json('../input/test.json')
    print("Number of rows in submission data:", submission.shape[0])

    for index, model in model_series.items():
        create_submission(submission, model, index + ".csv")


models = train_models()
save_all_predictions(models)
