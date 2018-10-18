import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import math

# Main reason for going with Random Forest is because of it's ability to handle
# features which are co-related

NY_LAT = 40.785091
NY_LON = -73.968285

# Haversine equation for calculating distance
def get_distance_from_centre(lat_srs, lon_srs):
    p = np.pi / 180
    dist_srs = 0.5 - (np.cos((lat_srs - NY_LAT) * p) / 2) + ((np.cos(lat_srs * p) * math.cos(NY_LAT * p) * (1 - np.cos((lon_srs - NY_LON) * p))) / 2)
    return 12742 * np.arcsin(np.sqrt(dist_srs))


class Model:
    """Multinomial logistic regression model that can be trained on flexible
    attribute list.
    """
    def __init__(self, data):
        self.select_training_features(data)
        print("Before sanitization:", self.train.shape[0])

        self.sanitization()
        print("After sanitization:", self.train.shape[0])

    def select_training_features(self, data):
        # Numeric attributes
        self.train = data[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']].copy()
        self.attributes = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']

        self.train['interest_level'] = data['interest_level']

        # Number of features
        self.train["num_features"] = data.features.apply(len)
        self.attributes.append("num_features")

        # Number of photos
        self.train["num_photos"] = data.photos.apply(len)
        self.attributes.append("num_photos")

        # Number of words in description
        self.train["num_desc"] = data.description.apply(lambda x: len(x.split(" ")))
        self.attributes.append("num_desc")

        # Extracting created details
        temp = pd.to_datetime(data.created)
        self.train["created_year"] = temp.dt.year
        self.train["created_month"] = temp.dt.month
        self.train["created_day"] = temp.dt.day
        self.attributes.extend(["created_year", "created_month", "created_day"])

        # Generating distance (from centre)
        self.train["distance"] = get_distance_from_centre(data.latitude, data.longitude)
        self.attributes.append("distance")

        # Manager Id
        self.train["manager_id"] = data.manager_id

        # Getting manager ability
        # Generating counts of each interest_level grouped by manager
        man_df = pd.crosstab(data.manager_id, data.interest_level)
        # Generating the weights as inversely proportional to number of instances
        intgp = data.groupby('interest_level').size()
        intgp = intgp.apply(lambda x: data.shape[0] / x)
        # Create man_ability as a weighted sum
        man_df['man_ability'] = sum([intgp[i] * man_df[i] for i in ['high', 'low', 'medium']])
        self.man_df = man_df # We store man_df so that it can be used for predictions afterwards
        # Merge the previous data set to add man_ability
        self.train = pd.merge(self.train, man_df[['man_ability']], on='manager_id')
        self.attributes.append("man_ability")

    def sanitization(self):
        tr = Transform(self.train)

        tr.bound_drop('price', 15046)
        tr.bound_drop('distance', 23)
        tr.bound_drop('distance', 0.14, False)

    def select_features(self, data):
        # Numeric attributes
        df = data[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'listing_id']].copy()

        # Number of features
        df["num_features"] = data.features.apply(len)

        # Number of photos
        df["num_photos"] = data.photos.apply(len)

        # Number of words in description
        df["num_desc"] = data.description.apply(lambda x: len(x.split(" ")))

        # Extracting created details
        temp = pd.to_datetime(data.created)
        df["created_year"] = temp.dt.year
        df["created_month"] = temp.dt.month
        df["created_day"] = temp.dt.day

        # Manager Id
        df["manager_id"] = data.manager_id

        # Generating distance (from centre)
        df["distance"] = get_distance_from_centre(data.latitude, data.longitude)

        # Getting manager ability
        mn = self.man_df.man_ability.mean()
        df["man_ability"] = df.apply(lambda x: self.man_df.man_ability.get(x.manager_id, mn), axis=1)

        return df


    def fit(self, n_estimators=1000):
        self.model = RF(n_estimators=n_estimators, random_state=200)
        self.model.fit(self.train[self.attributes], self.train['interest_level'])

    def predict(self, data):
        features = self.select_features(data)
        prob = self.model.predict_proba(features[self.attributes])

        df = pd.DataFrame(prob, columns=self.model.classes_)
        return df.assign(listing_id=features['listing_id'].values)


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


def train_model():
    # Read from data files
    train = pd.read_json('../input/train.json')

    md = Model(train)
    md.fit()

    return md


def create_submission(model, filename):
    test = pd.read_json("../input/test.json")

    output = model.predict(test)
    print("File:", filename)
    print(output.shape)
    print(output.head())

    output.to_csv(filename, index=False)

model = train_model()
create_submission(model, "basic_random_forest.csv")
