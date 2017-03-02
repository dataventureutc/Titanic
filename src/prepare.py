import pandas as pd

#####################################
# Helpers
#####################################

def debug(df):
    print(df.head())
    exit()

#####################################
# Load data
#####################################

def load_data():

    data = pd.read_csv('data/train.csv')

    # drop rows with empty features / gaps in columns
    data = data.dropna()

    # Categorical values into numerical (one hot encoding)
    one_hot_embarked = pd.get_dummies(data['Embarked'], prefix='embarked')
    data = data.join(one_hot_embarked)

    one_hot_pclass = pd.get_dummies(data['Pclass'], prefix='pclass')
    data = data.join(one_hot_pclass)

    # The sex column has only two values (M/F), so that only one column is required for encoding (0/1)
    # Intead of one hot encoding with two columns
    data['sex'] = data.apply(lambda x: 1 if (x['Sex'] == 'female') else 0, axis=1)

    # Drop features not used for training the model
    data = data.drop(['Cabin', 'Name', 'PassengerId', 'Pclass', 'Sex', 'Ticket', 'Embarked'], axis=1)

    return data.drop(['Survived'], axis=1), data[['Survived']]
