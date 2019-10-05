# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize, Imputer, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

# import dataset
from sklearn.datasets import load_boston

class Modelling:


    def data_preparation(self):


        # load boston dataset
        boston = load_boston()

        # create a pandas dataframe
        df = pd.DataFrame(boston.data)

        # assign features names to columns
        df.columns = boston.feature_names

        # add price values as a column to the dataframe
        df['PRICE'] = boston.target

        CRIM = df.CRIM.unique().tolist()
        ZN = df.ZN.unique().tolist()
        INDUS = df.INDUS.unique().tolist()
        CHAS = df.CHAS.unique().tolist()
        NOX = df.NOX.unique().tolist()
        RM = df.RM.unique().tolist()
        AGE = df.AGE.unique().tolist()
        DIS = df.DIS.unique().tolist()
        RAD = df.RAD.unique().tolist()
        TAX = df.TAX.unique().tolist()
        PTRATIO = df.PTRATIO.unique().tolist()
        B = df.B.unique().tolist()
        LSTAT = df.LSTAT.unique().tolist()

        user_input_list = [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]

        # return clean data
        return df, user_input_list

    def dataTraining(self):

        model = Modelling()
        df, _ = model.data_preparation()

        # split data into X and y - features and target
        X = df.drop('PRICE', axis = 1)
        y = df['PRICE']


        # split data into train, test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

        # Standradize the data

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform (X_test)

        # instantiate classifier
        classifier = GradientBoostingRegressor(learning_rate=0.5, loss='huber')
        # train the model
        classifier.fit(X_train, y_train)

        print("Modelling is done")
        # return the model
        return classifier, X


    def predictUserInput(self, CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT):

        # run training data method
        model = Modelling()
        classifier, X = model.dataTraining()

        # create an empty row
        userInput_df = pd.DataFrame(columns=X.columns)

        # instantiate an empty list
        l = []
        # loop through the length of our empty dataframe
        for i in list(userInput_df.columns):
            # append a value of 0 to the list
            l.append(0)
            
        # plug the first row with 0s 'the list'
        userInput_df.loc[1,:] = l

        # plug user's inputs into the dataframe
        userInput_df['CRIM'] = CRIM
        userInput_df['ZN'] = ZN
        userInput_df['INDUS'] = INDUS
        userInput_df['CHAS'] = CHAS
        userInput_df['NOX'] = NOX
        userInput_df['RM'] = RM
        userInput_df['AGE'] = AGE
        userInput_df['DIS'] = DIS
        userInput_df['RAD'] = RAD
        userInput_df['TAX'] = TAX
        userInput_df['PTRATIO'] = PTRATIO
        userInput_df['B'] = B
        userInput_df['LSTAT'] = LSTAT

        # predict
        prediction = str(classifier.predict(userInput_df)[0])

        return prediction
