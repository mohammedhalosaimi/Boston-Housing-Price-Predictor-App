# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize, Imputer, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression, LassoCV, RidgeCV, SGDRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

# from keras.models import Sequential
# from keras.optimizers import adam, SGD
# from keras.activations import relu, sigmoid, softmax
# from keras.layers import Dense
# from keras.callbacks import EarlyStopping

class Modelling:


    def data_preparation(self):

        # load two data
        df1 = pd.read_csv('../data/otp.csv')
        df2 = pd.read_excel('../data/OTP_Time_Series_Master_Current_0719.xlsx')

        # df 1

        # drop route column since we have it splitted into two columns (Departing_Port, Arriving_Port)
        # drop Month columns since it is splitted into two columns (Year, Month_Num)
        df1.drop(['Route', 'Month'], axis=1, inplace=True)

        # df 2

        # first, convert Month datatype (time-stamp) to string
        df2['Month'] = df2['Month'].astype('str')
        # split Month column into two columns (Month & Year) to match the first dataframe
        df2['Month_Num'] = df2['Month'].str.split('-').str[1].str.lstrip('0')
        df2['Year'] = df2['Month'].str.split('-').str[0]

        # after modifying the columns as we wish, we will convert them to int data type

        df2['Month_Num'] = df2['Month_Num'].astype(int)
        df2['Year'] = df2['Year'].astype(int)

        # df 2

        # drop route column since we have it splitted into two columns (Departing_Port, Arriving_Port)
        # drop Month columns since it is splitted into two columns (Year, Month_Num)
        df2.drop(['Route', 'Month', 'OnTime Departures \n(%)', 'OnTime Arrivals \n(%)', 
                'Cancellations \n\n(%)'], axis=1, inplace=True)

        # df 2

        # set df2 columns names to match df1 columns names setting
        df2.columns = df2.columns.str.replace(' ', '_')

        # concatenate the two dataframes
        df = pd.concat([df1, df2], axis=0, ignore_index=False, keys=None,
                levels=None, names=None, verify_integrity=False, copy=True)

        # fill null value with the mean of the column
        df['Cancellations'].fillna(df['Cancellations'].mean(), inplace=True)
        df['Departures_Delayed'].fillna(df['Departures_Delayed'].mean(), inplace=True)

        # drop all null values
        df.dropna(inplace=True)

        # get numeric and categorical columns
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(include='object').columns.tolist()

        # convert numeric columns to int data type
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.astype(int))

        # convert categorical values to lower case & strip any white space
        df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.lower().str.strip())
        

        Departing_Port = df.Departing_Port.unique().tolist()
        Arriving_Port = df.Arriving_Port.unique().tolist()
        Airline = df.Airline.unique().tolist()
        Month_Num = df.Month_Num.unique().tolist()

        user_input_list = [Departing_Port, Arriving_Port, Airline, Month_Num]

        print("success returning clean data")
        # return clean data
        return df, user_input_list

    def dataTraining(self):

        model = Modelling()
        df, _ = model.data_preparation()

        # dummify data
        dummies_df = pd.get_dummies(df, drop_first=True)

        # get our X and y
        X = dummies_df.drop(['Departures_On_Time', 'Arrivals_On_Time',
                    'Departures_Delayed', 'Arrivals_Delayed', 'Year'], axis=1)
        y = dummies_df[['Departures_On_Time', 'Arrivals_On_Time',
                    'Departures_Delayed', 'Arrivals_Delayed']]


        # split data into train, test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Standradize the data

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform (X_test)

        # instantiate classifier
        classifier = MultiOutputRegressor(SVR())
        # train the model
        classifier.fit(X_train, y_train)

        print("Modelling is done")
        # return the model
        return classifier, X


    def predictUserInput(self, Departing_Port, Arriving_Port, Airline, Month_Num):

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

        # concatenate the parameters with the accutal value in the dataframe since we dummified the dataframe
        Departing_Port = 'Departing_Port_' + Departing_Port
        Arriving_Port = 'Arriving_Port_' + Arriving_Port
        Airline = 'Airline_' + Airline


        # change the value of corresponded columns to 1 if user chooses it
        userInput_df[Departing_Port] = 1
        userInput_df[Arriving_Port] = 1
        userInput_df[Airline] = 1
        userInput_df['Month_Num'] = Month_Num

        # predict
        prediction = classifier.predict(userInput_df)

        return prediction
