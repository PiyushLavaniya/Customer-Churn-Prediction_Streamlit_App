import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def get_model(path):
    scalar = pickle.load(open(path,'rb'))
    model = pickle.load(open(path,'rb'))

    return model
    

def preprocess(df):
    """
    This function is to cover all the preprocessing steps on the churn dataframe. It involves selecting important features, encoding categorical data, handling missing values,feature scaling and splitting the data
    """
    #Defining the map function
    def binary_map(feature):
        return feature.map({'Yes':1, 'No':0, 'Male':1, 'Female':0})

    # Encode binary categorical features
    binary_list = ['SeniorCitizen', 'Gender']
    df[binary_list] = df[binary_list].apply(binary_map)

    
    #Drop values based on operational options
    columns = ['SeniorCitizen', 'Location', 'Gender', 'tenure', 'MonthlyCharges', 'total_GB_usage']
    #Encoding the other categorical categoric features with more than two categories
    df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    df['Location'] = df['Location'].replace(['Los Angeles'],'0')
    df['Location'] = df['Location'].replace(['New York'],'1')
    df['Location'] = df['Location'].replace(['Miami'],'2')
    df['Location'] = df['Location'].replace(['Chicago'],'3')
    df['Location'] = df['Location'].replace(['Houston'],'4')

    #feature scaling
    sc = MinMaxScaler()
    df['tenure'] = sc.fit_transform(df[['tenure']])
    df['MonthlyCharges'] = sc.fit_transform(df[['MonthlyCharges']])
    df['total_GB_usage'] = sc.fit_transform(df[['total_GB_usage']])
    
    return df