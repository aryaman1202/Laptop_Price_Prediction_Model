import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# data = pd.read_csv('Cleaned_data.csv')
df = pickle.load(open('df.pkl', 'rb'))

X = df.drop(columns=['Price'])
Y = np.log(df['Price'])
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=2)
step1 = ColumnTransformer(transformers=[('col_tnf', OneHotEncoder(
    sparse=False, drop='first'), [0, 1, 7, 10, 11])], remainder='passthrough')
step2 = RandomForestRegressor(
    n_estimators=100, random_state=3, max_samples=0.5, max_features=0.75, max_depth=15)
pipe = Pipeline([('step1', step1),
                ('step2', step2)])
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu Brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    # prediction = pipe.predict(pd.DataFrame([[company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]], columns=[
    #                           'Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen', 'Ips', 'ppi', 'Cpu Brand', 'HDD', 'SSD', 'Gpu Brand', 'os']))
    query = np.array([company, type, ram, weight, touchscreen,
                     ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title("Rs: " +
             str(int(np.exp(pipe.predict(query)[0]))))
    # st.title(prediction[0])
