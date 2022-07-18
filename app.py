
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
# df = pickle.load(open('df.pkl', 'rb'))

df = pd.read_csv('laptop_data.csv')
df = df.drop(columns=['Unnamed: 0'])
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
df['Ram'] = df['Ram'].astype(int)
df['Weight'] = df['Weight'].astype(float)
df['TouchScreen'] = df['ScreenResolution'].apply(
    lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
new = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['X_res'] = new[0]
df['Y_res'] = new[1]


def take_only_num(x_res):
    L = ""
    for i in x_res:
        if i.isnumeric() == True:
            L += i
    if len(L) > 4:
        L = int(L)//10
    return str(L)


df['X_res'] = df['X_res'].apply(take_only_num)
df['X_res'] = df['X_res'].astype(int)
df['Y_res'] = df['Y_res'].astype(int)
df['ppi'] = ((df['X_res']**2)+(df['Y_res']**2))**0.5/df['Inches']
df.drop(columns=['ScreenResolution'], inplace=True)
df.drop(columns=['Inches', 'X_res', 'Y_res'], inplace=True)
df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


df['Cpu Brand'] = df['Cpu Name'].apply(fetch_processor)
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '')
df['Memory'] = df['Memory'].str.replace('TB', '000')
new = df['Memory'].str.split("+", n=1, expand=True)

df['First'] = new[0]
df['First'] = df['First'].str.strip()

df['Second'] = new[1]

df['Layer1HDD'] = df['First'].apply(lambda x: 1 if "HDD" in x else 0)
df['Layer1SSD'] = df['First'].apply(lambda x: 1 if "SSD" in x else 0)
df['Layer1Hybrid'] = df['First'].apply(lambda x: 1 if "Hybrid" in x else 0)
df['Layer1Flash_Storage'] = df['First'].apply(
    lambda x: 1 if "Flash Storage" in x else 0)

df['First'] = df['First'].str.replace(r'\D', '')

df['Second'].fillna("0", inplace=True)

df['Layer2HDD'] = df['Second'].apply(lambda x: 1 if "HDD" in x else 0)
df['Layer2SSD'] = df['Second'].apply(lambda x: 1 if "SSD" in x else 0)
df['Layer2Hybrid'] = df['Second'].apply(lambda x: 1 if "Hybrid" in x else 0)
df['Layer2Flash_Storage'] = df['Second'].apply(
    lambda x: 1 if "Flash Storage" in x else 0)

df['Second'] = df['Second'].str.replace(r'\D', '')

df['First'] = df['First'].astype(int)
df['Second'] = df['Second'].astype(int)

df['HDD'] = (df['First']*df['Layer1HDD']+df['Second']*df['Layer2HDD'])
df['SSD'] = (df['First']*df['Layer1SSD']+df['Second']*df['Layer2SSD'])
df['Hybrid'] = (df['First']*df['Layer1Hybrid']+df['Second']*df['Layer2Hybrid'])
df['Flash_Storage'] = (df['First']*df['Layer1Flash_Storage'] +
                       df['Second']*df['Layer2Flash_Storage'])

df.drop(columns=['First', 'Second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
        'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'], inplace=True)

df.drop(columns=['Memory'], inplace=True)
df.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True)
df['Gpu Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df = df[df['Gpu Brand'] != 'ARM']
df.drop(columns=['Gpu'], inplace=True)


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'MacOS'
    elif inp == 'Linux':
        return 'Linux'
    else:
        return 'Others/No OS'


df['os'] = df['OpSys'].apply(cat_os)
df.drop(columns=['OpSys'], inplace=True)


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
