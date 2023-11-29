from fastapi import FastAPI, File, UploadFile, HTTPException

from pydantic import BaseModel
from typing import List
import pandas as pd
import re
import pickle
import sklearn
import logging
from typing import Annotated
import io
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

regexp = r'\d+\.?\,?\d*'

model = None
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def string_to_float(s):
    return float(s.replace(',', '.'))

def extract_float(df_string):
    if pd.isna(df_string):
        return df_string
    findings = re.findall(regexp, df_string)    
    return string_to_float(findings[0]) if len(findings) != 0  else None

def cast_torque_and_max_torque_rpm(df):
    new_columns = pd.DataFrame(df['torque'].apply(extract_torque).tolist(), columns=['torque', 'max_torque_rpm'])
    df['torque'] = new_columns['torque']
    df['max_torque_rpm'] = new_columns['max_torque_rpm']

def encode_onehot(df, columns):
    pd.get_dummies(df,columns=columns)
    df = df.drop(columns, axis=1)
    return df

def cast_columns(df):
    for column in ['mileage', 'engine', 'max_power']:
        df[column] = df[column].apply(extract_float)
    cast_torque_and_max_torque_rpm(df)
    return df

def data_preprocessing(df):
    df = df.drop(['selling_price', 'name'], axis=1)
    df = cast_columns(df)
    
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    df = data_preprocessing(df)
    prediction = model.predict(df)
    return prediction[0]


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    original_df = pd.read_csv(file.file)
    df = original_df.copy()
    for index, row in df.iterrows():
        try:
            Item(**row.to_dict())
        except Exception as e:
            raise HTTPException(detail=f'Error in row {index}: {e}', status_code=403)
    df = data_preprocessing(df)
    predictions = model.predict(df)
    predictions_df = pd.DataFrame(predictions, columns=['predicted_selling_price'])
    new_df = pd.concat([original_df, predictions_df], axis=1)
    stream = io.StringIO()
    new_df.to_csv(stream, index=True)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predicted_price.csv"
    return response