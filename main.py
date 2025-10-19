from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle
from pydantic import BaseModel, Field
from typing import Annotated
import pandas as pd

model = pickle.load(open("pipe2.pkl", "rb"))
el = pickle.load(open("el1.pkl", "rb"))

app =  FastAPI()
class User(BaseModel):
    crop:Annotated[str,Field(...,description="type of crop for pesticide")]
    soil_type:Annotated[str,Field(...,description="soil type")]
    temp:Annotated[float,Field(...,gt=0,description="temperature")]
    humi:Annotated[float,Field(...,gt=0,description="humidity")]
    pest:Annotated[str,Field(...,description="pest type")]
    area:Annotated[float,Field(...,description="acrea of land")]
    pesticide_lbs:Annotated[float,Field(...,description="pesticide")]

@app.get("/")
def home():
    return {"message":"AI crop pesticide API"}    


@app.post("/predict1")
def predict_pest(data:User):
    input= pd.DataFrame([{
        "Crop":data.crop,
        "Soil_Type":data.soil_type,
        "Temperature_C":data.temp,
        "Humidity_%":data.humi,
        "Pest_Type":data.pest,
        "Area_Acres":data.area,
        "Pesticide_Lbs":data.pesticide_lbs
    }])
    probas = model.predict_proba(input)[0]
    crop_names = el.classes_ 

    crop_probs = list(zip(crop_names, probas))
    top_crops = sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]

    formatted_output = [
        {"crop": crop.title(), "suitability": round(prob * 100, 2)} for crop, prob in top_crops
    ]

    x = [f"✅ {item['crop']} – {item['suitability']}%" for item in formatted_output]

    return JSONResponse(status_code=200,content={"top_pesticide": x})