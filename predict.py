import pickle
from fastapi import FastAPI
from typing import Dict, Any
import uvicorn

app = FastAPI(title="lead-score-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(lead):
    result = pipeline.predict_proba(lead)[0, 1]
    return float(result)

@app.post("/predict")
def predict(lead: Dict[str, Any]):
    prob = predict_single(lead)

    return{
        "converted_probability": prob,
        "converted": bool(prob >= 0.5)
    }

## Q3      
#record = {
#    "lead_source": "paid_ads",
#    "number_of_courses_viewed": 2,
#    "annual_income": 79276.0
#}

#print(predict_single(record))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)