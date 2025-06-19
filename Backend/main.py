from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import pandas as pd 

app = FastAPI()

# Enable CORS (for frontend-backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("LinearRegressionModel.pkl", "rb") as f:
    model = pickle.load(f)




# Mount static files like CSS
app.mount("/static", StaticFiles(directory="../Frontend/static"), name="static")

# HTML templates folder
templates = Jinja2Templates(directory="../Frontend/templates")


# Home page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    name: str = Form(...),
    company: str = Form(...),
    year: int = Form(...),
    kms_driven: int = Form(...),
    fuel_type: str = Form(...)
):
    try:
        input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)
        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

# For running via terminal: uvicorn main:app --reload
