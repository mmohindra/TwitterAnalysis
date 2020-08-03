from fastapi import  FastAPI
import pickle


import numpy as np




app = FastAPI()

@app.get("/pingtest")
def pingtest():
    return {"Hello message": "Ping success"}

@app.get("/pingteststr/{intext}")
def pingteststr(intext):
    return {"Hello message": intext}

@app.post("/predictpost/")
def predictpost(text: str):
    model = initialize_model()

    prediction = predict(model, text)
    print(prediction)
    return ({
        "Tweet": text,
        "Sentiment": prediction
    })

@app.get("/predict_tweet/{text}")
async def predict_tweet(text: str):
    model = initialize_model()

    prediction = predict(model,  text)
    print(prediction)
    return ({
        "Tweet": text,
        "Sentiment": prediction
    })

def initialize_model():
    model = model_pipe = pickle.load(open('basicmodel_pipe.pkl', 'rb'))
    return model

def predict(model, text):
    textlist = [text]
    pred = model.predict(textlist)

    if (pred.item() == 0):
        predstr = 'prediction negative sentiment'
    else:
        predstr = 'prediction positive sentiment'
    return predstr