import pickle
import re
import uvicorn
import torch
import transformers

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field


app = FastAPI()


class TextInput(BaseModel):
    text: str = Field(min_length=2, description="The input text") # Validator for empty or one char strings


class Prediction(BaseModel):
    is_italian: bool


# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
file = open('128_weights.pkl', "rb")
model = pickle.load(file)

# Set the device to use for inference (GPU if available, otherwise CPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)
model.eval()


def is_special_characters_only(text):
    pattern = r'^[\W\d]+$'
    return bool(re.match(pattern, text))

@app.post("/predict")
def predict(text: TextInput) -> Prediction:

    if is_special_characters_only(text.text):
        raise HTTPException(status_code=400, detail="The input must contain words")

    # Tokenize input text
    try:
        inputs = tokenizer.encode_plus(
            text.text,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )
    except:
        raise HTTPException(status_code=400, detail="The input can't be tokenized")

    # Make prediction
    try:
        with torch.no_grad():
                outputs = model(input_ids= inputs['input_ids'].to(device))[0]
                prediction = torch.sigmoid(outputs).item() >= 0.5
                return Prediction(is_italian=prediction)

    except:
        raise HTTPException(status_code=401, detail="Model Error")

if __name__ == '__main__':
    # server api
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)