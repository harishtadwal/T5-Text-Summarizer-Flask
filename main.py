
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import torch
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# Initialize FastAPI
app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization Using T5",
    version="1.0"
)


# Configure static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# Configure templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# Load trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(
    BASE_DIR / "saved_summary_model"
)

tokenizer = T5Tokenizer.from_pretrained(
    BASE_DIR / "saved_summary_model"
)


# Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


# Move model to selected device
model.to(device)
model.eval()

print(f"Model loaded successfully on: {device}")


# Input schema
class DialogueInput(BaseModel):
    dialogue: str


# Clean input text
def clean_data(text):

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip().lower()

    return text


# Summarization function
def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    # Tokenization
    inputs = tokenizer(
        dialogue,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Generate summary
    with torch.inference_mode():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    # Decode summary
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# Summarization API
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    if not dialogue_input.dialogue.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter some text."
        )

    try:
        summary = summarize_dialogue(dialogue_input.dialogue)

        return {
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
