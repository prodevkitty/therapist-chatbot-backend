from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import BertTokenizer, BertForSequenceClassification
from googletrans import Translator
import torch

app = FastAPI()

# Load the pre-trained BERT model and tokenizer
model_path = '/app/models/your_model_directory'  # Update with the actual path to your model
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Initialize the translator
translator = Translator()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    
    # Translate Portuguese input to English
    translated_input = translator.translate(user_input, src='pt', dest='en').text
    
    # Tokenize the input
    inputs = tokenizer(translated_input, return_tensors='pt')
    
    # Generate response in English
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    
    # Map the predicted class ID to a response (this mapping should be defined based on your model's training)
    response_mapping = {
        0: "I'm here to help you.",
        1: "Can you please elaborate on that?",
        # Add more mappings based on your model's output classes
    }
    english_response = response_mapping.get(predicted_class_id, "I'm not sure how to respond to that.")
    
    # Translate response back to Portuguese
    portuguese_response = translator.translate(english_response, src='en', dest='pt').text
    
    return JSONResponse(content={"response": portuguese_response})