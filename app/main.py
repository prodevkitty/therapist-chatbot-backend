from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import BertTokenizer
from googletrans import Translator
import tensorflow as tf
import asyncio
import httpx

app = FastAPI()

# Load the pre-trained BERT model and tokenizer
model_path = 'app/models/experts-bert-tensorflow2-pubmed-v2'  # Update with the actual path to your model
tokenizer = BertTokenizer.from_pretrained(f"{model_path}/assets")
model = tf.saved_model.load(model_path)

# Initialize the translator
translator = Translator()

async def translate_text(text: str, src_lang: str, dest_lang: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            'https://translate.googleapis.com/translate_a/single',
            params={
                'client': 'gtx',
                'sl': src_lang,
                'tl': dest_lang,
                'dt': 't',
                'q': text,
            },
        )
        translated_text = response.json()[0][0][0]
        return translated_text

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    
    # Detect the language of the input
    detected_lang = translator.detect(user_input).lang
    
    # Supported languages
    supported_languages = ['en', 'es', 'pt', 'pt-BR', 'it', 'ar', 'qu', 'zh', 'ko']
    
    # If the input is in a supported language other than English, translate it to English
    if detected_lang in supported_languages and detected_lang != 'en':
        translated_input = await translate_text(user_input, src_lang=detected_lang, dest_lang='en')
    else:
        translated_input = user_input
    
    # Tokenize the input
    inputs = tokenizer(translated_input, return_tensors='tf')
    
    # Generate response in English
    outputs = model(inputs)
    logits = outputs['logits']
    predicted_class_id = tf.argmax(logits, axis=-1).numpy()[0]
    
    # Map the predicted class ID to a response (this mapping should be defined based on your model's training)
    response_mapping = {
        0: "I'm here to help you.",
        1: "Can you please elaborate on that?",
        # Add more mappings based on your model's output classes
    }
    english_response = response_mapping.get(predicted_class_id, "I'm not sure how to respond to that.")
    
    # If the original input was in a supported language other than English, translate the response back to that language
    if detected_lang in supported_languages and detected_lang != 'en':
        final_response = await translate_text(english_response, src_lang='en', dest_lang=detected_lang)
    else:
        final_response = english_response
    
    return JSONResponse(content={"response": final_response})