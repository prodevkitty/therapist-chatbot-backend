from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator

app = FastAPI()

# Load the pre-trained model and tokenizer
model_name = 'path_to_downloaded_model'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the translator
translator = Translator()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    
    # Translate Portuguese input to English
    translated_input = translator.translate(user_input, src='pt', dest='en').text
    
    # Generate response in English
    inputs = tokenizer.encode(translated_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1)
    english_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Translate response back to Portuguese
    portuguese_response = translator.translate(english_response, src='en', dest='pt').text
    
    return JSONResponse(content={"response": portuguese_response})