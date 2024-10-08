FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Set working directory
WORKDIR /app

# Copy pre-downloaded packages
COPY ./therapist_chatbot_env/Lib/site-packages /packages

# Copy only the requirements file first to leverage Docker cache
COPY ./app/requirements.txt /app/requirements.txt

# Install dependencies from local directory
RUN pip install --no-index --find-links=/packages -r /app/requirements.txt

# Copy the application code
COPY ./app /app

# Copy the model files
COPY ./app/models/experts-bert-tensorflow2-pubmed-v2/assets/vocab.txt /app/models/assets/vocab.txt
COPY ./app/models/experts-bert-tensorflow2-pubmed-v2/variables/variables.data-00000-of-00001 /app/models/variables/variables.data-00000-of-00001
COPY ./app/models/experts-bert-tensorflow2-pubmed-v2/variables/variables.index /app/models/variables/variables.index
COPY ./app/models/experts-bert-tensorflow2-pubmed-v2/saved_model.pb /app/models/saved_model.pb

# Set the command to run the FastAPI application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:80", "--workers", "4"]