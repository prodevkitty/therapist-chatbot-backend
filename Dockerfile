FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Copy the application code
COPY ./app /app

# Copy the model files
COPY ./models /app/models

# Install dependencies
RUN pip install -r /app/requirements.txt

# Set the command to run the FastAPI application
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:80", "--workers", "4"]