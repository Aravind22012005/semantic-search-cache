FROM python:3.10

WORKDIR /app

COPY requirements.txt .

# install CPU-only torch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]