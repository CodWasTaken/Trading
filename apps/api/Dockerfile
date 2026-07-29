FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY apps/api ./apps/api
COPY config ./config
RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "trading_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
