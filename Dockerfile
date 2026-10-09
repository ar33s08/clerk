FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY clerk/ clerk/
ENV CLERK_DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "clerk.api:app", "--host", "0.0.0.0", "--port", "8000"]
