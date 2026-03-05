FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY BotTrainer/requirements.txt BotTrainer/requirements.txt
RUN pip install --no-cache-dir -r BotTrainer/requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "run.py", "--server.port=8501", "--server.address=0.0.0.0"]
