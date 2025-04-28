FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --quiet --no-cache-dir -r requirements.txt
EXPOSE 8080
ENV PORT=8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:server"]