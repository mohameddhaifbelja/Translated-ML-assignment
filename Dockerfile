FROM python:3.8.16
WORKDIR app

COPY . .
RUN pip install -r requirements.txt
RUN pip install "fastapi[all]"
EXPOSE 5000
CMD ["python", "app.py"]