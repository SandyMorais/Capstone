FROM python:3.12

ADD . /app
WORKDIR /app

# install packages by conda
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
