FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
EXPOSE 80
WORKDIR /app
RUN pip3 install -r requirements.txt
CMD ["uvicorn", "app.Main:app", "--host", "0.0.0.0", "--port", "8000"]