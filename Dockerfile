FROM ubuntu:16.04

MAINTAINER Nishant Dayal "dayal.nishant1997@gmail.com"

RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]