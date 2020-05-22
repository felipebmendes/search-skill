FROM python:3.8

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN pip install -r requirements.txt

ADD . /app

RUN wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distiluse-base-multilingual-cased.zip -P /tmp/
RUN unzip /tmp/distiluse-base-multilingual-cased.zip -d /app/model

RUN rm -rf tmp

EXPOSE 5000

CMD gunicorn -c /app/gunicorn.conf.py main:application
