FROM ubuntu:16.04

# Copying the requirements file to install them
COPY . /app/mock
RUN cp -r ../lib/ /app/]

# Change working directory
WORKDIR /app/mock

# Installing python
RUN apt-get update -y && \
    apt-get install -y python3 python-pip

# Install the libaries in reqiurements.txt
RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

# Run simulator
CMD [ "main.py" ]