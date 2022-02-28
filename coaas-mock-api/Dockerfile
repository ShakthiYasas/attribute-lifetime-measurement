FROM ubuntu

# Copying the requirements file to install them
COPY . /app/mock

# Installing python
RUN apt-get update -y
RUN apt-get install -y python3 python3-pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip

# Change working directory
WORKDIR /app/mock

# Install the libaries in reqiurements.txt
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3" ]

# Run simulator
CMD [ "main.py" ]