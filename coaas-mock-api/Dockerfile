FROM python:3.8.2

# Set up and activate virtual environment
ENV VIRTUAL_ENV "/venv"
RUN python -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

# Python commands run inside the virtual environment
RUN python -m pip install \
        numpy \
        scipy \
        matplotlib \
        traceback \ 
        configparser \ 
        flask \ 
        flask_restful \ 
        json \ 
        threading \ 
        pymongo \
        multiprocessing

WORKDIR /usr/src/app
COPY . .
COPY .\lib .
CMD ["main.py"]