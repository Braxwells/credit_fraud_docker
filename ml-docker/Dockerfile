# Use base image with mostly all dependencies installed
FROM jupyter/scipy-notebook

# Install another dependency using pip
RUN pip install joblib
RUN pip install tensorflow

# Create folder to save models
RUN mkdir models


# Copy data into image
COPY data/creditcard_2023.csv ./data/creditcard_2023.csv

#Copy model into image
COPY train.py ./train.py
COPY inference.py ./inference.py

# Run training
RUN ["python","train.py"]