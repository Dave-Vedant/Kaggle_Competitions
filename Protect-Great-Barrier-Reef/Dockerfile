# set base image (host OS)
FROM python:3.8.10

# set the working directory in the container
WORKDIR /protect_gbr

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local whole directory to the working directory
COPY src/ src/

# command to run on container start
CMD [ "python3", "run.py" ]