# Start with a base image that includes Python 3.10 and CUDA support
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV TZ=Europe/Amsterdam  \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y build-essential --no-install-recommends
RUN apt-get install -y curl --no-install-recommends
RUN apt-get install -y git --no-install-recommends
RUN apt-get install -y python3.8 --no-install-recommends
RUN apt-get install -y python3.8-dev --no-install-recommends
RUN apt-get install -y python3-pip --no-install-recommends
RUN apt-get install -y python3.8-venv --no-install-recommends
RUN apt-get install -y python3.8-distutils --no-install-recommends
RUN apt-get install -y python-is-python3 --no-install-recommends

# # Use curl to install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 

ENV PATH="${PATH}:/root/.local/bin"

# Copy your entire project folder to the container
WORKDIR /app
COPY . /app/ 

# # Install project dependencies
RUN poetry config virtualenvs.create true
RUN poetry config virtualenvs.in-project true
RUN poetry config installer.max-workers 10
RUN poetry install --no-root # .max-workers 10

# # Set the default command for the container to Python shell
CMD ["python"]