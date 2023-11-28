# Start with a base image that includes Python 3.10 and CUDA support
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV TZ=Europe/Amsterdam  \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y build-essential curl git python3.8 python3-pip python-is-python3

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

# install apex?
RUN git clone https://github.com/NVIDIA/apex
RUN poetry run python -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option=--cpp_ext --global-option=--cuda_ext /app/apex/


# # Set the default command for the container to Python shell
CMD ["python"]
