# Use a base image with the desired operating system and dependencies
FROM python:3.8-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install the project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install AWS CLI using apt package manager
RUN apt-get update && apt-get install -y groff less
RUN pip install awscli

# Copy the project files into the container
COPY . /app

# Expose the port on which the Flask application will run (if applicable)
EXPOSE 8080

# Set the environment variables
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
ENV MONGO_DB_URL=${MONGO_DB_URL}
ENV BUCKET_NAME=${BUCKET_NAME}

# Define the command to run your Flask application
CMD ["python", "app.py"]
