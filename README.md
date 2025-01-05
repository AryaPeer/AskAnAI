# AskAnAI

## Introduction
AskAnAI is a tool that records, transcribes, and queries an AI. This project is designed to leverage AI capabilities to process and interact with audio input.

## Prerequisites
- Docker
- NVIDIA or AMD GPU (optional for enhanced performance but can compile using cpu installation just fine)

### GPU Support
If you have an AMD or NVIDIA GPU, you can optimize the installation of PyTorch to leverage your GPU. Follow the instructions provided on the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) to modify the installation command accordingly under Dockerfile.

## Project Structure
All code files are located under the `main` folder.

## Building the Docker Image
To get started, first build the Docker image. Open your terminal and navigate to the project directory, then run the following command:

```sh
docker build -t askanai .
```

## Enable X11 access
```sh
xhost +local: 
```

## Running the Docker Container
Once the Docker image is built, you can run the container with the following command (it takes a while to load though):

```sh
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/snd askanai
```