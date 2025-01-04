FROM python:3.10-slim

RUN apt-get update -qq && apt-get install -qq -y \
    curl \
    libgl1-mesa-glx \
    libqt6widgets6 \
    libqt6gui6 \
    libportaudio2 \
    libsndfile1 \
    git \
    gcc \
    build-essential \
    ffmpeg \
    libxcb-cursor0 \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1

WORKDIR /app

COPY . /app

COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

ENV DISPLAY=:0
ENV QT_DEBUG_PLUGINS=0

ENTRYPOINT ["/app/entrypoint.sh"]