# Gunakan image Python yang stabil
FROM python:3.10-slim

# Set folder kerja di dalam container
WORKDIR /app

# Instal dependencies sistem yang dibutuhkan (untuk library C/C++)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan instal (Tanpa pusing versi numpy)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy semua file project
COPY . .

# Jalankan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]