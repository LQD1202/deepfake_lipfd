# Sử dụng image Python 3.10 để hỗ trợ các thư viện
FROM python:3.10

# Cài đặt các gói hệ thống cần thiết cho librosa, opencv, và matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc với đường dẫn tuyệt đối
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY ./requirements.txt /app/requirements.txt

# Cài đặt pip và các thư viện Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Sao chép mã nguồn vào container
COPY ./app.py /app/app.py

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh chạy ứng dụng với python -m uvicorn để tránh lỗi PATH
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]