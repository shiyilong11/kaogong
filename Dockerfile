FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 避免生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖，包含构建 NumPy/SciPy 所需的库与工具
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       gfortran \
       libopenblas-dev \
       liblapack-dev \
       libatlas-base-dev \
       pkg-config \
       cmake \
       wget \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单并先升级 pip，再安装依赖（以便安装 wheel 或构建包）
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# 复制应用代码
COPY . /app

# 对外暴露端口（默认 Streamlit 端口可改为 8503）
EXPOSE 8503

# 使用非 root 用户（先安装依赖为 root）
RUN useradd -m appuser || true
USER appuser

# 启动命令
CMD ["bash", "-lc", "streamlit run 11.py --server.address 0.0.0.0 --server.port ${PORT:-8503} --server.headless true"]
