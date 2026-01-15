FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 避免生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（如果需要可扩展）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装 Python 依赖
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制应用代码
COPY . /app

# 对外暴露端口（默认 Streamlit 端口可改为 8503）
EXPOSE 8503

# 使用非 root 用户（可选）
RUN useradd -m appuser || true
USER appuser

# 启动命令
CMD ["streamlit", "run", "11.py", "--server.address", "0.0.0.0", "--server.port", "8503", "--server.headless", "true"]
