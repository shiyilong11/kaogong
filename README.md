# 上岸概率估算器

一个基于 Streamlit 的交互式小工具，用于估算进入复试后的录取概率（解析近似与蒙特卡洛两种计算方式），并提供可视化分布与简单动画提示。

## 要求
- Python 3.8+（推荐 3.10/3.11）
- 推荐在虚拟环境中运行
- 已列在 `requirements.txt`（建议直接使用该文件安装固定版本）

## 安装（Windows PowerShell）
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 运行
```
streamlit run 11.py --server.address localhost --server.port 8503
```
或在 Linux/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run 11.py
```

- 可通过 `--server.port` 更改端口。

## 主要功能说明
- 评估模式：友情评估（带趣味提示与动画）/ 客观评估（简化视图）
- 计算方式：解析近似（解析/数值卷积） / 蒙特卡洛模拟（更精确，耗时更长）
- 当选择“使用总体样本并进行数值卷积”时，应用将用经验密度（KDE 近似）并对加权分数进行数值卷积以生成非正态的综合分布。

## 可选：Docker 运行
```
docker build -t kaogong-app .
docker run -p 8501:8501 kaogong-app
```

## 常见问题
- 如果脚本无法启动，请先运行语法检查：
```
python -m py_compile 11.py
```
- 如果看到动画或图表异常，尝试在浏览器打开 Streamlit 控制台查看错误日志。

## 发布与分发
- 推荐将项目打包为 ZIP 或推送到 Git 仓库，并附上 `requirements.txt` 与本 `README.md`。

---

如果你希望我同时生成 `.gitignore` 与 `Dockerfile`、并替你运行一次本地语法检查与试跑（需要允许我运行终端命令），我可以继续。