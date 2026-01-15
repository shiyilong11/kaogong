import math
import numpy as np
from typing import Tuple

import altair as alt
import pandas as pd
import streamlit as st


# ===================== 原有核心函数（保留11.py全部逻辑） =====================
def normal_cdf(z: float) -> float:
    """Standard normal CDF using erf to avoid extra dependencies."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def normal_ppf(p: float) -> float:
    """Approximate inverse CDF for standard normal (Acklam's approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def normal_pdf(z: float) -> float:
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)


def truncated_normal_mean(mu: float, sigma: float, a: float) -> float:
    """Mean of truncated normal distribution, truncated at a (lower bound)."""
    if sigma <= 0:
        return max(mu, a)
    z_a = (a - mu) / sigma
    phi_a = normal_pdf(z_a)
    Phi_a = normal_cdf(z_a)
    if Phi_a >= 1:
        return mu
    return mu + sigma * (phi_a / (1 - Phi_a))


def truncated_normal_var(mu: float, sigma: float, a: float) -> float:
    """Variance of truncated normal distribution, truncated at a (lower bound)."""
    if sigma <= 0:
        return 0.0
    z_a = (a - mu) / sigma
    phi_a = normal_pdf(z_a)
    Phi_a = normal_cdf(z_a)
    if Phi_a >= 1:
        return sigma ** 2
    ratio = phi_a / (1 - Phi_a)
    var = sigma ** 2 * (1 - ratio * (z_a + ratio))
    return max(0.0, var)  # Ensure non-negative


def estimate_overall_from_data(
    applicants: float,
    interview_count: float,
    min_written: float,
    estimate_mode: str = "已知最高分",
    known_max: float = None,
    historical_min: float = None,
    historical_max: float = None,
    ratio: float = None,
) -> Tuple[float, float]:
    """Estimate overall written mean and sd."""
    if applicants <= 0 or interview_count <= 0 or min_written <= 0:
        return 0.55 * 200, 0.15 * 200  # Fallback

    p = interview_count / applicants
    if p >= 1:
        return min_written, 1.0
    if p <= 0:
        return min_written, 1.0

    z = normal_ppf(1 - p)

    # Base (fallback) estimate
    sigma = min_written / (z + 1.5)
    mu = min_written - sigma * z

    # Adjust based on mode
    if estimate_mode == "已知最高分" and known_max is not None and known_max > min_written:
        try:
            z_min = z  # corresponds to min_written quantile (1 - p)
            z_max = normal_ppf(1.0 - 1.0 / (applicants + 1.0))
            if abs(z_max - z_min) > 1e-6:
                sigma = (known_max - min_written) / (z_max - z_min)
                mu = min_written - z_min * sigma
            else:
                z_max = max(z_min + 2.5, 3.0)
                sigma = (known_max - min_written) / (z_max - z_min)
                mu = min_written - z_min * sigma
        except Exception:
            pass
    elif estimate_mode == "历年进面分差值" and historical_min is not None and historical_max is not None:
        diff = historical_max - historical_min
        if diff > 0:
            try:
                q_bottom = 1.0 - (interview_count / applicants)
                z_bottom = normal_ppf(q_bottom)
                z_top = normal_ppf(1.0 - 1.0 / (applicants + 1.0))
                if abs(z_top - z_bottom) > 1e-6:
                    sigma = diff / (z_top - z_bottom)
                    mu = min_written - z * sigma
                else:
                    sigma = diff / 3.0
                    mu = min_written - z * sigma
            except Exception:
                sigma = diff / 3.0
                mu = min_written - z * sigma
    elif estimate_mode == "比例估算最高分" and ratio is not None:
        estimated_max = min_written * ratio
        try:
            z_min = z
            z_max = normal_ppf(1.0 - 1.0 / (applicants + 1.0))
            if abs(z_max - z_min) > 1e-6:
                sigma = (estimated_max - min_written) / (z_max - z_min)
                mu = min_written - z_min * sigma
        except Exception:
            pass

    if mu < 0:
        mu = min_written * 0.5
        sigma = (min_written - mu) / z if z > 0 else sigma
    return mu, sigma


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def calc_combined_score(
    written_score: float,
    interview_score: float,
    written_full: float,
    interview_full: float,
    written_weight: float,
) -> float:
    """Return combined score on 0-100 scale."""
    w_ratio = written_weight
    i_ratio = 1 - written_weight
    written_norm = (written_score / written_full) * 100 if written_full else 0
    interview_norm = (interview_score / interview_full) * 100 if interview_full else 0
    return w_ratio * written_norm + i_ratio * interview_norm


def estimate_distribution(
    min_written: float,
    written_full: float,
    interview_mean: float,
    interview_sd: float,
    written_sd: float,
    written_mean_hint: float,
    written_weight: float,
    interview_full: float,
    overall_written_mean: float,
    overall_written_sd: float,
    use_truncated_pool: bool = True,
    written_pool_type: str = "截断正态",
    skew_k: float = 0.4,
    # 新增：面试成绩分布类型参数
    interview_dist_type: str = "正态分布",
) -> Tuple[float, float]:
    """Approximate combined score mean and sd for the interview pool"""
    # 笔试分布处理（原有逻辑）
    if use_truncated_pool:
        mean_written = truncated_normal_mean(overall_written_mean, overall_written_sd, min_written)
        var_written = truncated_normal_var(overall_written_mean, overall_written_sd, min_written)
        sd_written = math.sqrt(max(1e-6, var_written))
        if written_pool_type == "截断偏态（右偏）":
            mean_written = mean_written + skew_k * sd_written
            sd_written = sd_written * max(0.3, 1.0 - 0.4 * skew_k)
    else:
        mean_written = overall_written_mean
        sd_written = max(1e-6, overall_written_sd)

    if math.isnan(mean_written) or math.isnan(sd_written):
        mean_written = max(min_written + 0.15 * written_full, written_mean_hint)
        sd_written = max(5.0, written_sd)

    # 新增：面试成绩分布类型处理
    if interview_dist_type == "均匀分布":
        # 均匀分布方差 = (max-min)²/12，这里简化为等价标准差
        interview_sd = (interview_full - 0) / math.sqrt(12)
        interview_mean = interview_full / 2  # 均匀分布均值默认中点
    elif interview_dist_type == "偏态分布（右偏）":
        interview_mean = interview_mean + 0.2 * interview_sd  # 右偏调整均值
        interview_sd = interview_sd * 0.8  # 右偏调整标准差
    elif interview_dist_type == "偏态分布（左偏）":
        interview_mean = interview_mean - 0.2 * interview_sd  # 左偏调整均值
        interview_sd = interview_sd * 0.8  # 左偏调整标准差
    # 正态分布保持原有逻辑
    interview_sd = max(5.0, interview_sd)

    # 综合分计算（原有逻辑）
    mean_combined = calc_combined_score(
        written_score=mean_written,
        interview_score=interview_mean,
        written_full=written_full,
        interview_full=interview_full,
        written_weight=written_weight,
    )

    w_ratio = written_weight
    i_ratio = 1 - written_weight
    var_combined = (
        (w_ratio ** 2) * ((sd_written / written_full) * 100) ** 2
        + (i_ratio ** 2) * ((interview_sd / interview_full) * 100) ** 2
    )
    sd_combined = math.sqrt(var_combined)
    return mean_combined, sd_combined


def estimate_interview_score(
    interview_mean: float, interview_sd: float, percentile: float, interview_full: float
) -> float:
    percentile = clamp(percentile, 0.0, 100.0)
    p = 1 - percentile / 100
    z = normal_ppf(p)
    estimated = interview_mean + z * interview_sd
    return clamp(estimated, 0, interview_full)


def compute_probability(
    entered_interview: bool,
    written_score: float,
    interview_score_est: float,
    written_full: float,
    interview_full: float,
    written_weight: float,
    min_written: float,
    interview_mean: float,
    interview_sd: float,
    written_sd: float,
    written_mean_hint: float,
    admit_count: float,
    interview_count: float,
    overall_written_mean: float,
    overall_written_sd: float,
    use_truncated_pool: bool = True,
    written_pool_type: str = "截断正态",
    skew_k: float = 0.4,
    use_mc: bool = False,
    mc_samples: int = 0,
    applicants: int = 0,
    # 新增参数
    compute_mode: str = "解析计算",  # 计算方式：解析计算/蒙特卡洛模拟
    interview_dist_type: str = "正态分布",  # 面试成绩分布类型
) -> Tuple[float, float, float, float, float]:
    """Return probability, user_combined, cutoff, sd_combined, mean_combined."""
    # 蒙特卡洛模拟路径（仅高级模式可选）
    if use_mc and compute_mode == "蒙特卡洛模拟":
        N = int(applicants)
        M = int(interview_count)
        K = int(admit_count)
        mu = overall_written_mean
        sigma = overall_written_sd

        success = 0
        cutoffs = []
        combined_sds = []
        mean_combined_list = []

        for _ in range(int(mc_samples)):
            others = np.random.normal(loc=mu, scale=sigma, size=N - 1)
            all_written = np.concatenate([others, np.array([written_score])])
            top_idx = np.argsort(all_written)[-M:]
            user_index = len(all_written) - 1
            user_in_top = user_index in top_idx
            top_written = all_written[top_idx]
            
            # 复试分布计算模式：应用笔试分布类型调整
            if written_pool_type == "截断偏态（右偏）":
                top_written = top_written + skew_k * np.std(top_written)
            elif written_pool_type == "偏态（左偏）":
                top_written = top_written - skew_k * np.std(top_written)

            # 面试成绩分布类型采样调整
            if interview_dist_type == "均匀分布":
                top_interview = np.random.uniform(0, interview_full, size=M)
            elif interview_dist_type == "偏态分布（右偏）":
                # 对数正态分布模拟右偏
                top_interview = np.random.lognormal(
                    mean=np.log(interview_mean), 
                    sigma=interview_sd/interview_mean, 
                    size=M
                )
            elif interview_dist_type == "偏态分布（左偏）":
                # 反转对数正态分布模拟左偏
                top_interview = interview_full - np.random.lognormal(
                    mean=np.log(interview_mean), 
                    sigma=interview_sd/interview_mean, 
                    size=M
                )
            else:  # 正态分布
                top_interview = np.random.normal(loc=interview_mean, scale=interview_sd, size=M)

            # 修正面试分数范围
            top_interview = np.clip(top_interview, 0, interview_full)

            if user_in_top:
                pos = list(top_idx).index(user_index)
                top_interview[pos] = interview_score_est

            # 计算综合分
            written_norm = (top_written / written_full) * 100
            interview_norm = (top_interview / interview_full) * 100
            combined = written_weight * written_norm + (1 - written_weight) * interview_norm

            cutoff_sim = np.sort(combined)[-K] if K <= M else np.min(combined)
            cutoffs.append(float(cutoff_sim))
            mean_combined_list.append(float(np.mean(combined)))
            combined_sds.append(float(np.std(combined, ddof=1)))

            if user_in_top:
                user_combined = calc_combined_score(
                    written_score=written_score,
                    interview_score=interview_score_est,
                    written_full=written_full,
                    interview_full=interview_full,
                    written_weight=written_weight,
                )
                if user_combined >= cutoff_sim:
                    success += 1

        prob = success / int(mc_samples) if mc_samples > 0 else 0.0
        user_combined = calc_combined_score(
            written_score=written_score,
            interview_score=interview_score_est,
            written_full=written_full,
            interview_full=interview_full,
            written_weight=written_weight,
        )
        cutoff = float(np.mean(cutoffs)) if len(cutoffs) else 0.0
        sd_combined = float(np.mean(combined_sds)) if len(combined_sds) else 0.0
        mean_combined = float(np.mean(mean_combined_list)) if len(mean_combined_list) else 0.0
        return prob, user_combined, cutoff, sd_combined, mean_combined

    # 解析计算路径（默认）
    mean_combined, sd_combined = estimate_distribution(
        min_written=min_written,
        written_full=written_full,
        interview_mean=interview_mean,
        interview_sd=interview_sd,
        written_sd=written_sd,
        written_mean_hint=written_mean_hint,
        written_weight=written_weight,
        interview_full=interview_full,
        overall_written_mean=overall_written_mean,
        overall_written_sd=overall_written_sd,
        use_truncated_pool=use_truncated_pool,
        written_pool_type=written_pool_type,
        skew_k=skew_k,
        interview_dist_type=interview_dist_type,  # 传递面试分布类型
    )

    user_combined = calc_combined_score(
        written_score=written_score,
        interview_score=interview_score_est,
        written_full=written_full,
        interview_full=interview_full,
        written_weight=written_weight,
    )

    if interview_count <= 0 or admit_count <= 0:
        return 0.0, user_combined, 0.0, sd_combined, mean_combined

    admit_ratio = clamp(admit_count / interview_count, 0.0, 1.0)
    if admit_ratio >= 1:
        return 1.0, user_combined, 0.0, sd_combined, mean_combined

    cutoff_quantile = 1 - admit_ratio
    z_line = normal_ppf(cutoff_quantile)
    cutoff_score = mean_combined + z_line * sd_combined

    if sd_combined <= 1e-6:
        probability = 1.0 if user_combined >= cutoff_score else 0.0
    else:
        probability = 1 - normal_cdf((cutoff_score - user_combined) / sd_combined)

    probability = clamp(probability, 0.0, 1.0)
    return probability, user_combined, cutoff_score, sd_combined, mean_combined


def show_fireworks():
    """Render a simple fireworks-like animation."""
    fireworks_html = """
    <div class="fireworks">
      <div class="after"></div>
      <div class="before"></div>
    </div>
    <style>
      .fireworks, .fireworks::before, .fireworks::after {
        position: fixed;
        top: 50%;
        left: 50%;
        width: 8px;
        height: 8px;
        background: transparent;
        pointer-events: none;
        transform: translate(-50%, -50%);
        box-shadow: -60px -60px #ff4d4f, 0 -70px #ffc53d, 60px -60px #40a9ff,
                    -70px 0 #73d13d, 70px 0 #9254de, -60px 60px #ff85c0,
                    0 70px #5cdbd3, 60px 60px #ffec3d;
        animation: pop 900ms ease-out forwards;
        opacity: 0.9;
      }
      .fireworks::before, .fireworks::after {
        content: "";
        display: block;
      }
      .fireworks::before {
        box-shadow: -50px -80px #ff4d4f, 50px -80px #ffc53d,
                    -80px 50px #40a9ff, 80px 50px #73d13d,
                    -80px -20px #9254de, 80px -20px #ff85c0,
                    -20px 80px #5cdbd3, 20px 80px #ffec3d;
        animation: pop 1000ms ease-out forwards;
      }
      .fireworks::after {
        box-shadow: -30px -90px #73d13d, 30px -90px #9254de,
                    -90px 30px #ff4d4f, 90px 30px #5cdbd3,
                    -90px -10px #ffc53d, 90px -10px #40a9ff,
                    -10px 90px #ff85c0, 10px 90px #ffec3d;
        animation: pop 1100ms ease-out forwards;
      }
      @keyframes pop {
        0% { transform: translate(-50%, -50%) scale(0.2); opacity: 1; }
        80% { opacity: 1; }
        100% { transform: translate(-50%, -50%) scale(1.1); opacity: 0; }
      }
    </style>
    """
    st.components.v1.html(fireworks_html, height=0, width=0)


def render_feedback(probability: float, mode: str, entered: bool):
    if not entered:
        st.write("❌ 未进入面试，无法评估录取概率。")
        return

    if mode.startswith("客观"):
        tiers = [
            (0.8, "🎉 很有希望！上岸在望！"),
            (0.6, "不错的机会，保持信心"),
            (0.4, "潜力很大，继续努力"),
            (0.2, "加油，还有戏"),
            (0.0, "客观评估：进面即有机会 ✨"),
        ]
    else:
        tiers = [
            (0.9, "🎉 祝贺！大概率上岸，放烟花庆祝！"),
            (0.8, "很稳，保持节奏即可"),
            (0.6, "有戏，认真准备面试细节"),
            (0.4, "五五开，补齐短板提升稳定性"),
            (0.2, "需要加油，针对弱项冲刺"),
            (0.0, "风险较高，尽量多做备选"),
        ]

    for threshold, text in tiers:
        if probability >= threshold:
            st.write(text)
            break

    if probability >= 0.9:
        show_fireworks()
        st.balloons()


def render_distribution_chart(mean_c: float, sd_c: float, user_c: float, cutoff_c: float):
    if sd_c <= 1e-6:
        st.info("分布标准差过小，无法绘制曲线。")
        return

    x_min = mean_c - 3.5 * sd_c
    x_max = mean_c + 3.5 * sd_c
    xs = np.linspace(x_min, x_max, 201)  # 优化：替换列表推导式
    ys = [normal_pdf((x - mean_c) / sd_c) / sd_c for x in xs]
    df = pd.DataFrame({"score": xs, "density": ys})

    base = alt.Chart(df).mark_line(color="#1890ff", strokeWidth=2).encode(
        x=alt.X("score", title="综合分"),
        y=alt.Y("density", title="密度", axis=alt.Axis(labels=False)),
    )

    user_rule = (
        alt.Chart(pd.DataFrame({"score": [user_c], "label": ["你"]}))
        .mark_rule(color="#fa541c", strokeWidth=2, strokeDash=[5, 3])
        .encode(x="score")
    )

    cutoff_rule = (
        alt.Chart(pd.DataFrame({"score": [cutoff_c], "label": ["录取线"]}))
        .mark_rule(color="#52c41a", strokeWidth=2)
        .encode(x="score")
    )

    text_layer = (
        alt.Chart(
            pd.DataFrame(
                {
                    "score": [user_c, cutoff_c],
                    "density": [max(ys) * 0.9, max(ys) * 0.8],
                    "label": ["你的分", "预计录取线"],
                }
            )
        )
        .mark_text(dy=-6, fontSize=11)
        .encode(x="score", y="density", text="label", color=alt.value("#595959"))
    )

    chart = base + user_rule + cutoff_rule + text_layer
    st.altair_chart(chart, use_container_width=True)


# ===================== 新增：友情评估主界面（初级/高级模式） =====================
def main():
    st.set_page_config(page_title="上岸概率估算器", page_icon="🎓", layout="wide")
    st.title("🎓 上岸概率估算器 - 友情评估版")

    # 1. 模式选择：初级/高级
    mode_level = st.sidebar.radio(
        "选择评估模式",
        ["初级模式", "高级模式"],
        help="初级模式：仅保留核心笔试分布估算；高级模式：全功能扩展"
    )

    # 2. 通用基础参数（所有模式共享）
    st.sidebar.header("📝 基础参数")
    applicants = st.sidebar.number_input("报名总人数", min_value=1, value=1000)
    interview_count = st.sidebar.number_input("进面人数", min_value=1, value=200)
    admit_count = st.sidebar.number_input("录取人数", min_value=1, value=50)
    min_written = st.sidebar.number_input("进面最低笔试分", min_value=0.0, value=120.0)
    written_full = st.sidebar.number_input("笔试满分", min_value=1.0, value=200.0)
    interview_full = st.sidebar.number_input("面试满分", min_value=1.0, value=100.0)
    written_weight_pct = st.sidebar.slider("笔试权重(%)", min_value=0, max_value=100, value=60)
    written_weight = written_weight_pct / 100

    # 3. 初级模式：仅保留笔试分布估算
    if mode_level == "初级模式":
        st.header("🔰 初级模式 - 笔试分布估算")
        
        # 仅显示笔试分布估算按钮
        if st.button("📊 开始笔试分布估算"):
            # 估算整体笔试均值和标准差
            overall_written_mean, overall_written_sd = estimate_overall_from_data(
                applicants=applicants,
                interview_count=interview_count,
                min_written=min_written,
                estimate_mode="已知最高分",
                known_max=written_full
            )
            st.subheader("📈 笔试分布估算结果")
            st.write(f"整体笔试平均分：{overall_written_mean:.2f}")
            st.write(f"整体笔试标准差：{overall_written_sd:.2f}")
            st.write(f"进面笔试分下限：{min_written:.2f}")

            # 绘制笔试分布曲线
            x_min = overall_written_mean - 3.5 * overall_written_sd
            x_max = overall_written_mean + 3.5 * overall_written_sd
            xs = np.linspace(x_min, x_max, 201)
            ys = [normal_pdf((x - overall_written_mean) / overall_written_sd) / overall_written_sd for x in xs]
            df = pd.DataFrame({"笔试分数": xs, "密度": ys})
            
            # 标记进面线
            cutoff_line = alt.Chart(pd.DataFrame({"x": [min_written]})).mark_rule(color="#ff4d4f", strokeWidth=2).encode(x="x")
            base = alt.Chart(df).mark_line(color="#1890ff").encode(
                x=alt.X("笔试分数", title="笔试分数"),
                y=alt.Y("密度", title="密度")
            )
            st.altair_chart(base + cutoff_line, use_container_width=True)

    # 4. 高级模式：全功能扩展
    elif mode_level == "高级模式":
        st.header("⚡ 高级模式 - 全功能评估")
        
        # 4.1 计算方式选择
        compute_mode = st.sidebar.selectbox(
            "计算方式",
            ["解析计算", "蒙特卡洛模拟"],
            help="解析计算：快速估算；蒙特卡洛模拟：更精准但耗时"
        )
        mc_samples = st.sidebar.number_input(
            "蒙特卡洛模拟次数",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=1000,
            disabled=(compute_mode != "蒙特卡洛模拟")
        )

        # 4.2 复试分布计算模式
        st.sidebar.header("🔍 复试分布参数")
        distribution_mode = st.sidebar.radio(
            "复试分布计算模式",
            ["基础模式", "进阶模式（含偏态调整）"],
            help="进阶模式支持笔试分布类型调整"
        )
        
        # 4.3 笔试分布类型（进阶模式显示）
        written_pool_type = st.sidebar.selectbox(
            "笔试分布类型",
            ["截断正态", "截断偏态（右偏）", "偏态（左偏）"],
            disabled=(distribution_mode != "进阶模式（含偏态调整）")
        )
        skew_k = st.sidebar.slider(
            "偏态强度",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            disabled=(distribution_mode != "进阶模式（含偏态调整）")
        )

        # 4.4 面试成绩分布类型
        interview_dist_type = st.sidebar.selectbox(
            "面试成绩分布类型",
            ["正态分布", "均匀分布", "偏态分布（右偏）", "偏态分布（左偏）"],
            help="选择面试成绩的分布特征"
        )
        interview_mean = st.sidebar.number_input(
            "面试平均分",
            min_value=0.0,
            max_value=interview_full,
            value=80.0
        )
        interview_sd = st.sidebar.number_input(
            "面试标准差",
            min_value=0.1,
            max_value=20.0,
            value=5.0
        )

        # 4.5 个人成绩输入
        st.sidebar.header("👤 个人成绩")
        written_score = st.sidebar.number_input("你的笔试分数", min_value=0.0, max_value=written_full, value=130.0)
        interview_percentile = st.sidebar.slider(
            "你的面试成绩百分位（0=最好，100=最差）",
            min_value=0.0,
            max_value=100.0,
            value=20.0
        )
        interview_score_est = estimate_interview_score(
            interview_mean=interview_mean,
            interview_sd=interview_sd,
            percentile=interview_percentile,
            interview_full=interview_full
        )

        # 4.6 核心计算按钮
        if st.button("🚀 开始综合录取概率评估"):
            # 估算整体笔试分布
            overall_written_mean, overall_written_sd = estimate_overall_from_data(
                applicants=applicants,
                interview_count=interview_count,
                min_written=min_written,
                estimate_mode="已知最高分",
                known_max=written_full
            )

            # 计算录取概率
            probability, user_combined, cutoff_score, sd_combined, mean_combined = compute_probability(
                entered_interview=True,
                written_score=written_score,
                interview_score_est=interview_score_est,
                written_full=written_full,
                interview_full=interview_full,
                written_weight=written_weight,
                min_written=min_written,
                interview_mean=interview_mean,
                interview_sd=interview_sd,
                written_sd=overall_written_sd,
                written_mean_hint=overall_written_mean,
                admit_count=admit_count,
                interview_count=interview_count,
                overall_written_mean=overall_written_mean,
                overall_written_sd=overall_written_sd,
                use_truncated_pool=(distribution_mode == "进阶模式（含偏态调整）"),
                written_pool_type=written_pool_type,
                skew_k=skew_k,
                use_mc=(compute_mode == "蒙特卡洛模拟"),
                mc_samples=mc_samples,
                applicants=applicants,
                compute_mode=compute_mode,
                interview_dist_type=interview_dist_type
            )

            # 展示结果
            st.subheader("🎯 评估结果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("你的综合分", f"{user_combined:.2f}")
            with col2:
                st.metric("预计录取线", f"{cutoff_score:.2f}")
            with col3:
                st.metric("录取概率", f"{probability:.2%}")

            # 反馈提示
            render_feedback(probability, "高级模式", entered=True)

            # 绘制综合分分布
            st.subheader("📊 综合分分布")
            render_distribution_chart(mean_combined, sd_combined, user_combined, cutoff_score)

            # 展示详细参数
            with st.expander("📋 详细参数明细"):
                st.write(f"计算方式：{compute_mode}")
                st.write(f"复试分布模式：{distribution_mode}")
                st.write(f"笔试分布类型：{written_pool_type}")
                st.write(f"面试分布类型：{interview_dist_type}")
                st.write(f"整体笔试平均分：{overall_written_mean:.2f}")
                st.write(f"整体笔试标准差：{overall_written_sd:.2f}")
                st.write(f"面试平均分：{interview_mean:.2f}")
                st.write(f"面试标准差：{interview_sd:.2f}")

# ====================== Streamlit 主程序入口（粘贴到文件末尾） ======================
def main():
    # 页面基础配置
    st.set_page_config(
        page_title="公考录取概率估算",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 页面标题
    st.title("📊 面试/录取概率估算工具")
    st.divider()

    # 侧边栏：参数输入区
    with st.sidebar:
        st.header("⚙️ 基础参数设置")
        applicants = st.number_input("总报名人数", min_value=1, value=1000, step=10)
        interview_count = st.number_input("进面人数", min_value=1, value=200, step=10)
        min_written = st.number_input("进面最低笔试分", min_value=0.0, value=120.0, step=1.0)
        written_full = st.number_input("笔试满分", min_value=1.0, value=200.0, step=1.0)
        interview_full = st.number_input("面试满分", min_value=1.0, value=100.0, step=1.0)
        written_weight = st.slider("笔试权重（0-1）", 0.0, 1.0, 0.6, step=0.05)
        admit_count = st.number_input("最终录取人数", min_value=1, value=50, step=5)
        interview_mean = st.number_input("面试平均分（预估）", min_value=0.0, value=85.0, step=0.5)
        interview_sd = st.number_input("面试分标准差（预估）", min_value=0.1, value=5.0, step=0.1)
        user_written = st.number_input("你的笔试分数", min_value=0.0, value=130.0, step=0.5)
        user_interview_est = st.number_input("你的面试预估分", min_value=0.0, value=88.0, step=0.5)

        st.divider()
        st.header("🔧 高级设置")
        estimate_mode = st.selectbox("笔试分估算模式", ["已知最高分", "历年进面分差值", "比例估算最高分"])
        known_max = st.number_input("已知笔试最高分", min_value=min_written, value=180.0, step=1.0)
        historical_min = st.number_input("历年进面最低分（仅差值模式）", min_value=0.0, value=110.0, step=1.0)
        historical_max = st.number_input("历年进面最高分（仅差值模式）", min_value=historical_min, value=170.0, step=1.0)
        ratio = st.number_input("最高分/进面最低分（仅比例模式）", min_value=1.0, value=1.5, step=0.1)
        use_truncated_pool = st.checkbox("使用截断笔试分布（更精准）", value=True)
        written_pool_type = st.selectbox("笔试分布类型", ["截断正态", "截断偏态（右偏）"])
        skew_k = st.slider("偏态系数（仅偏态模式）", 0.0, 1.0, 0.4, step=0.1)
        use_mc = st.checkbox("启用蒙特卡洛模拟（更精准但慢）", value=False)
        mc_samples = st.number_input("模拟次数（仅模拟模式）", min_value=100, max_value=10000, value=1000, step=100)

    # 1. 估算整体笔试均值和标准差
    overall_written_mean, overall_written_sd = estimate_overall_from_data(
        applicants=applicants,
        interview_count=interview_count,
        min_written=min_written,
        estimate_mode=estimate_mode,
        known_max=known_max,
        historical_min=historical_min,
        historical_max=historical_max,
        ratio=ratio,
    )

    # 2. 判断是否进面
    entered_interview = user_written >= min_written

    # 3. 计算录取概率
    prob, user_combined, cutoff, sd_combined, mean_combined = compute_probability(
        entered_interview=entered_interview,
        written_score=user_written,
        interview_score_est=user_interview_est,
        written_full=written_full,
        interview_full=interview_full,
        written_weight=written_weight,
        min_written=min_written,
        interview_mean=interview_mean,
        interview_sd=interview_sd,
        written_sd=overall_written_sd,
        written_mean_hint=overall_written_mean,
        admit_count=admit_count,
        interview_count=interview_count,
        overall_written_mean=overall_written_mean,
        overall_written_sd=overall_written_sd,
        use_truncated_pool=use_truncated_pool,
        written_pool_type=written_pool_type,
        skew_k=skew_k,
        use_mc=use_mc,
        mc_samples=mc_samples,
        applicants=applicants,
    )

    # 4. 渲染结果区域
    st.subheader("📈 估算结果")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("是否进面", "✅ 是" if entered_interview else "❌ 否")
    with col2:
        st.metric("你的综合分", f"{user_combined:.2f}")
    with col3:
        st.metric("预估录取线", f"{cutoff:.2f}")
    with col4:
        st.metric("录取概率", f"{prob:.2%}")

    # 5. 渲染反馈提示
    render_feedback(probability=prob, mode="主观", entered=entered_interview)

    # 6. 绘制综合分分布图表
    st.subheader("📊 综合分分布曲线")
    render_distribution_chart(mean_c=mean_combined, sd_c=sd_combined, user_c=user_combined, cutoff_c=cutoff)

    # 7. 显示高级信息（折叠面板）
    with st.expander("🔍 详细参数与计算过程（高级）", expanded=False):
        st.write("### 整体笔试分数估算")
        st.write(f"- 整体笔试均值：{overall_written_mean:.2f}")
        st.write(f"- 整体笔试标准差：{overall_written_sd:.2f}")
        st.write("### 综合分分布参数")
        st.write(f"- 进面人群综合分均值：{mean_combined:.2f}")
        st.write(f"- 进面人群综合分标准差：{sd_combined:.2f}")

# 启动主程序
if __name__ == "__main__":
    main()
