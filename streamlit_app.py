import math
import numpy as np
from typing import Tuple

import altair as alt
import pandas as pd
import streamlit as st


# Simple math helpers ------------------------------------------------------
def normal_cdf(z: float) -> float:
    """Standard normal CDF using erf to avoid extra dependencies."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def normal_ppf(p: float) -> float:
    """Approximate inverse CDF for standard normal (Acklam's approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    # Coefficients for central region
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
    # Coefficients for tails
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
        # Use order-statistic based z for the maximum among all applicants
        # Expected max quantile ~ 1 - 1/(N+1)
        try:
            z_min = z  # corresponds to min_written quantile (1 - p)
            z_max = normal_ppf(1.0 - 1.0 / (applicants + 1.0))
            if abs(z_max - z_min) > 1e-6:
                sigma = (known_max - min_written) / (z_max - z_min)
                mu = min_written - z_min * sigma
            else:
                # degenerate: fall back to modest z_max
                z_max = max(z_min + 2.5, 3.0)
                sigma = (known_max - min_written) / (z_max - z_min)
                mu = min_written - z_min * sigma
        except Exception:
            # keep base estimate on error
            pass
    elif estimate_mode == "历年进面分差值" and historical_min is not None and historical_max is not None:
        diff = historical_max - historical_min
        if diff > 0:
            # Derive sigma from the z-values of the bottom and top of the interview pool.
            # bottom quantile (overall) corresponds to q_bottom = 1 - (M / N)
            # top quantile approximate as q_top = 1 - 1/(N+1) (expected global max quantile);
            # using these we solve sigma = diff / (z_top - z_bottom)
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
) -> Tuple[float, float]:
    """Approximate combined score mean and sd for the interview pool using truncated normal for written scores."""
    # Use truncated normal for written scores in interview pool
    if use_truncated_pool:
        # Use truncated normal for written scores in interview pool
        mean_written = truncated_normal_mean(overall_written_mean, overall_written_sd, min_written)
        var_written = truncated_normal_var(overall_written_mean, overall_written_sd, min_written)
        sd_written = math.sqrt(max(1e-6, var_written))  # Ensure positive for sqrt
        # optionally apply a simple skew adjustment to approximate right-skew
        if written_pool_type == "截断偏态（右偏）":
            # skew_k in [0,1], shift mean to the right by skew_k * sd, and slightly reduce sd
            mean_written = mean_written + skew_k * sd_written
            sd_written = sd_written * max(0.3, 1.0 - 0.4 * skew_k)
    else:
        # Use overall (untruncated) population moments
        mean_written = overall_written_mean
        sd_written = max(1e-6, overall_written_sd)
    # Fallback if calculation fails
    if math.isnan(mean_written) or math.isnan(sd_written):
        mean_written = max(min_written + 0.15 * written_full, written_mean_hint)

def estimate_interview_score(
    interview_mean: float, interview_sd: float, percentile: float, interview_full: float
) -> float:
    """Estimate an interview score from a percentile (0 best, 100 worst)."""
    percentile = clamp(percentile, 0.0, 100.0)
    p = 1 - percentile / 100.0
    z = normal_ppf(p)
    estimated = interview_mean + z * interview_sd
    return clamp(estimated, 0.0, interview_full)


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
) -> Tuple[float, float, float, float, float]:
    """Compute admission probability (analytic or MC).

    Returns: (probability, user_combined, cutoff, sd_combined, mean_combined)
    """
    # Use global mc_samples and applicants if available
    global mc_samples
    try:
        N = int(max(1, applicants))
    except Exception:
        N = int(max(1, overall_written_mean))

    # Monte-Carlo path (if mc_samples > 0)
    if globals().get("mc_samples", 0) and int(mc_samples) > 0:
        M = int(max(1, int(interview_count)))
        K = int(max(0, int(admit_count)))
        draws = int(mc_samples)

        success = 0
        cutoffs = []
        mean_combined_list = []
        combined_sds = []

        for _ in range(draws):
            # sample other applicants' written scores and include the user
            if N <= 1:
                other_written = np.array([])
            else:
                other_written = np.random.normal(loc=overall_written_mean, scale=max(1e-6, overall_written_sd), size=N - 1)
            all_written = np.concatenate([other_written, np.array([written_score])])

            # select top M by written score
            idx_sorted = np.argsort(all_written)
            top_idx = idx_sorted[-M:]
            top_written = all_written[top_idx]

            user_index = N - 1
            user_in_top = user_index in top_idx

            # simulate interview scores for top M
            top_interview = np.random.normal(loc=interview_mean, scale=max(1e-6, interview_sd), size=M)
            if user_in_top:
                pos = list(top_idx).index(user_index)
                top_interview[pos] = interview_score_est

            written_norm = (top_written / written_full) * 100
            interview_norm = (top_interview / interview_full) * 100
            combined = written_weight * written_norm + (1 - written_weight) * interview_norm

            # compute cutoff as K-th best among top M
            cutoff_sim = np.sort(combined)[-K] if (K <= M and K > 0) else np.min(combined)
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

        prob = success / float(draws) if draws > 0 else 0.0
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

    # Analytic (fast) path
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
        # 把高级选项放到折叠面板里，让主界面更清爽
    with st.expander("高级设置（展开可见）", expanded=False):
        estimate_mode = st.selectbox(
            "笔试分布估算模式",
            ["已知最高分", "历年进面分差值", "比例估算最高分"],
            index=0,
        )

        if estimate_mode == "已知最高分":
            known_max_written = st.number_input("已知最高笔试分", min_value=min_written, value=float(min_written * 1.2), step=1.0)
        elif estimate_mode == "历年进面分差值":
            historical_min_written = st.number_input("历年进面最低笔试分", min_value=0.0, value=float(min_written), step=1.0)
            historical_max_written = st.number_input("历年进面最高笔试分", min_value=historical_min_written, value=float(min_written * 1.2), step=1.0)
            diff = historical_max_written - historical_min_written
            st.write(f"差值: {diff:.1f}")
        elif estimate_mode == "比例估算最高分":
            ratio = st.slider("最高分比例（进面最低分的倍数）", 1.05, 1.30, 1.15, 0.01)
            estimated_max = min_written * ratio
            st.write(f"估算最高分: {estimated_max:.1f}")
        else:
            known_max_written = None
            historical_min_written = None
            historical_max_written = None
            diff = None
            ratio = None

            # 计算方式：解析近似或蒙特卡洛模拟
            compute_mode = st.selectbox("计算方式", ["解析近似", "蒙特卡洛模拟（更精确）"]) 
            use_mc = compute_mode.startswith("蒙特卡洛")
            if use_mc:
                mc_samples = st.slider("模拟次数（次）", 1000, 50000, 5000, step=1000)
            else:
                mc_samples = 0

            # 复试分布与面试分布模式（高级）
            dist_mode = st.selectbox(
                "复试分布计算模式",
                ["使用进面池截断后方差", "使用总体未截断方差"],
            )
            use_truncated_pool = dist_mode == "使用进面池截断后方差"
            st.caption(
                "说明：\n- “截断后方差”：把进面的人看成被筛出来的高分组，笔试分差会变小，最终更看重面试成绩。\n- “总体未截断”：不做筛选处理，直接用总体的均值与SD，适合已知总体或不想收缩方差时使用。"
            )

            # 进面笔试分布类型（用于截断后是否采用偏态近似，高级）
            written_pool_type = st.selectbox(
                "进面笔试分布类型",
                ["截断正态", "截断偏态（右偏）"],
            )
            if written_pool_type == "截断偏态（右偏）":
                skew_k = st.slider("偏态强度（右偏，0-1）", 0.0, 1.0, 0.4, 0.05)
                st.caption("说明：偏态表示进面人群分数右侧拉长（出现更多非常高分）。滑动越大，表示高分尾巴越长，结果更偏向少数高分。此为简易近似。")
            else:
                skew_k = 0.0
            st.caption(
                "提示：\n- ‘截断正态’：进面人群近似为截断后的常态分布；\n- ‘截断偏态（右偏）’：在截断基础上，向高分一侧延长尾部，模拟极高分更多的情况。"
            )

            interview_dist_type = st.selectbox("面试分布类型", ["正常（使用估算SD）", "更紧缩（SD乘以系数）"]) 
            if interview_dist_type == "更紧缩（SD乘以系数）":
                tighten_factor = st.slider("面试SD缩紧系数", 0.2, 1.0, 0.6, 0.05)
                st.caption(
                    "说明：把面试的随机波动（SD）乘以该系数可以模拟评分更一致的情况。\n"
                    "取值越小表示面试分更集中（面试对最终结果影响变小），取值越接近1表示按原始估算。建议试0.4–0.8。"
                )
            else:
                tighten_factor = 1.0
                st.caption(
                    "说明：默认使用估算的面试SD，表示历年面试分数的常见波动范围。\n"
                    "如果你对面试的离散程度没有特殊了解，保持默认；若观测到面试评分更集中，可选择“更紧缩”并调整系数。"
                )

            interview_sd = interview_sd_base * tighten_factor
            top_interview = np.random.normal(loc=interview_mean, scale=interview_sd, size=M)
            # if user in top, replace their interview score with estimated (fixed)
            if user_in_top:
                pos = list(top_idx).index(user_index)
                top_interview[pos] = interview_score_est

            # compute combined scores for top M
            written_norm = (top_written / written_full) * 100
            interview_norm = (top_interview / interview_full) * 100
            combined = written_weight * written_norm + (1 - written_weight) * interview_norm

            # compute cutoff as K-th best in combined among top M
            cutoff_sim = np.sort(combined)[-K] if K <= M else np.min(combined)
            cutoffs.append(float(cutoff_sim))
            mean_combined_list.append(float(np.mean(combined)))
            combined_sds.append(float(np.std(combined, ddof=1)))

            if user_in_top:
                # compute user's combined
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

    # Analytic (fast) path
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
    xs = [x_min + i * (x_max - x_min) / 200 for i in range(201)]
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

    chart = (base + user_rule + cutoff_rule + text_layer).properties(height=260)
    st.altair_chart(chart, use_container_width=True)


# UI -----------------------------------------------------------------------
st.set_page_config(page_title="上岸概率估算器", page_icon="🚀", layout="centered")
st.title("上岸概率估算器")
st.caption("说明：友情评估真朋友，客观评估不客观。")

mode = st.radio("评估模式", ["友情评估", "客观评估"], horizontal=True)

col_left, col_right = st.columns(2)
with col_left:
    applicants = st.number_input("报名人数", min_value=1, value=500, step=10)
    interview_count = st.number_input("进入复试人数", min_value=1, value=50, step=1)
    final_admit = st.number_input("最终录取人数", min_value=1, value=10, step=1)
    min_written = st.number_input("进面最低笔试分", min_value=0.0, value=120.0, step=1.0)

with col_right:
    your_written = st.number_input("你的笔试分", min_value=0.0, value=135.0, step=0.5)
    interview_mean = st.number_input("历年面试平均分", min_value=0.0, value=75.0, step=0.5)
    max_diff = st.number_input("面试最大分差（常见拉开差距）", min_value=1.0, value=5.0, step=0.5)
    interview_percentile = st.slider(
        "自估面试位次（%）0最好 100最末", min_value=0, max_value=100, value=30, step=1
    )

st.divider()

col_a, col_b, col_c = st.columns(3)
with col_a:
    written_full = st.number_input("笔试满分", min_value=1.0, value=200.0, step=1.0)
with col_b:
    interview_full = st.number_input("面试满分", min_value=1.0, value=100.0, step=1.0)
with col_c:
    written_weight_pct = st.slider("笔试占比（%）", 0, 100, 50, step=5)

estimate_mode = st.selectbox(
    "笔试分布估算模式",
    ["已知最高分", "历年进面分差值", "比例估算最高分"],
    index=0,
)

if estimate_mode == "已知最高分":
    known_max_written = st.number_input("已知最高笔试分", min_value=min_written, value=float(min_written * 1.2), step=1.0)
elif estimate_mode == "历年进面分差值":
    historical_min_written = st.number_input("历年进面最低笔试分", min_value=0.0, value=float(min_written), step=1.0)
    historical_max_written = st.number_input("历年进面最高笔试分", min_value=historical_min_written, value=float(min_written * 1.2), step=1.0)
    diff = historical_max_written - historical_min_written
    st.write(f"差值: {diff:.1f}")
elif estimate_mode == "比例估算最高分":
    ratio = st.slider("最高分比例（进面最低分的倍数）", 1.05, 1.30, 1.15, 0.01)
    estimated_max = min_written * ratio
    st.write(f"估算最高分: {estimated_max:.1f}")
else:
    known_max_written = None
    historical_min_written = None
    historical_max_written = None
    diff = None
    ratio = None

written_weight = written_weight_pct / 100

# 计算方式：解析近似或蒙特卡洛模拟
compute_mode = st.selectbox("计算方式", ["解析近似", "蒙特卡洛模拟（更精确）"]) 
use_mc = compute_mode.startswith("蒙特卡洛")
if use_mc:
    mc_samples = st.slider("模拟次数（次）", 1000, 50000, 5000, step=1000)
else:
    mc_samples = 0

# Estimate overall distribution
overall_written_mean, overall_written_sd = estimate_overall_from_data(
    applicants=applicants,
    interview_count=interview_count,
    min_written=min_written,
    estimate_mode=estimate_mode,
    known_max=known_max_written if estimate_mode == "已知最高分" else None,
    historical_min=historical_min_written if estimate_mode == "历年进面分差值" else None,
    historical_max=historical_max_written if estimate_mode == "历年进面分差值" else None,
    ratio=ratio if estimate_mode == "比例估算最高分" else None,
)

# Default values for other parameters
written_sd = max(8.0, 0.08 * written_full)
written_mean_hint = min_written + 0.15 * written_full
interview_sd_base = max(8.0, max_diff * 1.4)

# 复试分布与面试分布模式
dist_mode = st.selectbox(
    "复试分布计算模式",
    ["使用进面池截断后方差", "使用总体未截断方差"],
)
use_truncated_pool = dist_mode == "使用进面池截断后方差"
st.caption(
    "说明：\n- “截断后方差”：把进面的人看成被筛出来的高分组，笔试分差会变小，最终更看重面试成绩。\n- “总体未截断”：不做筛选处理，直接用总体的均值和SD，适合已知总体或不想收缩方差时使用。"
)

# 进面笔试分布类型（用于截断后是否采用偏态近似）
written_pool_type = st.selectbox(
    "进面笔试分布类型",
    ["截断正态", "截断偏态（右偏）"],
)
if written_pool_type == "截断偏态（右偏）":
    skew_k = st.slider("偏态强度（右偏，0-1）", 0.0, 1.0, 0.4, 0.05)
    st.caption("说明：偏态表示进面人群分数右侧拉长（出现更多非常高分）。滑动越大，表示高分尾巴越长，结果更偏向少数高分。此为简易近似。")
else:
    skew_k = 0.0
st.caption(
    "提示：\n- ‘截断正态’：进面人群近似为截断后的常态分布；\n- ‘截断偏态（右偏）’：在截断基础上，向高分一侧延长尾部，模拟极高分更多的情况。"
)

interview_dist_type = st.selectbox("面试分布类型", ["正常（使用估算SD）", "更紧缩（SD乘以系数）"]) 
if interview_dist_type == "更紧缩（SD乘以系数）":
    tighten_factor = st.slider("面试SD缩紧系数", 0.2, 1.0, 0.6, 0.05)
    st.caption(
        "说明：把面试的随机波动（SD）乘以该系数可以模拟评分更一致的情况。\n"
        "取值越小表示面试分更集中（面试对最终结果影响变小），取值越接近1表示按原始估算。建议试0.4–0.8。"
    )
else:
    tighten_factor = 1.0
    st.caption(
        "说明：默认使用估算的面试SD，表示历年面试分数的常见波动范围。\n"
        "如果你对面试的离散程度没有特殊了解，保持默认；若观测到面试评分更集中，可选择“更紧缩”并调整系数。"
    )

interview_sd = interview_sd_base * tighten_factor

if st.button("开始评估", use_container_width=True, type="primary"):
    entered = your_written >= min_written
    interview_score_est = estimate_interview_score(
        interview_mean=interview_mean,
        interview_sd=interview_sd,
        percentile=float(interview_percentile),
        interview_full=interview_full,
    )

    prob, user_combined, cutoff, sd_combined, mean_combined = compute_probability(
        entered_interview=entered,
        written_score=your_written,
        interview_score_est=interview_score_est,
        written_full=written_full,
        interview_full=interview_full,
        written_weight=written_weight,
        min_written=min_written,
        interview_mean=interview_mean,
        interview_sd=interview_sd,
        written_sd=written_sd,
        written_mean_hint=written_mean_hint,
        admit_count=final_admit,
        interview_count=interview_count,
        overall_written_mean=overall_written_mean,
        overall_written_sd=overall_written_sd,
        use_truncated_pool=use_truncated_pool,
        written_pool_type=written_pool_type,
        skew_k=skew_k,
    )

    # compute displayed written mean/sd based on mode
    if use_truncated_pool:
        mean_written_display = truncated_normal_mean(overall_written_mean, overall_written_sd, min_written)
        sd_written_display = math.sqrt(max(1e-6, truncated_normal_var(overall_written_mean, overall_written_sd, min_written)))
        if written_pool_type == "截断偏态（右偏）":
            mean_written_display = mean_written_display + skew_k * sd_written_display
            sd_written_display = sd_written_display * max(0.3, 1.0 - 0.4 * skew_k)
    else:
        mean_written_display = overall_written_mean
        sd_written_display = overall_written_sd

    if not entered:
        st.error("⚠️ 你的笔试分低于进面最低分，录取概率为 0%")
        prob = 0.0

    # Calculate written exam ranking
    if overall_written_sd > 0:
        p_higher = 1 - normal_cdf((your_written - overall_written_mean) / overall_written_sd)
        rank = round(p_higher * applicants) + 1
        written_ranking = f"{rank}"
    else:
        written_ranking = "未知"

    st.metric("录取概率", f"{prob * 100:.1f}%")
    st.progress(prob)

    cols = st.columns(4)
    cols[0].metric("你的综合分 (0-100)", f"{user_combined:.1f}")
    cols[1].metric("预计录取线综合分", f"{cutoff:.1f}")
    cols[2].metric("笔试排名", f"约 {written_ranking} 名")
    cols[3].metric("预估面试分", f"{interview_score_est:.1f}")

    render_distribution_chart(
        mean_c=mean_combined,
        sd_c=sd_combined,
        user_c=user_combined,
        cutoff_c=cutoff,
    )

    with st.expander("计算假设与细节"):
        st.write(
            f"全体考生笔试均值≈{overall_written_mean:.1f}，全体考生笔试SD≈{overall_written_sd:.1f}。"
        )
        st.write(
            f"进入面试人群笔试均值≈{mean_written_display:.1f}，"
            f"笔试SD≈{sd_written_display:.1f}；面试均值≈{interview_mean:.1f}，面试SD≈{interview_sd:.1f}。"
        )
        st.write(
            f"录取比例≈{final_admit}/{interview_count}，约 {final_admit / interview_count * 100:.1f}%；"
            f"综合分SD≈{sd_combined:.1f}。"
        )
        st.write(
            f"面试自估位次 {interview_percentile}% → 预估面试分 {interview_score_est:.1f}。"
        )
        if not entered:
            st.warning("你的笔试分低于进面线，结果仅供参考。")

    render_feedback(probability=prob, mode=mode, entered=entered)

    st.caption(
        "提示：移动端可直接用浏览器访问 Streamlit 部署地址。"
    )
