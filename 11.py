import math
import numpy as np
from typing import Tuple

import altair as alt
import pandas as pd
import streamlit as st


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


def normal_ppf_vec(p_array):
    """Vectorized wrapper for normal_ppf that accepts numpy arrays."""
    pa = np.atleast_1d(p_array)
    out = np.array([normal_ppf(float(x)) for x in pa])
    if np.isscalar(p_array):
        return float(out[0])
    return out


def normal_pdf(z: float) -> float:
    # Support both scalar and numpy array inputs
    coef = 1.0 / math.sqrt(2 * math.pi)
    try:
        # if z is array-like (numpy), use numpy operations
        if isinstance(z, (list, tuple)) or hasattr(z, "__array__"):
            zz = np.asarray(z)
            return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * zz * zz)
    except Exception:
        pass
    return coef * math.exp(-0.5 * z * z)


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
    # Correct variance formula for lower-truncated normal: Var = sigma^2 * (1 + z_a * lambda - lambda^2)
    var = sigma ** 2 * (1 + z_a * ratio - ratio * ratio)
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
    use_overall_moments_for_pool: bool = False,
    skew_k: float = 0.4,
) -> Tuple[float, float]:
    """Approximate combined score mean and sd for the interview pool using truncated normal for written scores."""
    # Use truncated normal for written scores in interview pool
    if use_truncated_pool:
        if use_overall_moments_for_pool:
            # Use overall population moments but account for truncation by
            # computing the truncated written PDF and numerically convolving
            # it with the interview normal PDF (both on 0-100 scale). This
            # produces the exact combined PDF without assuming normality.
            mu = overall_written_mean
            sigma = max(1e-6, overall_written_sd)
            a = min_written

            # Build empirical KDE for written scores (use samples from overall normal if raw samples not available)
            xmax = written_full
            # grid size for convolution
            M = 2048
            y_min = 0.0
            y_max = 100.0
            ys = np.linspace(y_min, y_max, M)
            dy = ys[1] - ys[0]

            # Draw a large sample from overall normal and truncate at a
            # Use a reasonably large sample to approximate the empirical density
            n_samp = 200000
            samp = np.random.normal(loc=mu, scale=sigma, size=n_samp)
            samp = samp[samp >= a]
            if samp.size < 100:
                # fallback to analytic truncated normal if sampling failed
                mean_written = truncated_normal_mean(mu, sigma, a)
                var_written = truncated_normal_var(mu, sigma, a)
                sd_written = math.sqrt(max(1e-6, var_written))
                # map analytic pdf to normalized axis
                xs = np.linspace(a, xmax, M)
                zs = (xs - mu) / sigma
                f_raw = normal_pdf(zs) / sigma
                Phi_a = normal_cdf((a - mu) / sigma)
                mass = 1.0 - Phi_a
                if mass <= 0:
                    return mean_written, sd_written
                g_raw = f_raw / mass
                written_norm_xs = xs / written_full * 100.0
                dx_dy = written_full / 100.0
                g_norm = g_raw * dx_dy
                sort_idx = np.argsort(written_norm_xs)
                written_sorted = written_norm_xs[sort_idx]
                g_sorted = g_norm[sort_idx]
                g_on_grid = np.interp(ys, written_sorted, g_sorted, left=0.0, right=0.0)
            else:
                # map samples to normalized 0-100 scale and compute histogram density
                samp_norm = np.clip(samp / written_full * 100.0, 0.0, 100.0)
                hist, edges = np.histogram(samp_norm, bins=M, range=(0.0, 100.0), density=True)
                bin_centers = (edges[:-1] + edges[1:]) / 2.0
                # smooth histogram with Gaussian kernel via FFT to approximate KDE
                # bandwidth heuristics: use a fraction of empirical std, but at least 1.0
                bw = max(1.0, np.std(samp_norm) * 0.2)
                # build kernel on same grid (centered)
                kernel_x = (np.arange(M) - M // 2) * dy
                kernel = np.exp(-0.5 * (kernel_x / bw) ** 2)
                kernel = kernel / (np.trapz(kernel, kernel_x))
                fft_len = int(2 ** np.ceil(np.log2(M * 2)))
                H = np.fft.rfft(hist, n=fft_len)
                K = np.fft.rfft(np.roll(kernel, M // 2), n=fft_len)
                smooth = np.fft.irfft(H * K, n=fft_len)[:M]
                # ensure non-negative and normalized
                smooth[smooth < 0] = 0.0
                sum_s = np.trapz(smooth, ys)
                if sum_s <= 0:
                    # fallback to histogram without smoothing
                    g_on_grid = hist
                else:
                    g_on_grid = smooth / sum_s

            # interview pdf on normalized axis (0-100)
            i_mu = interview_mean / interview_full * 100.0
            i_sd = max(1e-6, interview_sd / interview_full * 100.0)
            i_pdf = (1.0 / (i_sd * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((ys - i_mu) / i_sd) ** 2)

            # We need distribution of combined = w * W_norm + (1-w) * I_norm.
            # Use change-of-variables: f_S(s) = (1/w) * f_W(s/w) for s in [0,w*100],
            # and f_T(t) = (1/(1-w)) * f_I(t/(1-w)) for t in [0,(1-w)*100].
            w = written_weight
            if w <= 0 or w >= 1:
                # degenerate weights: fall back to analytic moments
                mean_combined = calc_combined_score(
                    written_score=truncated_normal_mean(mu, sigma, a),
                    interview_score=interview_mean,
                    written_full=written_full,
                    interview_full=interview_full,
                    written_weight=written_weight,
                )
                sd_combined = math.sqrt(
                    (written_weight ** 2) * ((math.sqrt(max(1e-6, truncated_normal_var(mu, sigma, a))) / written_full) * 100) ** 2
                    + ((1 - written_weight) ** 2) * ((interview_sd / interview_full) * 100) ** 2
                )
                return mean_combined, sd_combined

            # f_W defined on ys grid: g_on_grid (pdf over 0..100)
            f_W = g_on_grid.copy()
            # ensure normalization
            sum_fW = np.trapz(f_W, ys)
            if sum_fW <= 0:
                mean_combined = calc_combined_score(
                    written_score=truncated_normal_mean(mu, sigma, a),
                    interview_score=interview_mean,
                    written_full=written_full,
                    interview_full=interview_full,
                    written_weight=written_weight,
                )
                sd_combined = math.sqrt(
                    (written_weight ** 2) * ((math.sqrt(max(1e-6, truncated_normal_var(mu, sigma, a))) / written_full) * 100) ** 2
                    + ((1 - written_weight) ** 2) * ((interview_sd / interview_full) * 100) ** 2
                )
                return mean_combined, sd_combined
            f_W = f_W / sum_fW

            # build f_S(s) on ys grid: f_S(s) = (1/w) * f_W(s/w)
            s_coords = ys
            lookup_coords = s_coords / w
            f_S = np.interp(lookup_coords, ys, f_W, left=0.0, right=0.0) / w

            # build f_T(t) on ys grid: f_T(t) = (1/(1-w)) * f_I(t/(1-w))
            t_coords = ys
            lookup_coords_t = t_coords / (1.0 - w)
            f_I = i_pdf.copy()
            sum_fI = np.trapz(f_I, ys)
            if sum_fI <= 0:
                f_I = f_I
            else:
                f_I = f_I / sum_fI
            f_T = np.interp(lookup_coords_t, ys, f_I, left=0.0, right=0.0) / (1.0 - w)

            # perform convolution f_C = f_S * f_T via FFT
            fft_len = int(2 ** np.ceil(np.log2(len(ys) * 2)))
            G = np.fft.rfft(f_S, n=fft_len)
            I = np.fft.rfft(f_T, n=fft_len)
            conv = np.fft.irfft(G * I, n=fft_len)[:len(ys)] * dy

            # normalize conv to area 1
            conv_sum = np.trapz(conv, ys)
            if conv_sum <= 0:
                mean_combined = calc_combined_score(
                    written_score=truncated_normal_mean(mu, sigma, a),
                    interview_score=interview_mean,
                    written_full=written_full,
                    interview_full=interview_full,
                    written_weight=written_weight,
                )
                sd_combined = math.sqrt(
                    (written_weight ** 2) * ((math.sqrt(max(1e-6, truncated_normal_var(mu, sigma, a))) / written_full) * 100) ** 2
                    + ((1 - written_weight) ** 2) * ((interview_sd / interview_full) * 100) ** 2
                )
            else:
                conv = conv / conv_sum
                # store numeric convolution for later plotting (non-normal analytic curve)
                try:
                    global _LAST_CONV_YS, _LAST_CONV_PDF
                    _LAST_CONV_YS = ys.copy()
                    _LAST_CONV_PDF = conv.copy()
                except Exception:
                    _LAST_CONV_YS = None
                    _LAST_CONV_PDF = None
                mean_combined = float(np.trapz(ys * conv, ys))
                var_combined = float(np.trapz((ys - mean_combined) ** 2 * conv, ys))
                sd_combined = math.sqrt(max(1e-6, var_combined))

            return mean_combined, sd_combined
        else:
            # Use overall-normal then truncate at `min_written`, and recompute
            # the conditional mean/variance for the high-score segment.
            mean_written = truncated_normal_mean(overall_written_mean, overall_written_sd, min_written)
            var_written = truncated_normal_var(overall_written_mean, overall_written_sd, min_written)
            sd_written = math.sqrt(max(1e-6, var_written))  # Ensure positive for sqrt
    else:
        # Use overall (untruncated) population moments
        mean_written = overall_written_mean
        sd_written = max(1e-6, overall_written_sd)
    # Fallback if calculation fails
    if math.isnan(mean_written) or math.isnan(sd_written):
        mean_written = max(min_written + 0.15 * written_full, written_mean_hint)
        sd_written = max(5.0, written_sd)

    interview_sd = max(5.0, interview_sd)

    mean_combined = calc_combined_score(
        written_score=mean_written,
        interview_score=interview_mean,
        written_full=written_full,
        interview_full=interview_full,
        written_weight=written_weight,
    )

    # Assuming written and interview are roughly independent
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
    # percentile here: 0 is best, 100 is worst
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
    use_overall_moments_for_pool: bool = False,
    skew_k: float = 0.4,
    use_mc: bool = False,
    mc_samples: int = 0,
    applicants: int = 0,
) -> Tuple[float, float, float, float, float]:
    """Return probability, user_combined, cutoff, sd_combined, mean_combined.
    Supports analytic estimate or Monte-Carlo simulation when `use_mc` is True."""
    # Monte-Carlo simulation path
    if use_mc:
        N = int(applicants)
        M = int(interview_count)
        K = int(admit_count)
        mu = overall_written_mean
        sigma = overall_written_sd

        success = 0
        cutoffs = []
        combined_sds = []
        mean_combined_list = []
        combined_all = []

        for _ in range(int(mc_samples)):
            # Sample all other applicants from the overall normal (NOT conditional).
            others = np.random.normal(loc=mu, scale=sigma, size=N - 1)
            all_written = np.concatenate([others, np.array([written_score])])

            # Determine which applicants meet the min_written cutoff (enter interview pool)
            eligible_idx = np.where(all_written >= min_written)[0]

            # If more applicants passed than interview slots, pick top M by written among those.
            if len(eligible_idx) > M:
                eligible_scores = all_written[eligible_idx]
                sel = np.argsort(eligible_scores)[-M:]
                top_idx = eligible_idx[sel]
            else:
                top_idx = eligible_idx

            # Check whether user is in the selected interview pool
            user_index = len(all_written) - 1
            user_in_top = user_index in top_idx

            # If user is not eligible (below min_written), they cannot be in pool
            if not user_in_top and (written_score < min_written):
                # user not in pool; selected pool is top_idx as above
                pass

            top_written = all_written[top_idx] if len(top_idx) > 0 else np.array([])

            # sample interview scores for the selected pool
            top_interview = np.random.normal(loc=interview_mean, scale=interview_sd, size=len(top_written))
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
            combined_all.append(combined)

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
        # concatenate combined samples for plotting KDE/histogram
        if len(combined_all):
            combined_all_arr = np.concatenate(combined_all)
        else:
            combined_all_arr = np.array([])

        mc_info = {
            "cutoffs": np.array(cutoffs),
            "mean_combined_list": np.array(mean_combined_list),
            "combined_sds": np.array(combined_sds),
            "combined_all": combined_all_arr,
        }
        return prob, user_combined, cutoff, sd_combined, mean_combined, mc_info

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
        use_overall_moments_for_pool=use_overall_moments_for_pool,
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
                position: relative;
                margin: 0 auto;
                width: 8px;
                height: 8px;
                background: transparent;
                pointer-events: none;
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
                0% { transform: scale(0.2); opacity: 1; }
                80% { opacity: 1; }
                100% { transform: scale(1.1); opacity: 0; }
            }
        </style>
        """
        st.components.v1.html(fireworks_html, height=300, width=600)


def show_confetti():
        """Simple confetti effect using CSS animation."""
        confetti_html = """
        <div class="confetti-wrap">
            <div class="confetti">
                <div class="c c1"></div>
                <div class="c c2"></div>
                <div class="c c3"></div>
                <div class="c c4"></div>
                <div class="c c5"></div>
            </div>
        </div>
        <style>
            .confetti-wrap{ position: fixed; inset:0; width:100%; height:100%; display:flex; align-items:flex-start; justify-content:center; pointer-events:none; z-index:9998 }
            .confetti { position: relative; width:100%; height:100%; overflow: hidden; }
            .confetti .c { width: 12px; height: 18px; position: absolute; top: -20px; opacity: 0.95; }
            .c1 { background: #ff4d4f; left: 10%; animation: fall 2.2s linear infinite; }
            .c2 { background: #ffc53d; left: 30%; animation: fall 2.6s linear infinite; }
            .c3 { background: #73d13d; left: 50%; animation: fall 2.4s linear infinite; }
            .c4 { background: #40a9ff; left: 70%; animation: fall 2.8s linear infinite; }
            .c5 { background: #9254de; left: 85%; animation: fall 2.0s linear infinite; }
            @keyframes fall { 0% { transform: translateY(-20px) rotate(0deg); } 100% { transform: translateY(120vh) rotate(360deg); } }
        </style>
        """
        st.components.v1.html(confetti_html, height=600, width=900)


def show_streamers():
        """Streamers animation (subtle)"""
        stream_html = """
        <div class="stream-wrap">
            <div class="streamers" aria-hidden="true">
                <div class="s s1"></div>
                <div class="s s2"></div>
                <div class="s s3"></div>
            </div>
        </div>
        <style>
            .stream-wrap{ position: fixed; inset:0; width:100%; height:100%; display:flex; align-items:flex-start; justify-content:center; pointer-events:none; z-index:9997 }
            .streamers{ position: relative; width:100%; height:100%; overflow:hidden }
            .streamers .s{ position:absolute; top:-40px; width:6px; height:80px; opacity:0.85 }
            .s1{ left:20%; background:linear-gradient(#ff85c0,#ff4d4f); animation:drop 1.6s linear infinite }
            .s2{ left:50%; background:linear-gradient(#5cdbd3,#40a9ff); animation:drop 1.9s linear infinite }
            .s3{ left:80%; background:linear-gradient(#ffd666,#ffc53d); animation:drop 2.1s linear infinite }
            @keyframes drop{ 0%{ transform: translateY(-40px) } 100%{ transform: translateY(160vh) } }
        </style>
        """
        st.components.v1.html(stream_html, height=600, width=900)


def render_feedback(probability: float, mode: str, entered: bool):
    if not entered:
        st.write("❌ 未进入面试，无法评估录取概率。")
        return

    if mode.startswith("友情"):
        tiers = [
            (0.8, "🐳 超有戏！上岸buff叠满啦～"),
            (0.6, "🐰 超棒的！稳稳拿捏机会✨"),
            (0.4, "🐱 冲鸭！潜力值拉满咯～"),
            (0.2, "🐥 加油鸭！还有超多希望～"),
            (0.0, "🐾 进面就是胜利！冲就对啦～"),
        ]
    else:
        tiers = [
            (0.9, "录取概率极高，面试正常发挥即可上岸"),
            (0.8, "优势明显，重点打磨面试细节避免失误"),
            (0.6, "有较大录取机会，需全神贯注准备面试"),
            (0.4, "录取概率持平，需针对性补强薄弱环节"),
            (0.2, "录取难度较大，需高强度冲刺提升表现"),
            (0.0, "录取风险高，建议同步准备备选方案"),
        ]

    for threshold, text in tiers:
        if probability >= threshold:
            st.write(text)
            break

    # In 友情评估模式, show different animations by probability tier
    if mode.startswith("友情"):
        if probability >= 0.9:
            show_fireworks()
            st.balloons()
        elif probability >= 0.6:
            show_confetti()
        elif probability >= 0.4:
            show_streamers()
        elif probability >= 0.2:
            show_confetti()


def render_distribution_chart(mean_c: float, sd_c: float, user_c: float, cutoff_c: float):
    # If sd is extremely small, show a vertical rule at mean instead of failing.
    if sd_c <= 1e-6:
        st.info("分布标准差过小，显示均值参考线。")
        # show a simple rule at mean
        base = (
            alt.Chart(pd.DataFrame({"score": [mean_c]}))
            .mark_rule(color="#1890ff", strokeWidth=2)
            .encode(x=alt.X("score:Q", title="综合分"))
        )
    else:
        x_min = mean_c - 3.5 * sd_c
        x_max = mean_c + 3.5 * sd_c
        xs = [x_min + i * (x_max - x_min) / 200 for i in range(201)]
        ys = [normal_pdf((x - mean_c) / sd_c) / sd_c for x in xs]
        df = pd.DataFrame({"score": xs, "density": ys})

        base = alt.Chart(df).mark_line(color="#1890ff", strokeWidth=2).encode(
            x=alt.X("score:Q", title="综合分"),
            y=alt.Y("density:Q", title="密度", axis=alt.Axis(labels=False)),
        )

    user_rule = (
        alt.Chart(pd.DataFrame({"score": [user_c], "label": ["你"]}))
        .mark_rule(color="#fa541c", strokeWidth=2, strokeDash=[5, 3])
        .encode(x=alt.X("score:Q"))
    )

    cutoff_rule = (
        alt.Chart(pd.DataFrame({"score": [cutoff_c], "label": ["录取线"]}))
        .mark_rule(color="#52c41a", strokeWidth=2)
        .encode(x=alt.X("score:Q"))
    )

    # Text labels: position depends on whether we have density values
    try:
        max_y = max(ys) if 'ys' in locals() else 1.0
    except Exception:
        max_y = 1.0
    text_layer = (
        alt.Chart(
            pd.DataFrame(
                {
                    "score": [user_c, cutoff_c],
                    "density": [max_y * 0.9, max_y * 0.8],
                    "label": ["你的分", "预计录取线"],
                }
            )
        )
        .mark_text(dy=-6, fontSize=11)
        .encode(x=alt.X("score:Q"), y=alt.Y("density:Q"), text=alt.Text("label:N"), color=alt.value("#595959"))
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

# 在“友情评估”中提供初级/高级界面切换：
if mode == "友情评估":
    ui_level = st.radio("界面模式", ["初级模式", "高级模式"], index=0, horizontal=True)
elif mode == "客观评估":
    ui_level = "初级模式"
else:
    ui_level = "高级模式"



# 仅在高级模式下显示下面的高级选项
if ui_level == "高级模式":
    # 计算方式：解析近似或蒙特卡洛模拟
    compute_mode = st.selectbox("计算方式", ["解析近似", "蒙特卡洛模拟（更精确）"]) 
    use_mc = compute_mode.startswith("蒙特卡洛")
    if use_mc:
        mc_samples = st.slider("模拟次数（次）", 1000, 50000, 5000, step=1000)
    else:
        mc_samples = 0
    
    # 统一使用截断后方差（删除多种复试分布计算模式）
    use_truncated_pool = True
    skew_k = 0.0

    # 面试分布类型（高级）
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
    # 高级选项：是否把进面笔试人群视为新的正态分布（仅在解析近似时可选）
    if not use_mc:
        use_overall_moments_for_pool = st.checkbox(
            "不把进面笔试视为新的正态分布（使用总体均值和SD）",
            value=False,
        )
    else:
        use_overall_moments_for_pool = False
    # MC模式下：是否在MC图上同时叠加解析曲线
    if use_mc:
        show_analytic_in_mc = st.checkbox("在MC图上叠加解析正态曲线", value=False)
    else:
        show_analytic_in_mc = True
else:
    # 初级模式：使用安全的默认值，隐藏高级控件
    compute_mode = "解析近似"
    use_mc = False
    mc_samples = 0
    use_truncated_pool = True
    written_pool_type = "截断正态"
    skew_k = 0.0
    interview_dist_type = "正常（使用估算SD）"
    tighten_factor = 1.0
    use_overall_moments_for_pool = False
    show_analytic_in_mc = True

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

interview_sd = interview_sd_base * tighten_factor

if st.button("开始评估", use_container_width=True, type="primary"):
    entered = your_written >= min_written
    interview_score_est = estimate_interview_score(
        interview_mean=interview_mean,
        interview_sd=interview_sd,
        percentile=float(interview_percentile),
        interview_full=interview_full,
    )

    
    mc_info = None

    # If in 客观评估 (objective) and the user has entered the interview cutoff,
    # short-circuit to 100% admission and show celebratory animation.
    if mode == "客观评估" and entered:
        user_combined = calc_combined_score(
            written_score=your_written,
            interview_score=interview_score_est,
            written_full=written_full,
            interview_full=interview_full,
            written_weight=written_weight,
        )
        prob = 1.0
        cutoff = user_combined
        sd_combined = 0.0
        mean_combined = user_combined
        mc_info = None
        # simple celebrations
        show_fireworks()
        st.balloons()
    else:
        if use_mc:
            # compute_probability returns mc_info as 6th element in MC mode
            prob, user_combined, cutoff, sd_combined, mean_combined, mc_info = compute_probability(
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
                use_overall_moments_for_pool=use_overall_moments_for_pool,
                use_mc=use_mc,
                mc_samples=mc_samples,
                applicants=applicants,
                skew_k=skew_k,
            )
        else:
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
                use_overall_moments_for_pool=use_overall_moments_for_pool,
                use_mc=use_mc,
                mc_samples=mc_samples,
                applicants=applicants,
                skew_k=skew_k,
            )

    # compute displayed written mean/sd based on mode
    if use_truncated_pool:
        mean_written_display = truncated_normal_mean(overall_written_mean, overall_written_sd, min_written)
        sd_written_display = math.sqrt(max(1e-6, truncated_normal_var(overall_written_mean, overall_written_sd, min_written)))
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

    # Show analytic curve (normal) or numeric-convolution empirical analytic curve
    if not use_mc or show_analytic_in_mc:
        st.subheader("解析近似（数值/正态）")
        # If we computed a numeric convolution from empirical KDE, plot it directly
        if (not use_mc) and use_overall_moments_for_pool and (_LAST_CONV_YS is not None) and (_LAST_CONV_PDF is not None):
            try:
                df_conv = pd.DataFrame({"score": _LAST_CONV_YS, "density": _LAST_CONV_PDF})
                chart_conv = (
                    alt.Chart(df_conv).mark_line(color="#1890ff", strokeWidth=2)
                    .encode(x=alt.X("score:Q", title="综合分"), y=alt.Y("density:Q", title="密度"))
                )
                user_rule = (
                    alt.Chart(pd.DataFrame({"score": [user_combined], "label": ["你"]}))
                    .mark_rule(color="#fa541c", strokeWidth=2, strokeDash=[5, 3])
                    .encode(x=alt.X("score:Q"))
                )
                cutoff_rule = (
                    alt.Chart(pd.DataFrame({"score": [cutoff], "label": ["录取线"]}))
                    .mark_rule(color="#52c41a", strokeWidth=2)
                    .encode(x=alt.X("score:Q"))
                )
                chart = (chart_conv + user_rule + cutoff_rule).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                render_distribution_chart(
                    mean_c=mean_combined,
                    sd_c=sd_combined,
                    user_c=user_combined,
                    cutoff_c=cutoff,
                )
        else:
            render_distribution_chart(
                mean_c=mean_combined,
                sd_c=sd_combined,
                user_c=user_combined,
                cutoff_c=cutoff,
            )

    # If Monte-Carlo was used, show empirical histogram/KDE and cutoff distribution
    if mc_info is not None:
        combined_all = mc_info.get("combined_all", np.array([]))
        cutoffs_arr = mc_info.get("cutoffs", np.array([]))

        if combined_all.size:
            # Legend / explanation for MC plot
            legend_items = [
                "<span style='display:inline-block;width:12px;height:12px;background:#1890ff;margin-right:6px'></span>MC 经验密度",
            ]
            if show_analytic_in_mc:
                legend_items.append("<span style='display:inline-block;width:12px;height:12px;background:#fa541c;margin-left:12px;margin-right:6px'></span>解析正态曲线")
            legend_items.append("<span style='display:inline-block;width:12px;height:12px;background:#fa8c16;margin-left:12px;margin-right:6px'></span>你的分")
            legend_items.append("<span style='display:inline-block;width:12px;height:12px;background:#52c41a;margin-left:12px;margin-right:6px'></span>录取线均值/区间")
            legend_html = "<div style='font-size:14px;'>" + " ".join(legend_items) + "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

            df_comb = pd.DataFrame({"score": combined_all})
            # KDE area
            kde = (
                alt.Chart(df_comb)
                .transform_density('score', as_=['score', 'density'])
                .mark_area(opacity=0.3, color="#1890ff")
                .encode(x=alt.X("score:Q", title="综合分"), y=alt.Y("density:Q", title="密度"))
            )

            # analytic normal curve (from analytic mean/sd)
            x_min = max(0.0, float(np.min(combined_all)))
            x_max = min(100.0, float(np.max(combined_all)))
            xs = np.linspace(x_min, x_max, 201)
            ys = normal_pdf((xs - mean_combined) / sd_combined) / sd_combined
            df_a = pd.DataFrame({"score": xs, "density": ys})
            analytic = alt.Chart(df_a).mark_line(color="#fa541c", strokeWidth=2).encode(x=alt.X("score:Q"), y=alt.Y("density:Q"))

            user_rule = (
                alt.Chart(pd.DataFrame({"score": [user_combined], "label": ["你"]}))
                .mark_rule(color="#fa8c16", strokeWidth=2)
                .encode(x=alt.X("score:Q"))
            )

            # cutoff percentiles
            if cutoffs_arr.size:
                lo = float(np.percentile(cutoffs_arr, 2.5))
                hi = float(np.percentile(cutoffs_arr, 97.5))
                cut_mean = float(np.mean(cutoffs_arr))
            else:
                lo = cutoff
                hi = cutoff
                cut_mean = cutoff

            # show cutoff stats
            st.write(f"录取线均值：{cut_mean:.2f}；95% CI：{lo:.2f} - {hi:.2f} （基于 {mc_samples} 次模拟）")

            cutoff_rule = (
                alt.Chart(pd.DataFrame({"score": [cut_mean], "label": ["录取线均值"]}))
                .mark_rule(color="#52c41a", strokeWidth=2)
                .encode(x=alt.X("score:Q"))
            )

            ci_lo = (
                alt.Chart(pd.DataFrame({"score": [lo]})).mark_rule(color="#52c41a", strokeWidth=1, opacity=0.6).encode(x=alt.X("score:Q"))
            )
            ci_hi = (
                alt.Chart(pd.DataFrame({"score": [hi]})).mark_rule(color="#52c41a", strokeWidth=1, opacity=0.6).encode(x=alt.X("score:Q"))
            )

            # assemble layers depending on whether analytic overlay requested
            layers = [kde]
            if show_analytic_in_mc:
                layers.append(analytic)
            layers.extend([user_rule, cutoff_rule, ci_lo, ci_hi])
            chart_mc = layers[0]
            for layer in layers[1:]:
                chart_mc = chart_mc + layer
            chart_mc = chart_mc.properties(height=300)
            st.altair_chart(chart_mc, use_container_width=True)

        # cutoff histogram
        if cutoffs_arr.size:
            df_cut = pd.DataFrame({"cutoff": cutoffs_arr})
            cut_hist = (
                alt.Chart(df_cut).mark_bar(opacity=0.6, color="#52c41a").encode(
                    x=alt.X("cutoff", bin=alt.Bin(maxbins=30), title="录取线（综合分）"),
                    y=alt.Y("count()", title="频数"),
                )
            )
            st.altair_chart(cut_hist.properties(height=160), use_container_width=True)

        # show MC probability CI
        if mc_samples > 0:
            p = prob
            se = math.sqrt(p * (1 - p) / mc_samples)
            lo_p = max(0.0, p - 1.96 * se)
            hi_p = min(1.0, p + 1.96 * se)
            st.write(f"蒙特卡洛概率：{p*100:.1f}%，95% 置信区间：{lo_p*100:.1f}% - {hi_p*100:.1f}% （基于 {mc_samples} 次模拟）")

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
