import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln

DATA = "data/"
FIG = "figures/"
N_NAME = "matthewmcentee"

laser_T = [1, 2, 5, 10]
thermal_T = [1, 1, 1, 1]


def load_counts(fname):
    return np.loadtxt(DATA + fname, skiprows=1, usecols=0)


def load_scope(fname):
    a = np.loadtxt(DATA + fname, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def poisson(k, m):
    return np.exp(k * np.log(m) - m - gammaln(k + 1))


def bose_einstein(k, m):
    return (1 / (1 + m)) * (m / (1 + m)) ** k


def thermal_M(k, m, M):
    return np.exp(gammaln(k + M) - gammaln(k + 1) - gammaln(M)
                  + M * np.log(M / (M + m)) + k * np.log(m / (M + m)))


def histogram(k):
    kk = np.arange(0, int(k.max()) + 1)
    n = np.bincount(k.astype(int), minlength=len(kk))
    N = len(k)
    P = n / N
    err = np.sqrt(n) / N
    return kk, P, err, n


def chi2_test(n, expected, n_params=1):
    obs, exp = [], []
    o_acc = e_acc = 0
    for o, e in zip(n, expected):
        o_acc += o
        e_acc += e
        if e_acc >= 5:
            obs.append(o_acc)
            exp.append(e_acc)
            o_acc = e_acc = 0
    if e_acc > 0:
        obs[-1] += o_acc
        exp[-1] += e_acc
    obs, exp = np.array(obs), np.array(exp)
    chi2 = np.sum((obs - exp) ** 2 / exp)
    dof = len(obs) - 1 - n_params
    return chi2, dof, stats.chi2.sf(chi2, dof)


def summary(k):
    m = k.mean()
    v = k.var(ddof=1)
    N = len(k)
    return m, k.std(ddof=1) / np.sqrt(N), v, v * np.sqrt(2 / (N - 1))


def acf(k, max_lag):
    x = k - k.mean()
    return np.array([1.0] + [np.mean(x[:-l] * x[l:]) / x.var() for l in range(1, max_lag + 1)])


def g2_error(k, block=100, n_boot=1000, seed=1):
    rng = np.random.default_rng(seed)
    blocks = k[: len(k) // block * block].reshape(-1, block)
    g = []
    for _ in range(n_boot):
        s = blocks[rng.integers(0, len(blocks), len(blocks))].ravel()
        g.append(np.mean(s * (s - 1)) / s.mean() ** 2)
    return np.std(g)


def fwhm(t, v):
    half = v.max() / 2
    above = np.where(v >= half)[0]
    return t[above[-1]] - t[above[0]]


t1, v1 = load_scope(f"pulse_capture_{N_NAME}.csv")
t2, v2 = load_scope(f"pulses_25ns_open_{N_NAME}.csv")
t3, v3 = load_scope(f"pulses_5us_covered_{N_NAME}.csv")
t4, v4 = load_scope(f"pulses_10us_open_{N_NAME}.csv")

print("Pulse 1: height %.2f V, FWHM %.1f ns" % (v1.max(), fwhm(t1, v1) * 1e9))
print("Pulse 2: height %.2f V, FWHM %.1f ns" % (v2.max(), fwhm(t2, v2) * 1e9))
for name, t, v in [("5us covered", t3, v3), ("10us open", t4, v4)]:
    n_pulses = np.sum((v[1:] > 1) & (v[:-1] <= 1))
    print(f"{name}: {n_pulses} pulses in {(t[-1]-t[0])*1e6:.0f} us")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(t1 * 1e9, v1, label="pulse 1")
ax[0].plot(t2 * 1e9, v2, label="pulse 2")
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Voltage (V)")
ax[0].set_title("Single PMT pulses (25 ns/div)")
ax[0].legend()
ax[1].plot(t3 * 1e6, v3, lw=0.8)
ax[1].set_xlabel("Time (µs)")
ax[1].set_ylabel("Voltage (V)")
ax[1].set_title("Aperture covered (5 µs/div)")
ax[2].plot(t4 * 1e6, v4, lw=0.8)
ax[2].set_xlabel("Time (µs)")
ax[2].set_ylabel("Voltage (V)")
ax[2].set_title("Aperture open (10 µs/div)")
for a in ax:
    a.set_ylim(-0.5, 4)
plt.tight_layout()
plt.savefig(FIG + "fig1_pulses.png", dpi=200)
plt.close()

dark = load_counts(f"dark_count_data_{N_NAME}.txt")
m, dm, v, dv = summary(dark)
chi2_disp = (len(dark) - 1) * v / m
p_disp = stats.chi2.sf(chi2_disp, len(dark) - 1)
print("\nDark counts (T = 1 s, N = %d)" % len(dark))
print("mean = %.1f ± %.1f, variance = %.1f ± %.1f" % (m, dm, v, dv))
print("SD = %.2f, sqrt(mean) = %.2f, var/mean = %.2f" % (np.sqrt(v), np.sqrt(m), v / m))
print("dispersion test chi2 = %.0f on %d dof, p = %.1e" % (chi2_disp, len(dark) - 1, p_disp))
slope, intercept = np.polyfit(np.arange(len(dark)), dark, 1)
print("drift: %.4f counts per s over the run" % slope)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(np.arange(1, len(dark) + 1), dark, ".-", lw=0.6)
ax[0].axhline(m, color="k", ls="--", label=f"mean = {m:.1f}")
ax[0].set_xlabel("Window number (1 s each)")
ax[0].set_ylabel("Counts")
ax[0].set_title("Dark counts vs time")
ax[0].legend()
bins = np.arange(dark.min() - 0.5, dark.max() + 1.5, 2)
ax[1].hist(dark, bins=bins, density=True, alpha=0.6, edgecolor="k", label="data")
kk = np.arange(0, 60)
ax[1].plot(kk, poisson(kk, m), "r-", label=f"Poisson, K̄ = {m:.1f}")
ax[1].set_xlabel("Counts per 1 s")
ax[1].set_ylabel("Probability")
ax[1].set_title("Dark count distribution")
ax[1].set_xlim(5, 60)
ax[1].legend()
plt.tight_layout()
plt.savefig(FIG + "fig2_dark.png", dpi=200)
plt.close()

rows = []


def analyse(kind, i, T, ax):
    k = load_counts(f"{kind}_count_data_{i}_{N_NAME}.txt")
    m, dm, v, dv = summary(k)
    kk, P, err, n = histogram(k)
    N = len(k)
    g2 = np.mean(k * (k - 1)) / m ** 2
    g2_err = g2_error(k)
    M = m ** 2 / (v - m) if v > m else np.inf

    chi_p = chi2_test(n, N * poisson(kk, m))
    chi_b = chi2_test(n, N * bose_einstein(kk, m))

    k_line = np.arange(0, kk.max() + 1)
    ax.bar(kk, P, width=0.8, alpha=0.5, color="C0", label="data")
    ax.errorbar(kk, P, yerr=err, fmt="none", ecolor="k", capsize=2, lw=0.8)
    if kind == "laser":
        ax.plot(k_line, poisson(k_line, m), "ro-", ms=3, label="Poisson")
        title = f"Laser, T = {T} ms, K̄ = {m:.2f}"
    else:
        ax.plot(k_line, bose_einstein(k_line, m), "go-", ms=3, label="Bose–Einstein")
        ax.plot(k_line, poisson(k_line, m), "r--", lw=1, label="Poisson")
        if np.isfinite(M):
            ax.plot(k_line, thermal_M(k_line, m, M), "m:", lw=1.5, label=f"thermal, M = {M:.1f}")
        title = f"Pseudo-thermal {i}, T = {T} ms, K̄ = {m:.2f}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Counts K")
    ax.set_ylabel("P(K)")
    ax.legend(fontsize=8)

    rows.append(dict(run=f"{kind} {i}", T=T, mean=m, dmean=dm, var=v, dvar=dv,
                     fano=v / m, var_be=m + m * m, g2=g2, g2_err=g2_err, M=M,
                     chi_p=chi_p, chi_b=chi_b, k=k))


fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for i, T, ax in zip(range(1, 5), laser_T, axs.flat):
    analyse("laser", i, T, ax)
plt.tight_layout()
plt.savefig(FIG + "fig3_laser.png", dpi=200)
plt.close()

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for i, T, ax in zip(range(1, 5), thermal_T, axs.flat):
    analyse("pseudo_thermal", i, T, ax)
axs.flat[2].set_xlim(-1, 60)
axs.flat[3].set_xlim(-1, 80)
plt.tight_layout()
plt.savefig(FIG + "fig4_thermal.png", dpi=200)
plt.close()

print("\nrun              T(ms)  mean          variance        var/mean  K+K^2   g2(0)  M     chi2/dof Poisson     chi2/dof BE")
for r in rows:
    cp, cb = r["chi_p"], r["chi_b"]
    print(f"{r['run']:16s} {r['T']:4d}  {r['mean']:6.3f}±{r['dmean']:.3f}  {r['var']:7.2f}±{r['dvar']:5.2f}  "
          f"{r['fano']:6.3f}  {r['var_be']:7.2f}  {r['g2']:5.3f}±{r['g2_err']:.3f}  {r['M']:5.1f}  "
          f"{cp[0]:7.1f}/{cp[1]:<3d} p={cp[2]:.2g}   {cb[0]:9.1f}/{cb[1]:<3d} p={cb[2]:.2g}")

laser_rows = [r for r in rows if r["run"].startswith("laser")]
Ts = np.array([r["T"] for r in laser_rows])
ms = np.array([r["mean"] for r in laser_rows])
dms = np.array([r["dmean"] for r in laser_rows])
coef, cov = np.polyfit(Ts, ms, 1, w=1 / dms, cov="unscaled")
print("\nLaser K̄ vs T: slope = %.4f ± %.4f counts/ms, intercept = %.3f ± %.3f"
      % (coef[0], np.sqrt(cov[0, 0]), coef[1], np.sqrt(cov[1, 1])))
print("count rate = %.0f ± %.0f counts/s" % (coef[0] * 1000, np.sqrt(cov[0, 0]) * 1000))
print("dark contribution per 1 ms window = %.3f counts" % (m / 1000))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].errorbar(Ts, ms, yerr=dms, fmt="o", capsize=3, label="data")
tt = np.linspace(0, 11, 50)
ax[0].plot(tt, np.polyval(coef, tt), "r-", label=f"fit: {coef[0]*1000:.0f} counts/s")
ax[0].set_xlabel("Counting time T (ms)")
ax[0].set_ylabel("Mean count K̄")
ax[0].set_title("Laser: mean count vs counting time")
ax[0].legend()

mm = np.logspace(-0.3, 1.5, 100)
ax[1].loglog(mm, mm, "r-", label="Poisson: σ² = K̄")
ax[1].loglog(mm, mm + mm ** 2, "g-", label="Bose–Einstein: σ² = K̄ + K̄²")
for r in rows:
    marker = "o" if r["run"].startswith("laser") else "s"
    color = "C0" if r["run"].startswith("laser") else "C1"
    ax[1].errorbar(r["mean"], r["var"], yerr=r["dvar"], fmt=marker, color=color, capsize=2)
ax[1].plot([], [], "o", color="C0", label="laser")
ax[1].plot([], [], "s", color="C1", label="pseudo-thermal")
ax[1].set_xlabel("Mean count K̄")
ax[1].set_ylabel("Variance σ²")
ax[1].set_title("Variance vs mean")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG + "fig5_mean_variance.png", dpi=200)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t_show = 2000
ax[0].plot(rows[0]["k"][:t_show], lw=0.6, label="laser 1 (K̄ = %.1f)" % rows[0]["mean"])
ax[0].plot(rows[7]["k"][:t_show], lw=0.6, alpha=0.8, label="pseudo-thermal 4 (K̄ = %.1f)" % rows[7]["mean"])
ax[0].set_xlabel("Window number (1 ms each)")
ax[0].set_ylabel("Counts")
ax[0].set_title("First 2 s of counts")
ax[0].legend(fontsize=8)
lags = np.arange(0, 101)
for r in rows:
    if r["T"] == 1:
        ax[1].plot(lags, acf(r["k"], 100), label=r["run"])
ax[1].axhline(np.exp(-1), color="k", ls=":", lw=0.8)
ax[1].set_xlabel("Lag (ms)")
ax[1].set_ylabel("Autocorrelation")
ax[1].set_title("Correlation between counting windows")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG + "fig6_bunching.png", dpi=200)
plt.close()

for r in rows:
    if r["run"].startswith("pseudo"):
        a = acf(r["k"], 200)
        below = np.where(a < np.exp(-1))[0]
        print(f"{r['run']}: autocorrelation at 1 ms lag = {a[1]:.2f}, falls to 1/e at {below[0]} ms")

print("\nCorrecting for correlated windows (pseudo-thermal)")
for r in rows:
    if r["run"].startswith("pseudo"):
        k = r["k"]
        a = acf(k, 300)
        cut = np.where(a < 0)[0][0]
        tau_int = 1 + 2 * a[1:cut].sum()
        kk, P, err, n = histogram(k)
        N = len(k)
        chi_m = chi2_test(n, N * thermal_M(kk, r["mean"], r["M"]), 2)
        cb = r["chi_b"]
        print(f"{r['run']}: integrated correlation time = {tau_int:.1f} windows, N_eff = {N/tau_int:.0f}, "
              f"BE chi2/dof corrected = {cb[0]/tau_int:.1f}/{cb[1]}, "
              f"M-mode chi2/dof corrected = {chi_m[0]/tau_int:.1f}/{chi_m[1]}")
