"""
visualize.py
────────────
Six publication-ready plots for the dynamic pricing RL project.

Plot 1  plot_training_curves          Training dashboard (4 panels)
Plot 2  plot_episode_rollout          Single episode step-level analysis (6 panels)
Plot 3  plot_strategy_comparison      SAC vs constant-price baselines (3 panels)
Plot 4  plot_pricing_decisions        Case studies — what did the agent decide? (4 panels)
Plot 5  plot_demand_landscape         Circadian demand ground truth (3 panels)
Plot 6  plot_acceptance_analysis      Aggregate acceptance diagnostic (3 panels)

All plots saved to results/ directory as PNG files.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from environment import _circadian

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.05)
PAL     = sns.color_palette("tab10")
C_RL    = PAL[0]    # blue  — SAC
C_CONST = PAL[1:]   # rest  — baselines
C_REV   = "#2dc653"
C_PEN   = "#e63946"
C_WARN  = "#f4a261"
C_CHURN = "#9b2226"


def _ema(x, a=0.05):
    s, out = None, []
    for v in x:
        s = v if s is None else (1 - a) * s + a * v
        out.append(s)
    return np.array(out)


def _save(fig, directory, name):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Training dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(log: dict, save_dir="results"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("SAC Training Dashboard — Dynamic Pricing",
                 fontsize=15, fontweight="bold")

    eps = np.arange(1, len(log["episode_reward"]) + 1)
    raw = np.array(log["episode_reward"])

    # (1) Episode reward
    ax = axes[0, 0]
    ax.plot(eps, raw, color=C_RL, alpha=0.15, lw=0.6, label="Raw")
    ax.plot(eps, _ema(raw, 0.03), color=C_RL, lw=2.0, label="EMA")
    if log.get("eval_episodes"):
        ax.scatter(log["eval_episodes"], log["eval_reward"],
                   color="gold", edgecolors="k", s=60, zorder=5, label="Eval")
    ax.set_title("Episode Reward"); ax.set_xlabel("Episode")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # (2) Acceptance
    ax = axes[0, 1]
    if log.get("episode_accept"):
        acc = np.array(log["episode_accept"])
        ax.plot(eps, acc, color=PAL[3], alpha=0.20, lw=0.6)
        ax.plot(eps, _ema(acc, 0.03), color=PAL[3], lw=2.0)
        ax.axhline(0.60, color=C_PEN, ls="--", lw=1.2, alpha=0.8, label="Target 60%")
        if log.get("eval_accept"):
            ax.scatter(log["eval_episodes"], log["eval_accept"],
                       color="gold", edgecolors="k", s=50, zorder=5)
    ax.set_title("Acceptance Rate"); ax.set_xlabel("Episode"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # (3) Critic & Actor loss
    ax = axes[1, 0]
    if log.get("critic_loss"):
        cl  = np.array(log["critic_loss"])
        alx = np.linspace(1, len(eps), len(cl))
        ax.semilogy(alx, cl, color=PAL[1], alpha=0.3, lw=0.6, label="Critic")
        ax.semilogy(alx, _ema(cl, 0.05), color=PAL[1], lw=1.8, label="Critic EMA")
    if log.get("actor_loss"):
        al  = np.array(log["actor_loss"])
        alx = np.linspace(1, len(eps), len(al))
        ax.semilogy(alx, np.abs(al), color=PAL[2], alpha=0.3, lw=0.6, label="Actor")
        ax.semilogy(alx, _ema(np.abs(al), 0.05), color=PAL[2], lw=1.8, label="Actor EMA")
    ax.set_title("Losses (log scale)"); ax.set_xlabel("Episode")
    ax.legend(fontsize=7)

    # (4) Alpha
    ax = axes[1, 1]
    if log.get("alpha"):
        alp = np.array(log["alpha"])
        alx = np.linspace(1, len(eps), len(alp))
        ax.plot(alx, alp, color=PAL[3], lw=1.8)
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_title("Entropy Temperature α (auto-tuned)")
    ax.set_xlabel("Episode"); ax.set_ylabel("α")

    plt.tight_layout()
    return _save(fig, save_dir, "01_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Single-episode rollout
# ─────────────────────────────────────────────────────────────────────────────
def plot_episode_rollout(records: list, save_dir="results"):
    steps   = np.array([r["step"]       for r in records])
    prices  = np.array([r["price"]      for r in records])
    accepts = np.array([r["acceptance"] for r in records])
    demands = np.array([r["demand"]     for r in records])
    supplies= np.array([r["supply"]     for r in records])
    revenues= np.array([r["revenue"]    for r in records])
    served  = np.array([r["served"]     for r in records])
    idle    = np.array([r["idle"]       for r in records])
    unmet   = np.array([r["unmet"]      for r in records])
    churns  = np.array([r["churn"]      for r in records])
    rewards = np.array([r["reward"]     for r in records])
    hours   = np.array([r["hour"]       for r in records])
    weather = [r["weather"] for r in records]
    events  = [r["event"]   for r in records]

    w_cmap   = {"Clear": "#4CAF50", "Rain": "#2196F3", "Storm": "#9C27B0"}
    w_colors = [w_cmap.get(w, "grey") for w in weather]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("SAC Agent — Single Episode Rollout",
                 fontsize=15, fontweight="bold")

    # (1) Demand vs Supply
    ax = axes[0, 0]
    ax.plot(steps, demands,  color=C_RL,   lw=2.0, label="Demand")
    ax.plot(steps, supplies, color=PAL[1], lw=2.0, ls="--", label="Supply")
    ax.fill_between(steps, supplies, demands,
                    where=demands > supplies, alpha=0.18, color=C_RL, label="Unmet")
    ax.set_title("Demand vs Supply"); ax.set_xlabel("Step"); ax.set_ylabel("Rides")
    ax.legend(fontsize=8)

    # (2) Price (coloured by weather)
    ax = axes[0, 1]
    for i in range(len(steps) - 1):
        ax.plot(steps[i:i+2], prices[i:i+2], color=w_colors[i], lw=2.2)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ev_steps = [i for i, e in enumerate(events) if e == "MajorEvent"]
    if ev_steps:
        ax.vlines(ev_steps, prices.min(), prices.max(),
                  colors="orange", lw=1.0, ls="--", alpha=0.7, label="Event")
    ax.set_title("Price Multiplier (colour = weather)")
    ax.set_xlabel("Step"); ax.set_ylabel("Multiplier"); ax.set_ylim(0.6, 3.2)
    patches = [mpatches.Patch(color=c, label=w) for w, c in w_cmap.items()]
    ax.legend(handles=patches, fontsize=8)

    # (3) Acceptance + churn
    ax = axes[1, 0]
    ax.plot(steps, accepts, color=PAL[3], lw=2.0, label="Acceptance")
    ax.fill_between(steps, 0, accepts, alpha=0.12, color=PAL[3])
    ax.axhline(0.60, color=C_PEN, ls="--", lw=1.3, alpha=0.9, label="Target 60%")
    ax2 = ax.twinx()
    ax2.plot(steps, churns, color=C_CHURN, lw=1.5, ls="-.", alpha=0.8, label="Churn memory")
    ax2.set_ylabel("Churn Memory", color=C_CHURN)
    ax2.tick_params(axis="y", labelcolor=C_CHURN); ax2.set_ylim(0, 1.0)
    ax.set_title("Acceptance Rate & Churn Memory")
    ax.set_xlabel("Step"); ax.set_ylabel("Acceptance"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, loc="lower left")

    # (4) Revenue vs served/idle/unmet
    ax = axes[1, 1]
    ax.bar(steps, served,  color=C_REV,   alpha=0.80, label="Served")
    ax.bar(steps, unmet,   color=C_PEN,   alpha=0.70, label="Unmet", bottom=served)
    ax.bar(steps, idle,    color="#457b9d",alpha=0.60, label="Idle",  bottom=served+unmet)
    ax_r = ax.twinx()
    ax_r.plot(steps, revenues, color="gold", lw=1.8, label="Revenue $")
    ax_r.set_ylabel("Revenue", color="gold")
    ax_r.tick_params(axis="y", labelcolor="gold")
    ax.set_title("Rides: Served / Unmet / Idle vs Revenue")
    ax.set_xlabel("Step"); ax.set_ylabel("Rides")
    ax.legend(fontsize=7, loc="upper left")

    # (5) Price vs hour, overlaid on demand curve
    ax = axes[2, 0]
    hs = np.linspace(0, 24, 200)
    cf = np.array([_circadian(h) for h in hs])
    ax3 = ax.twinx()
    ax3.fill_between(hs, 0, cf, alpha=0.12, color="orange", label="Demand curve")
    ax3.set_ylabel("Demand Factor", color="orange")
    ax3.tick_params(axis="y", labelcolor="orange")
    sc = ax.scatter(hours, prices, c=accepts, cmap="RdYlGn",
                    vmin=0.2, vmax=1.0, s=30, alpha=0.7, zorder=3)
    plt.colorbar(sc, ax=ax, label="Acceptance")
    ax.set_title("Price by Hour of Day\n(colour = acceptance, bg = demand curve)")
    ax.set_xlabel("Hour"); ax.set_ylabel("Price Multiplier")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 3)])

    # (6) Net reward per step
    ax = axes[2, 1]
    pos = np.where(rewards >= 0, rewards, 0)
    neg = np.where(rewards < 0,  rewards, 0)
    ax.bar(steps, pos, color=C_REV, alpha=0.75, label="Positive reward")
    ax.bar(steps, neg, color=C_PEN, alpha=0.70, label="Negative reward")
    ax.plot(steps, np.cumsum(rewards) / 1000, color="gold",
            lw=1.6, ls="--", label="Cumulative /1000")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_title("Per-Step Reward"); ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.legend(fontsize=7)

    plt.tight_layout()
    return _save(fig, save_dir, "02_episode_rollout.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Strategy comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_strategy_comparison(results: dict, save_dir="results"):
    labels = list(results.keys())
    colors = {l: (C_RL if "SAC" in l else C_CONST[i % len(C_CONST)])
              for i, l in enumerate(labels)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("SAC vs Constant-Price Baselines\n(Common exogenous seeds per episode)",
                 fontsize=14, fontweight="bold")

    # (1) Cumulative reward
    ax = axes[0]
    for lbl, arr in results.items():
        ax.plot(np.cumsum(arr), label=lbl, color=colors[lbl],
                lw=2.2 if "SAC" in lbl else 1.2,
                ls="-" if "SAC" in lbl else "--")
    ax.set_title("Cumulative Reward"); ax.set_xlabel("Episode")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _:
            f"{x/1e6:.1f}M" if abs(x) >= 1e6 else f"{x:,.0f}"))

    # (2) Violin plot
    ax = axes[1]
    parts = ax.violinplot([results[l] for l in labels], positions=range(len(labels)),
                          showmeans=True, showmedians=True)
    for pc, lbl in zip(parts["bodies"], labels):
        pc.set_facecolor(colors[lbl]); pc.set_alpha(0.75)
    parts["cmeans"].set_color("white"); parts["cmedians"].set_color("gold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("Reward Distribution"); ax.set_ylabel("Episode Reward")

    # (3) Mean bar + % vs SAC
    ax = axes[2]
    means   = [float(np.mean(results[l])) for l in labels]
    rl_mean = means[0]
    bars    = ax.bar(range(len(labels)), means,
                     color=[colors[l] for l in labels],
                     edgecolor="white", lw=0.8, width=0.6)
    for bar, lbl, m in zip(bars, labels, means):
        pct = (m / rl_mean - 1.0) * 100 if "SAC" not in lbl else 0.0
        txt = "SAC" if "SAC" in lbl else f"{pct:+.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(max(means)) * 0.01,
                txt, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("Mean Reward (% vs SAC)"); ax.set_ylabel("Mean Reward")

    plt.tight_layout()
    return _save(fig, save_dir, "03_strategy_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pricing decision case studies
# ─────────────────────────────────────────────────────────────────────────────
def plot_pricing_decisions(all_records: list, save_dir="results"):
    """all_records: flat list of step dicts from multiple episodes."""
    prices  = np.array([r["price"]      for r in all_records])
    accepts = np.array([r["acceptance"] for r in all_records])
    revenues= np.array([r["revenue"]    for r in all_records])
    hours   = np.array([r["hour"]       for r in all_records])
    demands = np.array([r["demand"]     for r in all_records])
    supplies= np.array([r["supply"]     for r in all_records])
    ds      = demands / np.maximum(supplies, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("SAC Pricing Case Studies — What Did the Agent Decide?",
                 fontsize=14, fontweight="bold")

    # (1) D/S vs price scatter
    ax = axes[0, 0]
    sizes = np.clip(revenues / 50, 8, 200)
    sc = ax.scatter(ds, prices, c=accepts, cmap="RdYlGn",
                    vmin=0.2, vmax=1.0, s=sizes, alpha=0.55, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Acceptance")
    ax.axvline(1.0, color="gray", ls="--", alpha=0.4)
    ax.axhline(1.0, color="lightblue", ls=":", alpha=0.6)
    for (lo, hi, xpos) in [(0, 0.75, 0.4), (0.75, 1.5, 1.1), (1.5, 99, 2.2)]:
        mask = (ds >= lo) & (ds < hi)
        if mask.any():
            mp = np.median(prices[mask])
            ma = np.median(accepts[mask])
            ax.annotate(f"Med {mp:.2f}×\nacc {ma:.0%}",
                        xy=(xpos, mp), xytext=(xpos + 0.05, mp + 0.25),
                        fontsize=7, color="white", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax.set_xlabel("Demand / Supply"); ax.set_ylabel("Price Multiplier")
    ax.set_title("Price by D/S Ratio\n(size=revenue, colour=acceptance)")

    # (2) Price & acceptance by time-of-day period
    ax = axes[0, 1]
    periods = ["Morning\n(6–10)", "Midday\n(10–16)", "Evening\n(16–21)", "Night\n(21–6)"]
    def period_mask(h):
        h = h % 24
        if   6 <= h < 10:  return 0
        elif 10 <= h < 16: return 1
        elif 16 <= h < 21: return 2
        else:               return 3
    pidx = np.array([period_mask(h) for h in hours])

    avg_p = [np.mean(prices[pidx == i])  for i in range(4)]
    avg_a = [np.mean(accepts[pidx == i]) for i in range(4)]

    x    = np.arange(4)
    bars = ax.bar(x, avg_p, color="#e07b39", alpha=0.85, edgecolor="white")
    ax2  = ax.twinx()
    ax2.plot(x, avg_a, "o-", color=C_REV, lw=2.2, ms=9, label="Acceptance")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(periods, fontsize=9)
    ax.set_ylabel("Avg Price Multiplier")
    ax2.set_ylabel("Avg Acceptance", color=C_REV)
    ax2.set_ylim(0, 1.05); ax2.tick_params(axis="y", labelcolor=C_REV)
    ax.set_title("Price & Acceptance by Time of Day")

    # (3) Per-hour pricing bar (bar colour = demand intensity)
    ax = axes[1, 0]
    h_bins = np.zeros((24, 2))
    for i, (h, p, a) in enumerate(zip(hours, prices, accepts)):
        hi = int(h) % 24
        h_bins[hi, 0] += p; h_bins[hi, 1] += 1
    with np.errstate(invalid="ignore"):
        avg_ph = np.where(h_bins[:, 1] > 0, h_bins[:, 0] / h_bins[:, 1], np.nan)
    valid = ~np.isnan(avg_ph)
    circ  = np.array([_circadian(h + 0.5) for h in range(24)])
    norm  = (circ - circ.min()) / (circ.max() - circ.min())
    bcolors = [plt.cm.YlOrRd(0.25 + 0.65 * norm[h]) for h in range(24)]

    ax.bar(np.arange(24)[valid], avg_ph[valid],
           color=[bcolors[h] for h in np.arange(24)[valid]], alpha=0.85)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)], rotation=45, fontsize=7)
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg Price Multiplier")
    ax.set_title("Avg Price by Hour  (colour = demand intensity)")

    # (4) Sample decision table
    ax = axes[1, 1]
    ax.axis("off")

    high_idx   = np.argsort(-ds)[:5]
    low_idx    = np.argsort(ds)[:4]
    event_idx  = [i for i, r in enumerate(all_records) if r.get("event") == "MajorEvent"][:3]
    sample_idx = list(dict.fromkeys(
        [int(i) for i in list(high_idx) + list(low_idx) + event_idx]))[:14]

    rows = []
    for i in sample_idx:
        r  = all_records[i]
        h  = r["hour"]
        p  = "Morning" if 6 <= h < 10 else ("Midday" if 10 <= h < 16
             else ("Evening" if 16 <= h < 21 else "Night"))
        ev = " EVENT" if r.get("event") == "MajorEvent" else ""
        d  = r.get("demand", 50) / max(r.get("supply", 40), 1)
        tier = ("HIGH" if d > 1.5 else ("LOW" if d < 0.75 else "MED"))
        rows.append([
            f"{h:.0f}:00", p, r.get("weather","?")[:5],
            f"{tier}{ev}", f"{d:.2f}",
            f"{r['price']:.2f}×", f"{r['acceptance']*100:.0f}%",
            f"${r['revenue']:.0f}",
        ])

    cols = ["Hour", "Period", "Wthr", "Demand", "D/S", "Price", "Accept", "Rev"]
    tbl  = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.55)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2b2d42"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for ri, i in enumerate(sample_idx[:len(rows)]):
        d = all_records[i].get("demand", 50) / max(all_records[i].get("supply", 40), 1)
        bg = "#ffe0e0" if d > 1.5 else ("#e0ffe0" if d < 0.75 else "#fff8e0")
        for j in range(len(cols)):
            tbl[ri + 1, j].set_facecolor(bg)
    ax.set_title("Sample Decisions  (High / Medium / Low / Event)",
                 fontweight="bold", pad=14)

    plt.tight_layout()
    return _save(fig, save_dir, "04_pricing_decisions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Demand landscape
# ─────────────────────────────────────────────────────────────────────────────
def plot_demand_landscape(save_dir="results"):
    hours = np.linspace(0, 24, 200)
    w_map = {"Clear": (1.00, "#4CAF50"), "Rain": (1.30, "#2196F3"),
             "Storm": (1.75, "#9C27B0")}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Circadian Demand Landscape — Why Prices Change by Hour",
                 fontsize=14, fontweight="bold")

    # (1) Heatmap: hour × weather
    ax   = axes[0]
    base = np.array([_circadian(h) for h in np.linspace(0, 24, 48)])
    heat = np.array([[base[i] * wf for i, _ in enumerate(np.linspace(0, 24, 48))]
                     for _, (wf, _) in w_map.items()])
    im   = ax.imshow(heat, aspect="auto", cmap="YlOrRd",
                     extent=[0, 24, -0.5, 2.5], origin="lower")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(list(w_map.keys()), fontsize=10)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 3)], rotation=30)
    ax.set_title("Demand Factor: Hour × Weather")
    plt.colorbar(im, ax=ax, label="Demand Factor")
    for yi, (wlbl, (wf, _)) in enumerate(w_map.items()):
        curve = base * wf
        ph    = float(np.linspace(0, 24, 48)[np.argmax(curve)])
        ax.annotate(f"Peak\n{ph:.0f}:00", xy=(ph, yi),
                    fontsize=7.5, color="white", fontweight="bold",
                    ha="center", va="center")

    # (2) Smooth curves per weather
    ax = axes[1]
    ax.axvspan(7, 9, alpha=0.09, color="orange")
    ax.axvspan(17, 21, alpha=0.09, color="red")
    ax.text(8, 0.05, "Morning\nRush", ha="center", fontsize=7.5, color="darkorange")
    ax.text(19, 0.05, "Evening\nRush", ha="center", fontsize=7.5, color="red")
    for wlbl, (wf, wc) in w_map.items():
        curve = np.array([_circadian(h) * wf for h in hours])
        ax.plot(hours, curve, color=wc, lw=2.2, label=f"{wlbl} (×{wf})")
    ax.set_xlabel("Hour"); ax.set_ylabel("Demand Multiplier")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 3)], rotation=30)
    ax.set_title("Demand Curves by Weather")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    # (3) Illustrative heuristic (not the learned policy)
    ax = axes[2]
    for wlbl, (wf, wc) in w_map.items():
        curve  = np.array([_circadian(h) * wf for h in hours])
        t_price = 1.0 + np.clip((curve - 0.3) / 1.5, 0.0, 1.4) * 0.85
        ax.plot(hours, t_price, color=wc, lw=2.2, label=wlbl)
        ax.fill_between(hours, 1.0, t_price, alpha=0.08, color=wc)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.6, label="No surge")
    ax.axhline(2.0, color=C_PEN,  ls="--", alpha=0.5, label="High-risk (~2×)")
    ax.set_xlabel("Hour"); ax.set_ylabel("Suggested Price Multiplier")
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 3)], rotation=30)
    ax.set_title("Illustrative Demand-Proportional Heuristic\n(Not the learned policy)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return _save(fig, save_dir, "05_demand_landscape.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Acceptance fairness analysis
# ─────────────────────────────────────────────────────────────────────────────
def plot_acceptance_analysis(all_records: list, save_dir="results"):
    prices  = np.array([r["price"]      for r in all_records])
    accepts = np.array([r["acceptance"] for r in all_records])
    ds      = np.array([r["demand"]     for r in all_records]) / \
              np.maximum(np.array([r["supply"] for r in all_records]), 1.0)
    churns  = np.array([r["churn"]      for r in all_records])

    mean_a  = float(np.mean(accepts))
    pct_60  = 100 * float(np.mean(accepts >= 0.60))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Aggregate Acceptance Diagnostic — Price/Access Trade-off",
                 fontsize=14, fontweight="bold")

    # (1) Price vs acceptance (colour = D/S)
    ax = axes[0]
    sc = ax.scatter(prices, accepts, c=ds, cmap="RdYlGn_r",
                    vmin=0.5, vmax=2.5, s=25, alpha=0.40, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="D/S Ratio")
    ax.axhline(0.60, color=C_PEN,  ls="--", lw=1.8, alpha=0.9, label="Target 60%")
    ax.axhline(mean_a, color="gold", lw=1.8, label=f"Agent mean {mean_a:.0%}")
    ax.axhspan(0, 0.60,  alpha=0.07, color=C_PEN)
    ax.set_xlabel("Price Multiplier"); ax.set_ylabel("Acceptance Rate")
    ax.set_title("Price vs Acceptance\n(colour = D/S ratio)"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="lower left")

    # (2) Acceptance histogram
    ax = axes[1]
    ax.hist(accepts, bins=35, color=C_RL, alpha=0.75, edgecolor="white", lw=0.5)
    ax.axvline(mean_a, color="gold", lw=2.2, label=f"Mean {mean_a:.2f}")
    ax.axvline(0.60,   color=C_PEN,  ls="--", lw=1.5, label="Target 0.60")
    ax.text(0.04, 0.93, f"{pct_60:.1f}% meeting target",
            transform=ax.transAxes, fontsize=9.5, color=C_REV, fontweight="bold")
    ax.set_xlabel("Acceptance Rate"); ax.set_ylabel("Frequency")
    ax.set_title("Acceptance Distribution"); ax.legend(fontsize=8)

    # (3) Price vs churn memory
    ax = axes[2]
    sc2 = ax.scatter(prices, churns, c=accepts, cmap="RdYlGn",
                     vmin=0.2, vmax=1.0, s=28, alpha=0.45, edgecolors="none")
    plt.colorbar(sc2, ax=ax, label="Acceptance")
    ax.axhline(0.05, color=C_REV, ls="--", lw=1.2, alpha=0.7,
               label="Low churn zone")
    ax.axhline(0.20, color=C_PEN, ls="--", lw=1.4, alpha=0.8,
               label="High churn danger")
    ax.set_xlabel("Price Multiplier"); ax.set_ylabel("Churn Memory")
    ax.set_title("Price vs Churn Memory\n(aggregate diagnostic)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    return _save(fig, save_dir, "06_acceptance_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all plots
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_plots(log_path, env, agent, results, save_dir="results"):
    from evaluate import collect_rollout

    print("\n  Generating all 6 plots …")

    with open(log_path) as f:
        log = json.load(f)
    plot_training_curves(log, save_dir)

    records = collect_rollout(env, agent, seed=0)
    plot_episode_rollout(records, save_dir)

    plot_strategy_comparison(results, save_dir)

    all_recs = []
    for ep in range(6):
        recs = collect_rollout(env, agent, seed=500 + ep * 23)
        all_recs.extend(recs)
    plot_pricing_decisions(all_recs, save_dir)

    plot_demand_landscape(save_dir)
    plot_acceptance_analysis(all_recs, save_dir)

    print(f"  All plots saved to {save_dir}/")
