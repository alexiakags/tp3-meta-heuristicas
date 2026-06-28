"""
TP3 - Meta-heurísticas CCF-480
Algoritmo: PSO (Particle Swarm Optimization) - Inteligência Coletiva
Problemas: TP1 Problema 1 (config a) e TP2 Problema 1
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Arquivos serão salvos em:\n{OUTPUT_DIR}\n")

np.random.seed(42)  # Reprodutibilidade

# FUNÇÕES OBJETIVO

def f_tp1(x):
    """f(x) = 100*sqrt(|x2 - 0.01*x1^2|) + 0.01*|x1 + 10|"""
    x1, x2 = x[0], x[1]
    return 100 * np.sqrt(abs(x2 - 0.01 * x1**2)) + 0.01 * abs(x1 + 10)


def f_tp2_penalidade(x, penalty=1e6):
    """
    MIN x1^0.6 + x2^0.6 - 6*x1 - 4*u1 + 3*u2
    s.t.  x2 - 3*x1 - 3*u1 = 0
          x1 + 2*u1 <= 4,  x2 + 2*u2 <= 4
          x1 <= 3,  u2 <= 1,  x1,x2,u1,u2 >= 0
    Tratamento: Penalidade Estática
    """
    x1, x2, u1, u2 = x[0], x[1], x[2], x[3]
    obj = x1**0.6 + x2**0.6 - 6*x1 - 4*u1 + 3*u2
    eps = 0.0001
    h1 = abs(x2 - 3*x1 - 3*u1) - eps
    g1 = x1 + 2*u1 - 4
    g2 = x2 + 2*u2 - 4
    g3 = x1 - 3
    g4 = u2 - 1
    g5, g6, g7, g8 = -x1, -x2, -u1, -u2
    violation = (max(0, h1)**2 + max(0, g1)**2 + max(0, g2)**2 +
                 max(0, g3)**2 + max(0, g4)**2 + max(0, g5)**2 +
                 max(0, g6)**2 + max(0, g7)**2 + max(0, g8)**2)
    return obj + penalty * violation


# PSO

def pso(func, bounds, n_particles=40, max_iter=500, w=0.7, c1=1.5, c2=1.5):
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    pos = lb + np.random.rand(n_particles, dim) * (ub - lb)
    vel = np.zeros((n_particles, dim))
    v_max = (ub - lb) * 0.2
    pbest_pos = pos.copy()
    pbest_val = np.array([func(p) for p in pos])
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    for _ in range(max_iter):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        vel = (w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, lb, ub)
        vals = np.array([func(p) for p in pos])
        improved = vals < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = vals[improved]
        best_idx = np.argmin(pbest_val)
        if pbest_val[best_idx] < gbest_val:
            gbest_val = pbest_val[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()
    return gbest_pos, gbest_val


def run_experiment(func, bounds, n_runs=30, label="", **pso_kwargs):
    print(f"\n{'='*55}\n  Executando: {label}\n{'='*55}")
    results, best_sol, best_val = [], None, np.inf
    for i in range(n_runs):
        pos, val = pso(func, bounds, **pso_kwargs)
        results.append(val)
        if val < best_val:
            best_val = val
            best_sol = pos.copy()
        print(f"  Run {i+1:02d}: f = {val:.6f}")
    results = np.array(results)
    stats = dict(label=label, min=np.min(results), max=np.max(results),
                 mean=np.mean(results), std=np.std(results),
                 all=results, best_sol=best_sol, best_val=best_val)
    print(f"\n  Mínimo : {stats['min']:.6f}")
    print(f"  Máximo : {stats['max']:.6f}")
    print(f"  Média  : {stats['mean']:.6f}")
    print(f"  Desvio : {stats['std']:.6f}")
    print(f"  Melhor x: {best_sol}")
    return stats


# ─────────────────────────────────────────────
# EXECUÇÕES PSO
# ─────────────────────────────────────────────

bounds_tp1a = [(-15, -5), (-3, 3)]
stats_pso_tp1 = run_experiment(
    f_tp1, bounds_tp1a, n_runs=30, label="PSO - TP1 Prob.1 config a",
    n_particles=50, max_iter=1000, w=0.7, c1=1.5, c2=1.5)

bounds_tp2 = [(0, 3), (0, 4), (0, 2), (0, 1)]
stats_pso_tp2 = run_experiment(
    f_tp2_penalidade, bounds_tp2, n_runs=30, label="PSO - TP2 Prob.1 Penalidade Estática",
    n_particles=80, max_iter=2000, w=0.729, c1=1.494, c2=1.494)


# DADOS REAIS DOS TPs ANTERIORES

# TP1 - Busca Tabu, Problema 1, config a) — 30 execuções reais
bt_tp1 = np.array([
    0.2115, 0.2200, 0.2130, 0.1483, 0.0576, 0.2609, 0.1313, 0.1430,
    0.0570, 0.1398, 0.2345, 0.1461, 0.2217, 0.1792, 0.1128, 0.0620,
    0.1645, 0.1017, 0.2776, 0.2150, 0.0922, 0.1342, 0.1670, 0.2900,
    0.0619, 0.1138, 0.1617, 0.0618, 0.0919, 0.1770
])

# TP2 - ED Config A (Penalidade Estática), Problema 1 — 30 execuções reais
ed_tp2 = np.array([
    0.0000, -4.5144, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, -3.0361, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
])

print("\n--- Estatísticas TP1 (BT real) ---")
print(f"  Mín={bt_tp1.min():.4f} | Máx={bt_tp1.max():.4f} | Méd={bt_tp1.mean():.4f} | DP={bt_tp1.std():.4f}")
print("\n--- Estatísticas TP2 (ED Config A real) ---")
print(f"  Mín={ed_tp2.min():.4f} | Máx={ed_tp2.max():.4f} | Méd={ed_tp2.mean():.4f} | DP={ed_tp2.std():.4f}")


# BOXPLOT 1: TP1 - PSO vs BT

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f9fa')

data_tp1   = [stats_pso_tp1['all'], bt_tp1]
labels_tp1 = ['PSO\n(Cooperativo)', 'Busca Tabu\n(TP1)']
colors_tp1 = ['#2196F3', '#FF9800']

bp1 = ax.boxplot(data_tp1, patch_artist=True, widths=0.5,
                 medianprops=dict(color='black', linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=2),
                 flierprops=dict(marker='o', markersize=6, linestyle='none'))

for patch, color in zip(bp1['boxes'], colors_tp1):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for i, color in enumerate(colors_tp1):
    bp1['whiskers'][2*i].set_color(color)
    bp1['whiskers'][2*i+1].set_color(color)
    bp1['caps'][2*i].set_color(color)
    bp1['caps'][2*i+1].set_color(color)
    bp1['fliers'][i].set_markerfacecolor(color)
    bp1['fliers'][i].set_markeredgecolor(color)

ax.set_xticks([1, 2])
ax.set_xticklabels(labels_tp1, fontsize=12)
ax.set_ylabel('Valor da Função Objetivo f(x)', fontsize=12)
ax.set_title('TP1 – Problema 1, Config. a)\nf(x) = 100√|x₂−0.01x₁²| + 0.01|x₁+10|\n30 Execuções Independentes',
             fontsize=13, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

p1 = mpatches.Patch(color='#2196F3', alpha=0.75,
    label=f"PSO   mín={stats_pso_tp1['min']:.4f}  méd={stats_pso_tp1['mean']:.4f}  dp={stats_pso_tp1['std']:.4f}")
p2 = mpatches.Patch(color='#FF9800', alpha=0.75,
    label=f"BT     mín={bt_tp1.min():.4f}  méd={bt_tp1.mean():.4f}  dp={bt_tp1.std():.4f}")
ax.legend(handles=[p1, p2], fontsize=9.5, loc='upper right', framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "boxplot_tp1_tp3.png"),
    dpi=150,
    bbox_inches="tight"
)
print("\nBoxplot TP1 salvo!")
plt.close()


# BOXPLOT 2: TP2 - PSO vs ED Config A

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f9fa')

data_tp2   = [stats_pso_tp2['all'], ed_tp2]
labels_tp2 = ['PSO\n(Cooperativo)', 'Evol. Diferencial\n(TP2 Config A)']
colors_tp2 = ['#2196F3', '#E91E63']

bp2 = ax.boxplot(data_tp2, patch_artist=True, widths=0.5,
                 medianprops=dict(color='black', linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=2),
                 flierprops=dict(marker='o', markersize=6, linestyle='none'))

for patch, color in zip(bp2['boxes'], colors_tp2):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for i, color in enumerate(colors_tp2):
    bp2['whiskers'][2*i].set_color(color)
    bp2['whiskers'][2*i+1].set_color(color)
    bp2['caps'][2*i].set_color(color)
    bp2['caps'][2*i+1].set_color(color)
    bp2['fliers'][i].set_markerfacecolor(color)
    bp2['fliers'][i].set_markeredgecolor(color)

ax.set_xticks([1, 2])
ax.set_xticklabels(labels_tp2, fontsize=12)
ax.set_ylabel('Valor da Função Objetivo f(x)', fontsize=12)
ax.set_title('TP2 – Problema 1\nMIN x₁⁰·⁶ + x₂⁰·⁶ − 6x₁ − 4u₁ + 3u₂\n30 Execuções Independentes',
             fontsize=13, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

p3 = mpatches.Patch(color='#2196F3', alpha=0.75,
    label=f"PSO   mín={stats_pso_tp2['min']:.4f}  méd={stats_pso_tp2['mean']:.4f}  dp={stats_pso_tp2['std']:.4f}")
p4 = mpatches.Patch(color='#E91E63', alpha=0.75,
    label=f"ED     mín={ed_tp2.min():.4f}  méd={ed_tp2.mean():.4f}  dp={ed_tp2.std():.4f}")
ax.legend(handles=[p3, p4], fontsize=9.5, loc='upper right', framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "boxplot_tp2_tp3.png"),
    dpi=150,
    bbox_inches="tight"
)
print("Boxplot TP2 salvo!")
plt.close()


# RESUMO FINAL

print("\n" + "="*60)
print("RESUMO FINAL")
print("="*60)
print(f"\nTP1 Problema 1 (config a) – PSO:")
print(f"  Melhor solução: x1={stats_pso_tp1['best_sol'][0]:.6f}, x2={stats_pso_tp1['best_sol'][1]:.6f}")
print(f"  Melhor f(x)   : {stats_pso_tp1['best_val']:.6f}")
print(f"  Mín={stats_pso_tp1['min']:.4f} | Máx={stats_pso_tp1['max']:.4f} | Méd={stats_pso_tp1['mean']:.4f} | DP={stats_pso_tp1['std']:.4f}")

x1, x2, u1, u2 = stats_pso_tp2['best_sol']
print(f"\nTP2 Problema 1 (penalidade estática) – PSO:")
print(f"  Melhor solução: x1={x1:.6f}, x2={x2:.6f}, u1={u1:.6f}, u2={u2:.6f}")
print(f"  Melhor f(x)   : {stats_pso_tp2['best_val']:.6f}")
print(f"  Mín={stats_pso_tp2['min']:.4f} | Máx={stats_pso_tp2['max']:.4f} | Méd={stats_pso_tp2['mean']:.4f} | DP={stats_pso_tp2['std']:.4f}")

import json
stats_out = {
    'pso_tp1': { 'min': float(stats_pso_tp1['min']), 'max': float(stats_pso_tp1['max']),
                 'mean': float(stats_pso_tp1['mean']), 'std': float(stats_pso_tp1['std']),
                 'best_sol': [float(v) for v in stats_pso_tp1['best_sol']],
                 'best_val': float(stats_pso_tp1['best_val']) },
    'pso_tp2': { 'min': float(stats_pso_tp2['min']), 'max': float(stats_pso_tp2['max']),
                 'mean': float(stats_pso_tp2['mean']), 'std': float(stats_pso_tp2['std']),
                 'best_sol': [float(v) for v in stats_pso_tp2['best_sol']],
                 'best_val': float(stats_pso_tp2['best_val']) },
    'bt_tp1':  { 'min': float(bt_tp1.min()), 'max': float(bt_tp1.max()),
                 'mean': float(bt_tp1.mean()), 'std': float(bt_tp1.std()) },
    'ed_tp2':  { 'min': float(ed_tp2.min()), 'max': float(ed_tp2.max()),
                 'mean': float(ed_tp2.mean()), 'std': float(ed_tp2.std()) },
}
with open(os.path.join(OUTPUT_DIR, "stats_tp3.json"), "w") as f:
    json.dump(stats_out, f, indent=2)
print("\nStats salvas!")
