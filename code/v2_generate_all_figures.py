"""
v2_generate_all_figures.py — Complete figure generation for revised paper (v2).
======================================================================
All fixes applied:
  Fig1: Only label top models + remove inference gap bubble
  Fig3: Add A/B/C labels + add value labels to blue (A) line
  Fig4: Add A/B labels + annotate all confusion matrix cells
  Fig5: Add A/B/C labels + remove Type legend text, use clean legend
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, glob, warnings
from sklearn.metrics import confusion_matrix as sk_cm
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

CSV_PATH = r"D:\mWork\工作\小论文v2\v2_final_merged_all.csv"
RESULTS_DIR = r"D:\mWork\工作\小论文v2\code\results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'v2_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

FAMILY_COLORS = {
    'Alibaba (Qwen3)': '#1f77b4', 'Zhipu (GLM-4)': '#2ca02c',
    'Google (gemma/medgemma)': '#ff7f0e', 'OpenAI (gpt-oss)': '#d62728',
    'Tencent (Hunyuan)': '#9467bd', 'Meta (Llama)': '#8c564b',
    'Moonshot (Moonlight)': '#e377c2', 'IBM (granite)': '#7f7f7f',
}

def get_family(name):
    if 'Qwen3' in name: return 'Alibaba (Qwen3)'
    if 'GLM-4' in name: return 'Zhipu (GLM-4)'
    if 'gemma' in name.lower() or 'medgemma' in name.lower(): return 'Google (gemma/medgemma)'
    if 'gpt-oss' in name: return 'OpenAI (gpt-oss)'
    if 'Hunyuan' in name: return 'Tencent (Hunyuan)'
    if 'Llama' in name: return 'Meta (Llama)'
    if 'Moonlight' in name: return 'Moonshot (Moonlight)'
    if 'granite' in name: return 'IBM (granite)'
    return 'Other'

def short_name(full):
    m = {
        'DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf': 'DS-R1-0528-8B',
        'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf': 'DS-R1-7B',
        'DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf': 'DS-R1-14B',
        'DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf': 'DS-R1-32B',
        'GLM-4-9B-0414-Q4_K_M.gguf': 'GLM-4-9B',
        'GLM-4-32B-0414-Q4_K_M.gguf': 'GLM-4-32B',
        'Qwen3-4B-Instruct-2507-Q4_K_M.gguf': 'Q3-4B-Inst',
        'Qwen3-4B-Q4_K_M.gguf': 'Q3-4B',
        'Qwen3-4B-Thinking-2507-Q4_K_M.gguf': 'Q3-4B-Think',
        'Qwen3-8B-Q4_K_M.gguf': 'Q3-8B',
        'Qwen3-14B-Q4_K_M.gguf': 'Q3-14B',
        'Qwen3-32B-Q4_K_M.gguf': 'Q3-32B',
        'Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf': 'Q3-30B-MoE',
        'Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf': 'Q3-30B-MoE-T',
        'Moonlight-16B-A3B-Instruct-Q4_K_M.gguf': 'Moonlight-16B',
        'gemma-3-4b-it-Q4_K_M.gguf': 'Gemma-4B',
        'gemma-3-12b-it-Q4_K_M.gguf': 'Gemma-12B',
        'gemma-3-27b-it-Q4_K_M.gguf': 'Gemma-27B',
        'medgemma-4b-it-Q4_K_M.gguf': 'MedG-4B',
        'medgemma-27b-it-Q4_K_M.gguf': 'MedG-27B',
        'Llama-3.2-3B-Instruct-Q4_K_M.gguf': 'Llama-3B',
        'Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama-8B',
        'granite-3.3-8b-instruct-Q4_K_M.gguf': 'Granite-8B',
        'tencent_Hunyuan-7B-Instruct-Q4_K_M.gguf': 'Hunyuan-7B',
        'tencent_Hunyuan-4B-Instruct-Q4_K_M.gguf': 'Hunyuan-4B',
    }
    if 'gpt-oss' in full:
        if '(low)' in full: return 'gpt-oss-L'
        if '(medium)' in full: return 'gpt-oss-M'
        if '(high)' in full: return 'gpt-oss-H'
    return m.get(full, full[:20])

def save_figure(fig, name):
    for fmt in ['png', 'tif']:
        path = os.path.join(OUTPUT_DIR, f'{name}.{fmt}')
        fig.savefig(path, format='tiff' if fmt == 'tif' else 'png',
                    pil_kwargs={"compression": "tiff_lzw"} if fmt == 'tif' else {})
    plt.close(fig)

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df['family'] = df['model'].apply(get_family)
df['short'] = df['model'].apply(short_name)
df['is_unreliable'] = df['error_rate_mean'] > 0.05

# =============================================================================
# FIGURE 1: Scatter — only label reliable models, no bubble
# =============================================================================

def fig1_scatter(df):
    task = 'fibrosis'
    a = df[(df.condition == 'A_extraction') & (df.task == task)].set_index('model')
    b = df[(df.condition == 'B_inference') & (df.task == task)].set_index('model')
    merged = a[['macro_f1_mean', 'macro_f1_std', 'short', 'family', 'is_unreliable']].join(
        b[['macro_f1_mean', 'macro_f1_std']], lsuffix='_A', rsuffix='_B')
    merged['gap'] = merged['macro_f1_mean_A'] - merged['macro_f1_mean_B']

    fig, ax = plt.subplots(figsize=(8, 7))
    lims = [0, 1.02]
    ax.plot(lims, lims, 'k--', alpha=0.2, linewidth=1)

    # Sort: plot unreliable FIRST (bottom layer) so reliable labels sit on top
    reliable = merged[~merged['is_unreliable']]
    unreliable = merged[merged['is_unreliable']]

    # Unreliable: hollow, faint, no labels
    for fam, color in FAMILY_COLORS.items():
        sub = unreliable[unreliable['family'] == fam]
        if len(sub) == 0: continue
        ax.scatter(sub['macro_f1_mean_A'], sub['macro_f1_mean_B'],
                   s=70, facecolors='none', edgecolors=color, linewidths=1.2, alpha=0.35, zorder=1)

    # Reliable: filled, with labels
    for fam, color in FAMILY_COLORS.items():
        sub = reliable[reliable['family'] == fam]
        if len(sub) == 0: continue
        ax.scatter(sub['macro_f1_mean_A'], sub['macro_f1_mean_B'],
                   s=70, color=color, edgecolors='white', linewidths=0.5, alpha=0.85, zorder=2,
                   label=fam if len(sub) > 0 else "")

    # Label ALL reliable models (they won't overlap much with 28 models)
    # Use adjustText-style manual offsets to avoid overlap
    offsets = {}
    for _, r in reliable.iterrows():
        s = r['short']
        x, y = r['macro_f1_mean_A'], r['macro_f1_mean_B']
        # Manual offsets for crowded areas
        if s == 'Q3-14B': ox, oy = 8, -14
        elif s == 'Q3-32B': ox, oy = 8, -14
        elif s == 'Q3-8B': ox, oy = 8, 8
        elif s == 'Q3-4B-Inst': ox, oy = 8, 8
        elif s == 'Q3-4B': ox, oy = -60, 8
        elif s == 'GLM-4-32B': ox, oy = 6, -14
        elif s == 'GLM-4-9B': ox, oy = 8, 8
        elif s == 'DS-R1-32B': ox, oy = -60, 8
        elif s == 'DS-R1-14B': ox, oy = 8, -14
        elif s == 'Gemma-27B': ox, oy = 8, 0
        elif s == 'Gemma-12B': ox, oy = 8, 0
        elif s == 'Gemma-4B': ox, oy = 8, 8
        elif s == 'MedG-27B': ox, oy = -55, 0
        elif s == 'MedG-4B': ox, oy = 8, 8
        elif s == 'Llama-8B': ox, oy = 8, -12
        elif s == 'Llama-3B': ox, oy = 8, 12
        elif s == 'Granite-8B': ox, oy = 8, 8
        elif s == 'Hunyuan-7B': ox, oy = 8, 8
        elif s == 'Hunyuan-4B': ox, oy = 8, 8
        elif s == 'gpt-oss-M': ox, oy = 8, 0
        elif s == 'gpt-oss-L': ox, oy = 8, 8
        elif s == 'gpt-oss-H': ox, oy = 8, -12
        elif s == 'Moonlight-16B': ox, oy = 8, 12
        elif s == 'Q3-30B-MoE': ox, oy = 8, 0
        else: ox, oy = 8, 8
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(ox, oy),
                    fontsize=6.2, alpha=0.7, ha='left',
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.25, lw=0.4))

    # Also label unreliable models (smaller, gray)
    for _, r in unreliable.iterrows():
        ax.annotate(r['short'], (r['macro_f1_mean_A'], r['macro_f1_mean_B']),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=5.5, alpha=0.4, color='gray', ha='left')

    ax.set_xlabel('Condition A (Extraction) Macro F1')
    ax.set_ylabel('Condition B (Reasoning) Macro F1')
    ax.set_xlim(0.08, 1.02); ax.set_ylim(0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.12, linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), framealpha=0.9, title='Model Family')
    plt.tight_layout()
    save_figure(fig, 'Figure_1')
    print('Figure_1 done')

# =============================================================================
# FIGURE 2: Family bar chart
# =============================================================================

def fig2_family(df):
    fib_a = df[(df.condition == 'A_extraction') & (df.task == 'fibrosis')].copy()
    fib_a = fib_a[fib_a['error_rate_mean'] < 0.10]
    stats = fib_a.groupby('family').agg(
        mf1=('macro_f1_mean', 'mean'), std=('macro_f1_mean', 'std'), n=('model', 'count')
    ).reset_index().sort_values('mf1', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [FAMILY_COLORS.get(f, '#999') for f in stats['family']]
    ax.barh(range(len(stats)), stats['mf1'], xerr=stats['std'],
            color=colors, edgecolor='white', linewidth=0.6, height=0.55, capsize=2.5)
    for i, (_, r) in enumerate(stats.iterrows()):
        ax.text(r['mf1'] + 0.012, i, f"{r['mf1']:.3f}", va='center', fontsize=9, fontweight='bold')
        ax.text(0.015, i, f"n={int(r['n'])}", va='center', fontsize=7.5, color='white', fontweight='bold')
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels(stats['family'], fontsize=10)
    ax.set_xlabel('Mean Macro F1 (Fibrosis, Condition A)')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.15, linestyle='--')
    ax.invert_yaxis()
    plt.tight_layout()
    save_figure(fig, 'Figure_2')
    print('Figure_2 done')

# =============================================================================
# FIGURE 3: Qwen3 Scale — A/B/C labels + blue line values
# =============================================================================

def fig3_scale(df):
    qwen = [
        ('Qwen3-4B-Instruct-2507-Q4_K_M.gguf', 4, '4B'),
        ('Qwen3-8B-Q4_K_M.gguf', 8, '8B'),
        ('Qwen3-14B-Q4_K_M.gguf', 14, '14B'),
        ('Qwen3-32B-Q4_K_M.gguf', 32, '32B'),
    ]
    tasks = ['fibrosis', 'inflammation', 'steatosis']
    tl = {'fibrosis': 'Fibrosis', 'inflammation': 'Inflammation', 'steatosis': 'Steatosis'}
    panel_labels = ['A', 'B', 'C']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for i, task in enumerate(tasks):
        ax = axes[i]
        xs = [p[1] for p in qwen]

        # Blue (A) line
        a_ys, a_es = [], []
        for mname, _, _ in qwen:
            r = df[(df.model == mname) & (df.condition == 'A_extraction') & (df.task == task)]
            a_ys.append(r.iloc[0]['macro_f1_mean'] if len(r) > 0 else np.nan)
            a_es.append(r.iloc[0]['macro_f1_std'] if len(r) > 0 else 0)
        ax.errorbar(xs, a_ys, yerr=a_es, fmt='-o', color='#1f77b4', linewidth=2.2,
                   markersize=8, capsize=3, alpha=0.9, label='Extraction (A)')

        # Red (B) line
        b_ys, b_es = [], []
        for mname, _, _ in qwen:
            r = df[(df.model == mname) & (df.condition == 'B_inference') & (df.task == task)]
            b_ys.append(r.iloc[0]['macro_f1_mean'] if len(r) > 0 else np.nan)
            b_es.append(r.iloc[0]['macro_f1_std'] if len(r) > 0 else 0)
        ax.errorbar(xs, b_ys, yerr=b_es, fmt='--s', color='#d62728', linewidth=1.6,
                   markersize=7, capsize=3, alpha=0.85, label='Reasoning (B)')

        # Value labels on BOTH lines
        for j, (x, ay, by) in enumerate(zip(xs, a_ys, b_ys)):
            if not np.isnan(ay):
                va = 'bottom' if j < 2 else 'top'
                ax.text(x, ay + 0.03, f'{ay:.2f}', ha='center', fontsize=7.5, color='#1f77b4', va=va, fontweight='bold')
            if not np.isnan(by):
                ax.text(x, by - 0.04, f'{by:.2f}', ha='center', fontsize=7.5, color='#d62728', va='top')

        # Δ gap annotations — centered on the dashed line
        for j, (mname, x, _) in enumerate(qwen):
            a_r = df[(df.model == mname) & (df.condition == 'A_extraction') & (df.task == task)]
            b_r = df[(df.model == mname) & (df.condition == 'B_inference') & (df.task == task)]
            if len(a_r) > 0 and len(b_r) > 0:
                ay = a_r.iloc[0]['macro_f1_mean']
                by = b_r.iloc[0]['macro_f1_mean']
                gap = ay - by
                mid_y = (ay + by) / 2
                # Vertical dashed line connecting A and B dots
                ax.plot([x, x], [by, ay], color='#888888', linewidth=0.8,
                       linestyle=':', alpha=0.5, zorder=0)
                # Gap label: placed LEFT of the dashed line
                ax.annotate(f'Δ={gap:.2f}', (x, mid_y),
                           textcoords="offset points", xytext=(0, -2),
                           fontsize=7, ha='center', va='top',
                           color='#555555', fontstyle='italic',
                           bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.7))

        # Panel label
        ax.text(-0.12, 1.04, panel_labels[i], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')

        ax.set_xticks(xs); ax.set_xticklabels([p[2] for p in qwen])
        ax.set_xlabel('Model Scale')
        if i == 0: ax.set_ylabel('Macro F1')
        ax.set_title(tl[task], fontweight='bold', fontsize=12)
        ax.set_ylim(0.05, 1.08)
        ax.grid(True, alpha=0.12, linestyle='--')
        if i == 2: ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

    plt.tight_layout()
    save_figure(fig, 'Figure_3')
    print('Figure_3 done')

# =============================================================================
# FIGURE 4: Confusion Matrix — A/B labels + all cells annotated
# =============================================================================

def fig4_confusion():
    pred_dir = os.path.join(RESULTS_DIR, 'v2_eval_20260508_111923')
    pred_file = os.path.join(pred_dir, 'v2_predictions_all.csv')
    if not os.path.exists(pred_file):
        print('Figure_4 SKIPPED')
        return

    preds = pd.read_csv(pred_file, encoding='utf-8-sig')
    sub = preds[(preds['model'] == 'Qwen3-14B-Q4_K_M.gguf') &
                (preds['condition'] == 'A_extraction') &
                (preds['task'] == 'fibrosis')]
    sub = sub.dropna(subset=['predicted_label'])
    y_true = sub['true_label'].astype(int)
    y_pred = sub['predicted_label'].astype(int)

    labels = [0, 1, 2, 3, 4]
    cm = sk_cm(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Counts — manual annotation for reliability
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax1,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count', 'shrink': 0.8}, linewidths=0.6, linecolor='white')
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = 'white' if val > cm.max() * 0.5 else 'black'
            ax1.text(j + 0.5, i + 0.5, str(val), ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)
    ax1.set_xlabel('Predicted Stage'); ax1.set_ylabel('True Stage')
    ax1.set_title('A. Counts', fontweight='bold', fontsize=12, loc='left')

    # Panel B: Normalized — manual annotation
    sns.heatmap(cm_norm, annot=False, cmap='Blues', ax=ax2,
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1,
                cbar_kws={'label': 'Recall', 'shrink': 0.8}, linewidths=0.6, linecolor='white')
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax2.text(j + 0.5, i + 0.5, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)
    ax2.set_xlabel('Predicted Stage'); ax2.set_ylabel('True Stage')
    ax2.set_title('B. Normalized (Recall)', fontweight='bold', fontsize=12, loc='left')

    plt.tight_layout()
    save_figure(fig, 'Figure_4')
    print(f'Figure_4 done: acc={np.mean(y_true==y_pred):.3f}')

# =============================================================================
# FIGURE 5: Language Advantage — A/B/C labels + clean legend
# =============================================================================

def fig5_language(df):
    pairs = [
        ('4B', ['Qwen3-4B-Instruct-2507-Q4_K_M.gguf'],
               ['gemma-3-4b-it-Q4_K_M.gguf', 'medgemma-4b-it-Q4_K_M.gguf',
                'tencent_Hunyuan-4B-Instruct-Q4_K_M.gguf']),
        ('8B', ['Qwen3-8B-Q4_K_M.gguf'],
               ['Llama-3.1-8B-Instruct-Q4_K_M.gguf', 'granite-3.3-8b-instruct-Q4_K_M.gguf',
                'tencent_Hunyuan-7B-Instruct-Q4_K_M.gguf']),
        ('12-14B', ['Qwen3-14B-Q4_K_M.gguf'],
                   ['gemma-3-12b-it-Q4_K_M.gguf']),
        ('27-32B', ['Qwen3-32B-Q4_K_M.gguf'],
                   ['gemma-3-27b-it-Q4_K_M.gguf', 'medgemma-27b-it-Q4_K_M.gguf']),
    ]
    tasks = ['fibrosis', 'inflammation', 'steatosis']
    tl = {'fibrosis': 'Fibrosis', 'inflammation': 'Inflammation', 'steatosis': 'Steatosis'}
    panel_labels = ['A', 'B', 'C']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    for i, task in enumerate(tasks):
        ax = axes[i]
        data = []
        for label, cn_models, en_models in pairs:
            cn_vals = df[(df.model.isin(cn_models)) & (df.condition == 'A_extraction') & (df.task == task)]['macro_f1_mean']
            en_vals = df[(df.model.isin(en_models)) & (df.condition == 'A_extraction') & (df.task == task)]['macro_f1_mean']
            if len(cn_vals) > 0 and len(en_vals) > 0:
                data.append({'Scale': label, 'Type': 'Chinese-native', 'MF1': cn_vals.mean()})
                data.append({'Scale': label, 'Type': 'English-native', 'MF1': en_vals.mean()})

        dfp = pd.DataFrame(data)
        if len(dfp) == 0: continue

        # Use hue without legend title (set title to empty string)
        sns.barplot(data=dfp, x='Scale', y='MF1', hue='Type', ax=ax,
                    palette={'Chinese-native': '#1f77b4', 'English-native': '#ff7f0e'},
                    edgecolor='white', linewidth=0.5)

        # Value labels on bars
        for p in ax.patches:
            if p.get_width() > 0 and p.get_height() > 0.01:
                ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.015,
                        f'{p.get_height():.2f}', ha='center', fontsize=8.5, fontweight='bold')

        ax.set_ylim(0, 1.12)
        ax.set_title(tl[task], fontweight='bold', fontsize=12)
        if i == 0: ax.set_ylabel('Macro F1 (Condition A)')
        ax.grid(axis='y', alpha=0.12, linestyle='--')

        # Clean legend: remove title, place properly
        leg = ax.legend(loc='upper right', framealpha=0.85, fontsize=8.5, title='')
        leg.set_title('')

        # Panel label
        ax.text(-0.1, 1.04, panel_labels[i], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top')

    plt.tight_layout()
    save_figure(fig, 'Figure_5')
    print('Figure_5 done')

# =============================================================================
# FIGURE 6: Thinking Model Errors
# =============================================================================

def fig6_thinking_errors(df):
    def is_thinking(m):
        return any(t in m for t in ['Think', '0528', 'Distill', 'gpt-oss'])
    df['is_think'] = df['model'].apply(is_thinking)
    f = df[(df.condition == 'A_extraction') & (df.task == 'fibrosis')].copy()
    f = f.sort_values('error_rate_mean', ascending=False)
    top_err = f.nlargest(10, 'error_rate_mean')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['#d62728' if is_thinking(m) else '#1f77b4' for m in top_err['model']]
    ax.barh(range(len(top_err)), top_err['error_rate_mean'] * 100,
            color=colors, edgecolor='white', linewidth=0.5, height=0.55)

    for i, (_, r) in enumerate(top_err.iterrows()):
        pct = r['error_rate_mean'] * 100
        ax.text(pct + 0.8, i, f"{r['short']}  ({pct:.1f}%)",
                va='center', fontsize=8.5)

    ax.set_yticks([])
    ax.set_xlabel('Parse Error Rate (%)')
    ax.set_xlim(0, max(top_err['error_rate_mean']) * 100 + 20)
    ax.grid(axis='x', alpha=0.12, linestyle='--')
    ax.invert_yaxis()

    legend_elements = [mpatches.Patch(facecolor='#d62728', label='Thinking / Reasoning Models'),
                       mpatches.Patch(facecolor='#1f77b4', label='Standard Instruction Models')]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    save_figure(fig, 'Figure_6')
    print('Figure_6 done')

# =============================================================================
# TABLES
# =============================================================================

def generate_tables(df):
    for cond, cond_name in [('A_extraction', 'Table_1_ConditionA'),
                              ('B_inference', 'Table_2_ConditionB')]:
        rows = []
        for task in ['fibrosis', 'inflammation', 'steatosis']:
            sub = df[(df.condition == cond) & (df.task == task)].sort_values('macro_f1_mean', ascending=False)
            for rank, (_, r) in enumerate(sub.iterrows(), 1):
                rows.append({
                    'Task': task.capitalize(), 'Rank': rank, 'Model': r['short'],
                    'Macro_F1': round(r['macro_f1_mean'], 3),
                    'F1_SD': round(r['macro_f1_std'], 3),
                    'Accuracy': round(r['accuracy_mean'], 3),
                    'Acc_SD': round(r['accuracy_std'], 3),
                    'Error_Rate': round(r['error_rate_mean'], 4),
                    'Macro_Precision': round(r['macro_precision_mean'], 3),
                    'Macro_Recall': round(r['macro_recall_mean'], 3),
                    'Macro_Specificity': round(r['macro_specificity_mean'], 3),
                })
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f'{cond_name}.csv'),
                                   index=False, encoding='utf-8-sig')
        print(f'{cond_name}: {len(rows)} rows')

    gaps = []
    for task in ['fibrosis', 'inflammation', 'steatosis']:
        a = df[(df.condition == 'A_extraction') & (df.task == task)].set_index('model')
        b = df[(df.condition == 'B_inference') & (df.task == task)].set_index('model')
        merged = a[['short', 'macro_f1_mean']].join(b[['macro_f1_mean']], lsuffix='_A', rsuffix='_B')
        merged['gap'] = merged['macro_f1_mean_A'] - merged['macro_f1_mean_B']
        for _, r in merged.sort_values('gap', ascending=False).iterrows():
            gaps.append({
                'Task': task.capitalize(), 'Model': r['short'],
                'A_MF1': round(r['macro_f1_mean_A'], 3),
                'B_MF1': round(r['macro_f1_mean_B'], 3),
                'Gap': round(r['gap'], 3),
            })
    pd.DataFrame(gaps).to_csv(os.path.join(OUTPUT_DIR, 'Table_3_InferenceGap.csv'),
                               index=False, encoding='utf-8-sig')
    print(f'Table_3: {len(gaps)} rows')

# =============================================================================
# FIGURE CAPTIONS
# =============================================================================

def write_captions():
    captions = """# Figure & Table Captions — v2 Revised Paper

## 文件映射

| 文件名 | 类型 | 子图 |
|--------|------|------|
| Figure_1 | 散点图 | 单面板, 28模型 |
| Figure_2 | 柱状图 | 单面板, 8家族 |
| Figure_3 | 折线图 | A/B/C 三面板 (Fibrosis/Inflammation/Steatosis) |
| Figure_4 | 混淆矩阵 | A/B 两面板 (Counts/Normalized) |
| Figure_5 | 柱状图 | A/B/C 三面板 (Fibrosis/Inflammation/Steatosis) |
| Figure_6 | 柱状图 | 单面板, Top10错误率 |
| Table_1 | CSV | Condition A 28模型 x 3任务 完整排名 |
| Table_2 | CSV | Condition B 28模型 x 3任务 完整排名 |
| Table_3 | CSV | Inference Gap (A-B) 排序 |

---

## Figure Captions

### Figure 1
**The extraction-reasoning gap in liver fibrosis grading.** Each point represents one of the 28 evaluated LLMs. The x-axis shows the Macro F1 score under Condition A (extraction from full pathology reports with explicit diagnostic conclusions), and the y-axis shows the Macro F1 under Condition B (reasoning from microscopy descriptions without diagnostic conclusions). The dashed diagonal line (y=x) represents the hypothetical scenario of no performance gap. All models fall below this line, confirming a systematic decline when explicit diagnostic labels are absent. Hollow circles denote models with >5% parse failure rate. Model families are differentiated by color as shown in the legend. All 28 models are individually labeled.

### Figure 2
**Performance by model family on clinical data extraction.** Horizontal bar chart showing the mean Macro F1 score for each of the eight model families under Condition A (extraction) for liver fibrosis grading. Error bars represent the standard deviation across models within each family. The number of models per family (n) is indicated within each bar. Chinese-native families (Alibaba Qwen3, Zhipu GLM-4) achieve the highest mean scores, while primarily English-trained families (Meta Llama, IBM Granite) rank lowest.

### Figure 3
**Scale effect within the Qwen3 family across extraction and reasoning tasks.** Three-panel line chart showing Macro F1 scores of four Qwen3 model variants (4B, 8B, 14B, 32B) under Condition A (extraction, solid blue line) and Condition B (reasoning, dashed red line). (A) Fibrosis staging. (B) Inflammation grading. (C) Steatosis grading. Value labels are shown for both conditions at each scale point. Vertical dotted lines connect the A and B values at each scale, with the Δ annotation indicating the inference gap. Performance peaks at 14B for fibrosis and steatosis extraction, with diminishing returns or slight decline at 32B.

### Figure 4
**Confusion matrix for Qwen3-14B on fibrosis extraction (Condition A), aggregated across 10 independent runs (410 total predictions).** (A) Absolute prediction counts. (B) Row-normalized recall values. The model achieves an overall accuracy of 0.961. Errors occur predominantly between adjacent fibrosis stages (e.g., Stage 1 misclassified as Stage 2), indicating that the model respects the ordinal nature of fibrosis staging and rarely commits gross misclassifications across non-adjacent stages.

### Figure 5
**Language advantage: Chinese-native versus English-native models at matched parameter scales.** Grouped bar chart comparing mean Macro F1 scores under Condition A across four scale tiers. (A) Fibrosis staging — Chinese-native models show a ~0.33 F1 advantage at the 8B scale, narrowing to ~0.06 at 27-32B. (B) Inflammation grading — a smaller but consistent advantage for Chinese-native models. (C) Steatosis grading — the language gap is most pronounced at 4B and diminishes at larger scales. Overall, Chinese-native LLMs demonstrate a decisive advantage at smaller parameter counts, with the performance gap narrowing as model scale increases.

### Figure 6
**Parse error rates of thinking-enhanced versus standard instruction models.** Horizontal bar chart showing the top 10 models ranked by parse error rate under Condition A for fibrosis grading. Models with built-in reasoning/thinking mechanisms (red bars: Qwen3-Thinking-2507, Qwen3-30B-A3B-Thinking, DeepSeek-R1 variants, gpt-oss-20b) dominate the high-error region, with Qwen3-4B-Thinking-2507 exceeding 56% failure rate. Standard instruction-tuned models (blue bars: Qwen3-Instruct, Qwen3-8B standard, GLM-4 standard, etc.) consistently achieve near-zero error rates. This demonstrates that reasoning-enhancement training can paradoxically degrade performance on format-constrained structured extraction tasks.

---

## Table Captions

### Table 1
**Complete Condition A (Extraction) rankings.** Macro F1, Accuracy, Precision, Recall, Specificity (all mean ± SD across 10 independent runs), and parse error rate for all 28 models on fibrosis, inflammation, and steatosis grading tasks under Condition A (extraction from full pathology reports with diagnostic conclusions).

### Table 2
**Complete Condition B (Reasoning) rankings.** Same metrics as Table 1, but under Condition B (reasoning from microscopy descriptions without diagnostic conclusions).

### Table 3
**Inference gap summary.** For each model and task, the Condition A Macro F1 (A_MF1), Condition B Macro F1 (B_MF1), and their difference (Gap = A_MF1 − B_MF1). Sorted by gap in descending order. Larger gaps indicate greater performance degradation when transitioning from extraction to reasoning.
"""
    with open(os.path.join(OUTPUT_DIR, 'figure_captions.md'), 'w', encoding='utf-8') as f:
        f.write(captions)
    print('Captions written')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    fig1_scatter(df)
    fig2_family(df)
    fig3_scale(df)
    fig4_confusion()
    fig5_language(df)
    fig6_thinking_errors(df)
    generate_tables(df)
    write_captions()
    print(f'\nAll outputs: {OUTPUT_DIR}')
