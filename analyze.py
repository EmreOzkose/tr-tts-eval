"""
analyze.py — TTS evaluation yardımcı fonksiyonları.
analyze.ipynb bu modülü import ederek sadece config + display çağrıları içerir.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Sabitler ──────────────────────────────────────────────────────────────────

LOWER_IS_BETTER = {"wer", "low_level_sibilance", "low_level_peak_clips", "low_level_dc_offset"}
IGNORE_SCORE_DEFAULT = ["low_level_dc_offset", "low_level_peak_clips", "wer"]

RADAR_PALETTE = [
    "#3B82F6", "#10B981", "#F43F5E", "#F59E0B",
    "#A855F7", "#06B6D4", "#84CC16", "#EF4444",
]
BG_COLOR    = "#0F1117"
GRID_COLOR  = "#2A2D3A"
TEXT_COLOR  = "#E2E8F0"
LABEL_COLOR = "#94A3B8"


# ── Veri yükleme ──────────────────────────────────────────────────────────────

def load_results(result_path_list, wer_results_path, test_path):
    """Sonuç CSV'lerini birleştir; ttsds/wer haritalarını döndür.

    Returns
    -------
    results_df, test_df, model2ttsds, model2wer
    """
    data = {}
    for f in result_path_list:
        df = pd.read_csv(f)
        key_cols = [c for c in ["File", "Input"] if c in df.columns]
        if not key_cols:
            raise KeyError(f"{f} icinde en azindan 'File' kolonu bulunmali.")
        for _, row in df.iterrows():
            key = tuple(row[c] for c in key_cols)
            if key not in data:
                data[key] = {c: row[c] for c in key_cols}
            for col, val in row.items():
                if col in key_cols:
                    continue
                if col not in data[key] or pd.isna(data[key][col]):
                    data[key][col] = val

    results_df = pd.DataFrame(data.values())
    sort_cols = [c for c in ["File", "Input"] if c in results_df.columns]
    if sort_cols:
        results_df = results_df.sort_values(sort_cols).reset_index(drop=True)

    models = results_df["Input"].unique()
    result_df_ttsds = results_df[results_df["File"].isin(models)]
    model2ttsds = (
        dict(zip(result_df_ttsds["Input"], result_df_ttsds["ttsds"]))
        if "ttsds" in result_df_ttsds.columns else {}
    )

    wer_df = pd.read_csv(wer_results_path)
    model2wer = (
        dict(zip(wer_df["Input"], wer_df["Average Result"]))
        if {"Input", "Average Result"}.issubset(wer_df.columns) else {}
    )

    if "ttsds" in results_df.columns:
        del results_df["ttsds"]

    required_cols = [c for c in ["File", "Input"] if c in results_df.columns]
    if required_cols:
        results_df = results_df.dropna(subset=required_cols).copy()

    test_df = pd.read_csv(test_path)
    return results_df, test_df, model2ttsds, model2wer


# ── length_group eşleştirme ───────────────────────────────────────────────────

def _norm_name(v):
    if pd.isna(v):
        return None
    s = os.path.basename(str(v).strip())
    s = re.sub(r"\.[^.]+$", "", s)
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else s.lower()


def _norm_text(v):
    if pd.isna(v):
        return None
    return re.sub(r"\s+", " ", str(v).strip().lower())


def merge_length_groups(results_df, test_df):
    """results_df'ye test_df'deki length_group bilgisini en iyi çabayla ekle."""
    if "length_group" not in test_df.columns:
        raise KeyError("test_df icinde 'length_group' kolonu bulunamadi.")

    grouped_df = None

    # 1) Aynı isimli anahtar
    key_candidates = [
        "utt_id", "id", "file_id", "file_name", "filename",
        "wav", "wav_path", "audio_path", "text",
    ]
    common_keys = [k for k in key_candidates if k in test_df.columns and k in results_df.columns]
    if common_keys:
        jk = common_keys[0]
        print(f"Esleme anahtari: {jk}")
        grouped_df = results_df.merge(test_df[[jk, "length_group"]], on=jk, how="left")

    # 2) Çapraz isimli anahtar
    if grouped_df is None or grouped_df["length_group"].isna().all():
        for tcol, rcol, norm_fn in [
            ("transcription", "Input", _norm_text),
            ("path",          "File",  _norm_name),
            ("id",            "File",  _norm_text),
        ]:
            if tcol in test_df.columns and rcol in results_df.columns:
                tmap = test_df[[tcol, "length_group"]].copy()
                tmap["_k"] = tmap[tcol].map(norm_fn)
                tmap = tmap.drop_duplicates(subset=["_k"], keep="first")
                tmp = results_df.copy()
                tmp["_k"] = tmp[rcol].map(norm_fn)
                cand = tmp.merge(tmap[["_k", "length_group"]], on="_k", how="left").drop(columns=["_k"])
                matched = cand["length_group"].notna().sum()
                if matched > 0:
                    grouped_df = cand
                    print(f"Secilen esleme: test_df[{tcol}] <-> results_df[{rcol}] | eslesen: {matched}")
                    break

    # 3) Blok bazlı fallback
    if grouped_df is None or grouped_df["length_group"].isna().all():
        if len(results_df) % len(test_df) == 0:
            r = len(results_df) // len(test_df)
            print(f"Dogrudan esleme yok; length_group {r}x tekrar edildi.")
            grouped_df = results_df.copy()
            grouped_df["length_group"] = pd.concat([test_df["length_group"]] * r, ignore_index=True)

    # 4) Satır sırası fallback
    if grouped_df is None:
        if len(test_df) == len(results_df):
            print("Satir sirasina gore eslestirme yapildi.")
            grouped_df = results_df.copy()
            grouped_df["length_group"] = test_df["length_group"].values
        else:
            print("Eslestirme yapilamadi; length_group NA atandi.")
            grouped_df = results_df.copy()
            grouped_df["length_group"] = pd.NA

    if "Input" not in grouped_df.columns:
        raise KeyError("results_df icinde 'Input' kolonu bulunamadi.")
    return grouped_df


# ── Model özeti ───────────────────────────────────────────────────────────────

def build_model_summary(grouped_df, model2ttsds, model2wer, ignore_score=None):
    """Model başına metrik ortalamaları + ttsds/wer + overall_score hesapla.

    Returns
    -------
    model_summary, score_cols, num_cols
    """
    if ignore_score is None:
        ignore_score = IGNORE_SCORE_DEFAULT

    num_cols_all = grouped_df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols_all if c not in ignore_score]

    model_summary = (
        grouped_df.groupby("Input")[num_cols].mean()
        .reset_index().sort_values("Input")
    )
    model_summary = model_summary.loc[:, ~model_summary.columns.duplicated()].copy()
    model_summary["ttsds"] = model_summary["Input"].map(model2ttsds)
    model_summary["wer"]   = model_summary["Input"].map(model2wer)

    score_cols = [
        c for c in (num_cols + ["ttsds", "wer"])
        if c in model_summary.columns and model_summary[c].notna().any()
    ]
    score_cols = list(dict.fromkeys(score_cols))

    norm = model_summary[score_cols].copy()
    for c in score_cols:
        c_min, c_max = norm[c].min(), norm[c].max()
        denom = (c_max - c_min) if (c_max - c_min) != 0 else 1
        norm[c] = (c_max - norm[c]) / denom if c in LOWER_IS_BETTER else (norm[c] - c_min) / denom

    model_summary["overall_score"] = norm.mean(axis=1)
    return model_summary, score_cols, num_cols


# ── Stil yardımcıları ─────────────────────────────────────────────────────────

def make_highlight_best_col(non_gold_mask):
    """Sütun başına en iyi değeri yeşil yapan Styler fonksiyonu üretir."""
    def _fn(col):
        best_val = (
            col[non_gold_mask].min(skipna=True)
            if col.name in LOWER_IS_BETTER
            else col[non_gold_mask].max(skipna=True)
        )
        return [
            "color: #006100; font-weight: bold;"
            if non_gold_mask.iloc[i] and pd.notna(v) and v == best_val
            else ""
            for i, v in enumerate(col)
        ]
    return _fn


def highlight_vertical_best(row, cols):
    """Dikey tabloda satır başına en iyi model sütununu yeşil yapar."""
    vals = row[cols]
    best_val = vals.min(skipna=True) if row.name in LOWER_IS_BETTER else vals.max(skipna=True)
    return [
        "color: #006100; font-weight: bold;" if (c in cols and pd.notna(v) and v == best_val) else ""
        for c, v in row.items()
    ]


# ── Tablo gösterimi ───────────────────────────────────────────────────────────

def display_summary_tables(grouped_df, model_summary, score_cols, num_cols, show_results=None):
    """Tüm özet tabloları göster; show_results ile filtreli dikey tablo ekle.

    Returns
    -------
    model_summary_vertical (DataFrame)
    """
    from IPython.display import display as _display  # notebook ortamı

    fmt = {"overall_score": "{:.4f}", "ttsds": "{:.4f}", "wer": "{:.4f}"}
    non_gold_mask = model_summary["Input"].ne("gold")
    _highlight_col = make_highlight_best_col(non_gold_mask)

    # Model dağılımı
    print("\nModel (Input) dagilimi:")
    print(grouped_df["Input"].value_counts(dropna=False))

    # Length + Model breakdown
    print("\nLength + Model bazinda sayisal ortalamalar:")
    _display(
        grouped_df.groupby(["length_group", "Input"])[num_cols].mean()
        .reset_index().sort_values(["length_group", "Input"])
    )

    # En iyi model
    if non_gold_mask.any():
        best_model = model_summary.loc[non_gold_mask].loc[
            model_summary.loc[non_gold_mask, "overall_score"].idxmax(), "Input"
        ]
    else:
        best_model = model_summary.loc[model_summary["overall_score"].idxmax(), "Input"]

    print(f"\nModel bazinda sayisal ortalamalar (length_group kirilimi olmadan):")
    print(f"En iyi model (overall_score, gold haric): {best_model}")

    def _highlight_row(row):
        if row["Input"] == best_model:
            return ["background-color: #c6efce; color: #111111;"] * len(row)
        return [""] * len(row)

    _display(
        model_summary.style
        .apply(_highlight_row, axis=1)
        .apply(_highlight_col, axis=0, subset=score_cols)
        .format(fmt)
    )

    print("\nTum modeller ozeti (sutun en iyileri yesil, gold haric):")
    _display(model_summary.style.apply(_highlight_col, axis=0, subset=score_cols).format(fmt))

    print("\nModel bazinda sayisal ortalamalar (renksiz):")
    _display(model_summary)

    # Dikey tablo
    msv = model_summary.set_index("Input").T.round(4)
    if "gold" in msv.columns:
        msv = msv[[c for c in msv.columns if c != "gold"] + ["gold"]]
    non_gold_cols = [c for c in msv.columns if c != "gold"]

    print("\nModel bazinda sayisal ortalamalar (dikey, 4 ondalik):")
    _display(msv.style.apply(highlight_vertical_best, axis=1, cols=non_gold_cols))

    # Filtreli dikey tablo
    if show_results:
        sl = [s.lower() for s in show_results]
        sel_cols = [c for c in msv.columns if any(s in str(c).lower() for s in sl)]
        title = f"show_results filtresi: {show_results}"
    else:
        sel_cols = non_gold_cols
        title = "tum modeller (gold haric)"

    if sel_cols:
        print(f"\nFiltreli dikey-renkli tablo ({title}, 4 ondalik):")
        _display(msv[sel_cols].style.apply(highlight_vertical_best, axis=1, cols=sel_cols))
    else:
        print("\nshow_results ile eslesen model bulunamadi.")

    return msv


# ── Radar grafiği ─────────────────────────────────────────────────────────────

def _normalize_for_radar(values_dict, lower_better=False):
    vals = np.array(list(values_dict.values()), dtype=float)
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return {k: 0.5 for k in values_dict}
    normed = (vals - mn) / (mx - mn)
    if lower_better:
        normed = 1 - normed
    return dict(zip(values_dict.keys(), normed))


def radar_chart(model_summary, radar_models, radar_metrics, output_name="tts_radar_chart.png", name_map=None):
    """model_summary'den radar grafiği çiz ve kaydet."""
    bg_color = "#FFFFFF"
    grid_color = "#CBD5E1"
    text_color = "#0F172A"
    label_color = "#475569"
    name_map = name_map or {}

    base_df = model_summary.copy()
    available_models  = set(base_df["Input"].astype(str))
    available_metrics = set(c for c in base_df.columns if c != "Input")

    valid_models  = [m for m in radar_models  if m in available_models]
    valid_metrics = [m for m in radar_metrics if m in available_metrics]

    missing_models  = [m for m in radar_models  if m not in available_models]
    missing_metrics = [m for m in radar_metrics if m not in available_metrics]
    if missing_models:
        print(f"Uyari - bulunamayan modeller: {missing_models}")
    if missing_metrics:
        print(f"Uyari - bulunamayan metrikler: {missing_metrics}")

    if len(valid_models) < 2:
        raise ValueError("Radar grafik icin en az 2 gecerli model gerekli.")
    if len(valid_metrics) < 3:
        raise ValueError("Radar grafik icin en az 3 gecerli metrik gerekli.")

    plot_df = base_df.set_index("Input").loc[valid_models, valid_metrics].astype(float)
    normalized = {
        metric: _normalize_for_radar(
            {model: float(plot_df.loc[model, metric]) for model in valid_models},
            lower_better=(metric in LOWER_IS_BETTER),
        )
        for metric in valid_metrics
    }

    N = len(valid_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]
    short_labels = [s.replace("_", "\n") if len(s) > 10 else s for s in valid_metrics]
    colors = {m: RADAR_PALETTE[i % len(RADAR_PALETTE)] for i, m in enumerate(valid_models)}

    fig = plt.figure(figsize=(12, 10), facecolor=bg_color)
    ax  = fig.add_subplot(111, polar=True, facecolor=bg_color)

    for lvl in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles, [lvl] * (N + 1), color=grid_color, linewidth=0.6, zorder=1)
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 1.0], color=grid_color, linewidth=0.5, zorder=1)

    for model in valid_models:
        vals  = [normalized[m][model] for m in valid_metrics]
        vals += vals[:1]
        color = colors[model]
        ax.plot(angles, vals, color=color, linewidth=2.2, zorder=3, solid_capstyle="round")
        ax.fill(angles, vals, color=color, alpha=0.12, zorder=2)
        ax.scatter(angles[:-1], vals[:-1], color=color, s=28, zorder=4,
                   edgecolors=bg_color, linewidths=1.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, color=label_color, fontsize=8.5, linespacing=1.3)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color=label_color, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", pad=10)
    ax.spines["polar"].set_color(grid_color)

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        deg = np.degrees(angle)
        if 90 < deg < 270:
            label.set_horizontalalignment("right")
        elif deg in (90.0, 270.0):
            label.set_horizontalalignment("center")
        else:
            label.set_horizontalalignment("left")

    # fig.text(0.5, 0.97, "TTS Model Comparison", ha="center", va="top",
    #          color=text_color, fontsize=17, fontweight="bold", fontfamily="monospace")

    win_counts = {
        model: sum(
            1 for metric in valid_metrics
            if max(normalized[metric], key=normalized[metric].get) == model
        )
        for model in valid_models
    }
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=colors[m], edgecolor=colors[m],
                        #    label=f"  {name_map.get(m, m)}   ({win_counts[m]}/{len(valid_metrics)} metrics)")
                           label=f"  {name_map.get(m, m)}")
            for m in valid_models
        ],
        loc="center left", bbox_to_anchor=(1.12, 0.5),
        ncol=1, frameon=True, framealpha=0.15,
        edgecolor=grid_color, facecolor=bg_color, fontsize=10, labelcolor=text_color,
    )

    plt.tight_layout(rect=[0, 0.02, 0.76, 0.95])
    plt.savefig(output_name, dpi=160, bbox_inches="tight", facecolor=bg_color, edgecolor="none")
    plt.show()
    print(f"Grafik '{output_name}' olarak kaydedildi.")
