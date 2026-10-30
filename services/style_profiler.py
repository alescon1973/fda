import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
import textwrap
from typing import Dict, Any

# Parametri metrici FIFA 105m x 68m su StatsBomb (120yd x 80yd)
CONV_X = 105.0 / 120.0
CONV_Y = 68.0 / 80.0
CONV_DIST = 0.9144

# Palette Opta Dark Uniformata
OPTA_BG = "#121212"
OPTA_CARD = "#1a1a1a"
OPTA_BORDER = "#2a2a2a"
OPTA_GREEN = "#00ff85"
OPTA_ORANGE = "#ff5722"
OPTA_BLUE = "#00b0ff"
OPTA_LINE = "#383838"
OPTA_TEXT = "#f8f9fa"
OPTA_MUTED = "#8e8e93"

def _is_true(val: Any) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ["true", "1", "1.0", "t"]

def normalize_metric(val: float, min_b: float, max_b: float, invert: bool = False) -> float:
    clipped = np.clip(val, min_b, max_b)
    norm = ((clipped - min_b) / (max_b - min_b)) * 100
    return float(round(100 - norm if invert else norm, 1))

# Matrice standard di Expected Threat (12 righe Y x 16 colonne X)
XT_GRID = np.array([
    [0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.018, 0.022, 0.028, 0.038, 0.054, 0.075, 0.108],
    [0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.014, 0.016, 0.019, 0.024, 0.031, 0.043, 0.063, 0.091, 0.132],
    [0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.013, 0.015, 0.017, 0.021, 0.026, 0.035, 0.050, 0.076, 0.115, 0.178],
    [0.007, 0.007, 0.008, 0.009, 0.010, 0.012, 0.014, 0.016, 0.019, 0.023, 0.030, 0.041, 0.061, 0.098, 0.160, 0.257],
    [0.007, 0.008, 0.008, 0.009, 0.011, 0.013, 0.015, 0.018, 0.021, 0.027, 0.036, 0.051, 0.082, 0.138, 0.245, 0.380],
    [0.007, 0.008, 0.009, 0.010, 0.011, 0.013, 0.016, 0.019, 0.023, 0.030, 0.041, 0.061, 0.101, 0.180, 0.315, 0.540],
    [0.007, 0.008, 0.009, 0.010, 0.011, 0.013, 0.016, 0.019, 0.023, 0.030, 0.041, 0.061, 0.101, 0.180, 0.315, 0.540],
    [0.007, 0.008, 0.008, 0.009, 0.011, 0.013, 0.015, 0.018, 0.021, 0.027, 0.036, 0.051, 0.082, 0.138, 0.245, 0.380],
    [0.007, 0.007, 0.008, 0.009, 0.010, 0.012, 0.014, 0.016, 0.019, 0.023, 0.030, 0.041, 0.061, 0.098, 0.160, 0.257],
    [0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.013, 0.015, 0.017, 0.021, 0.026, 0.035, 0.050, 0.076, 0.115, 0.178],
    [0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.014, 0.016, 0.019, 0.024, 0.031, 0.043, 0.063, 0.091, 0.132],
    [0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.018, 0.022, 0.028, 0.038, 0.054, 0.075, 0.108]
])

def _get_xt_cell(x_m: float, y_m: float) -> float:
    col = int(np.clip(x_m / (105.0 / 16.0), 0, 15))
    row = int(np.clip(y_m / (68.0 / 12.0), 0, 11))
    return float(XT_GRID[row, col])

# =====================================================================
# 1. COMPUTING ENGINE
# =====================================================================

def compute_team_style_metrics(events_df: pd.DataFrame, team_name: str, sample_size: int = 5) -> Dict[str, Any]:
    team_full_events = events_df[events_df["team"] == team_name].copy()
    available_matches = team_full_events["match_id"].dropna().unique().tolist()
    sampled_matches = available_matches[:sample_size]
    n_matches = max(len(sampled_matches), 1)

    team_events = team_full_events[team_full_events["match_id"].isin(sampled_matches)].copy()
    opp_events = events_df[
        (events_df["match_id"].isin(sampled_matches)) & 
        (events_df["team"].notna()) & 
        (events_df["team"] != team_name)
    ].copy()

    team_events["x_m"] = team_events["x"] * CONV_X
    team_events["y_m"] = team_events["y"] * CONV_Y
    team_events["end_x_m"] = team_events["pass_end_x"] * CONV_X
    team_events["end_y_m"] = team_events["pass_end_y"] * CONV_Y

    opp_events["x_m"] = opp_events["x"] * CONV_X
    opp_events["y_m"] = opp_events["y"] * CONV_Y

    # Costruzione Bassa (X <= 35m)
    passes = team_events[team_events["type"] == "Pass"].copy()
    comp_passes = passes[passes["pass_outcome"].isna()].copy()
    total_passes = len(passes)
    pass_acc = (len(comp_passes) / total_passes * 100) if total_passes > 0 else 0.0

    raw_len = float(comp_passes["pass_length"].mean()) if "pass_length" in comp_passes.columns else 20.0
    avg_pass_len_m = raw_len * CONV_DIST

    comp_passes["dx_m"] = comp_passes["end_x_m"] - comp_passes["x_m"]
    prog_passes = comp_passes[comp_passes["dx_m"] >= 9.14]
    prog_ratio = (len(prog_passes) / len(comp_passes) * 100) if len(comp_passes) > 0 else 0.0

    buildup_passes = passes[passes["x_m"] <= 35.0]
    buildup_comp = comp_passes[comp_passes["x_m"] <= 35.0]
    buildup_passes_p90 = len(buildup_passes) / n_matches
    buildup_acc = (len(buildup_comp) / len(buildup_passes) * 100) if not buildup_passes.empty else 0.0
    score_buildup = 0.5 * normalize_metric(buildup_passes_p90, 80.0, 220.0) + 0.5 * normalize_metric(buildup_acc, 75.0, 92.0)

    # Expected Threat (xT) da Passaggi
    xt_gains = []
    high_threat_actions = []

    for _, r in comp_passes.iterrows():
        x1, y1 = r["x_m"], r["y_m"]
        x2, y2 = r["end_x_m"], r["end_y_m"]
        xt_start = _get_xt_cell(x1, y1)
        xt_end = _get_xt_cell(x2, y2)
        gain = xt_end - xt_start
        pos_gain = max(gain, 0.0)
        xt_gains.append(pos_gain)

        if pos_gain >= 0.025:
            high_threat_actions.append({
                "x_start": x1,
                "y_start": y1,
                "x_end": x2,
                "y_end": y2,
                "player": r.get("player", "N/D"),
                "xt_gain": pos_gain
            })

    comp_passes["xt_gain"] = xt_gains
    tot_xt = float(sum(xt_gains))
    xt_p90 = tot_xt / n_matches
    df_high_threat = pd.DataFrame(high_threat_actions)

    # Sequenze e Direct Speed
    team_events_sorted = team_events.sort_values(["match_id", "minute", "second"])
    possession_blocks = (team_events_sorted["possession"] != team_events_sorted["possession"].shift(1)).cumsum()
    seq_stats = []

    for _, seq_df in team_events_sorted.groupby(possession_blocks):
        seq_passes = seq_df[seq_df["type"] == "Pass"]
        p_count = len(seq_passes)
        if p_count >= 1:
            start_x = seq_df.iloc[0]["x_m"]
            end_x = seq_df.iloc[-1]["x_m"]
            delta_x = max(end_x - start_x, 0.0)
            
            t_start = seq_df.iloc[0].get("second", 0) + (seq_df.iloc[0]["minute"] * 60)
            t_end = seq_df.iloc[-1].get("second", 0) + (seq_df.iloc[-1]["minute"] * 60)
            dur = max(t_end - t_start, 1.0)
            
            speed = delta_x / dur
            seq_stats.append({"pass_count": p_count, "speed": speed})

    df_seq = pd.DataFrame(seq_stats)
    avg_passes_per_sequence = float(df_seq["pass_count"].mean()) if not df_seq.empty else 3.2
    avg_direct_speed = float(df_seq["speed"].mean()) if not df_seq.empty else 1.55

    # Field Tilt & Possesso Nominale
    our_f3rd_passes = len(comp_passes[comp_passes["end_x_m"] >= 70.0])
    opp_comp = opp_events[(opp_events["type"] == "Pass") & (opp_events["pass_outcome"].isna())]
    opp_our_third = len(opp_comp[opp_comp["x_m"] <= 35.0])
    tot_deep_territory = max(our_f3rd_passes + opp_our_third, 1)
    field_tilt = round((our_f3rd_passes / tot_deep_territory) * 100, 1)

    opp_tot_passes = len(opp_events[opp_events["type"] == "Pass"])
    nominal_possession = round((total_passes / max(total_passes + opp_tot_passes, 1)) * 100, 1)

    # Canali d'Attacco
    final_third_comp = comp_passes[comp_passes["end_x_m"] >= 70.0]
    l_ch = len(final_third_comp[final_third_comp["end_y_m"] < 22.6])
    c_ch = len(final_third_comp[final_third_comp["end_y_m"].between(22.6, 45.4)])
    r_ch = len(final_third_comp[final_third_comp["end_y_m"] > 45.4])
    tot_third = max(l_ch + c_ch + r_ch, 1)
    channel_dist = {
        "Corsia Sinistra": round((l_ch / tot_third) * 100, 1),
        "Corridoio Centrale": round((c_ch / tot_third) * 100, 1),
        "Corsia Destra": round((r_ch / tot_third) * 100, 1)
    }

    # Cross e penetrazioni
    if "pass_cross" in passes.columns:
        passes["is_cross"] = passes["pass_cross"].apply(_is_true)
    else:
        passes["is_cross"] = False

    box_p = passes[(passes["end_x_m"] >= 88.5) & (passes["end_y_m"].between(13.84, 54.16)) & (passes["x_m"] < 88.5)]
    cross_p = box_p[box_p["is_cross"]]
    cross_ratio = (len(cross_p) / len(box_p) * 100) if len(box_p) > 0 else 0.0

    # Tiri e Qualità delle Conclusioni
    shots = team_events[team_events["type"] == "Shot"].copy()
    total_shots = len(shots)
    total_xg = float(shots["shot_statsbomb_xg"].sum()) if "shot_statsbomb_xg" in shots.columns else 0.0
    xg_per_shot = (total_xg / total_shots) if total_shots > 0 else 0.0

    # Non-Possesso & Pressing
    def_actions = team_events[team_events["type"].isin(["Pressure", "Tackle", "Interception", "Foul Committed", "Block"])].copy()
    raw_def_x = float(def_actions["x"].mean()) if not def_actions.empty else 50.0
    defensive_line_height_m = raw_def_x * CONV_X

    opp_passes_buildup = opp_events[(opp_events["type"] == "Pass") & (opp_events["pass_outcome"].isna()) & (opp_events["x_m"] <= 70.0)]
    team_def_high = def_actions[def_actions["x_m"] >= 35.0]
    ppda = (len(opp_passes_buildup) / len(team_def_high)) if len(team_def_high) > 0 else 0.0
    high_recoveries = len(team_events[(team_events["type"].isin(["Ball Recovery", "Interception"])) & (team_events["x_m"] >= 70.0)])

    # Archetipo Globale
    if ppda < 10.5 and field_tilt > 55.0:
        archetype = "Dominio Posizionale & Gegenpressing"
    elif avg_direct_speed > 1.85 and prog_ratio > 32.0:
        archetype = "Attacco Diretto & Ricerca Profondità"
    elif ppda > 14.0 and defensive_line_height_m < 39.0:
        archetype = "Blocco Basso & Transizione Rapida"
    else:
        archetype = "Manovra Ragionata & Consolidamento"

    return {
        "team_name": team_name,
        "matches_analyzed": n_matches,
        "sampled_matches": sampled_matches,
        "archetype": archetype,
        "buildup_score": round(score_buildup, 1),
        "buildup_accuracy": round(buildup_acc, 1),
        "passes_p90": round(total_passes / n_matches, 1),
        "pass_accuracy": round(pass_acc, 1),
        "avg_pass_length_m": round(avg_pass_len_m, 1),
        "progressive_pass_ratio": round(prog_ratio, 1),
        "xt_p90": round(xt_p90, 3),
        "df_high_threat": df_high_threat,
        "avg_passes_per_sequence": round(avg_passes_per_sequence, 2),
        "direct_speed_ms": round(avg_direct_speed, 2),
        "field_tilt": field_tilt,
        "nominal_possession": nominal_possession,
        "channel_dist": channel_dist,
        "cross_ratio": round(cross_ratio, 1),
        "shots_p90": round(total_shots / n_matches, 1),
        "xg_p90": round(total_xg / n_matches, 2),
        "xg_per_shot": round(xg_per_shot, 3),
        "ppda": round(ppda, 2),
        "defensive_height_m": round(defensive_line_height_m, 1),
        "high_recoveries_p90": round(high_recoveries / n_matches, 1),
        "shots_df": shots
    }

def compute_tactical_scores(metrics: dict) -> dict:
    s_build = metrics["buildup_score"]
    s_attack = round(0.5 * normalize_metric(metrics["xg_p90"], 0.6, 2.4) + 0.5 * normalize_metric(metrics["xt_p90"], 0.8, 2.2), 1)
    s_press = round(0.40 * normalize_metric(metrics["ppda"], 7.0, 18.0, invert=True) + 0.35 * normalize_metric(metrics["defensive_height_m"], 33.0, 50.0) + 0.25 * normalize_metric(metrics["high_recoveries_p90"], 1.0, 6.0), 1)
    s_territory = round(0.6 * normalize_metric(metrics["field_tilt"], 35.0, 65.0) + 0.4 * (100 - metrics["cross_ratio"] * 2), 1)

    scores = {
        "Costruzione & Uscita Bassa": s_build,
        "Pericolosità & xT Offensivo": s_attack,
        "Pressing & Baricentro": s_press,
        "Controllo Territoriale (Field Tilt)": s_territory
    }

    sorted_p = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_strength = sorted_p[0]
    top_weakness = sorted_p[-1]

    recommendations = {
        "Costruzione & Uscita Bassa": "Velocizzare i tempi di trasmissione ed evitare appoggi orizzontali sotto pressione.",
        "Pericolosità & xT Offensivo": "Privilegiare imbucate centrali e diagonali verso l'area per massimizzare il rendimento del modello xT.",
        "Pressing & Baricentro": "Mantenere le distanze corte tra i reparti evitando che il centrocampo si allunghi sui cambi gioco.",
        "Controllo Territoriale (Field Tilt)": "Aumentare la pressione sul secondo pallone per prolungare la permanenza negli ultimi 35 metri."
    }

    return {
        "scores": scores,
        "overall_rating": round(float(np.mean(list(scores.values()))), 1),
        "strength": top_strength,
        "weakness": top_weakness,
        "prescription": recommendations[top_weakness[0]]
    }

# =====================================================================
# 2. GRAFICI: WHEEL, SHOT MAP, xT FLOW, MATRICE OPTA & FIELD TILT
# =====================================================================

def plot_opta_style_wheel(metrics: dict) -> go.Figure:
    categories = [
        "Fase di Costruzione",
        "Volume Possesso",
        "Verticalità Manovra",
        "Minaccia Territoriale (xT)",
        "Intensità Pressing (PPDA)",
        "Altezza Baricentro",
        "Field Tilt Territoriale"
    ]

    scores = [
        metrics["buildup_score"],
        np.clip((metrics["passes_p90"] - 300) / 400 * 100, 10, 95),
        np.clip(metrics["progressive_pass_ratio"] * 2.5, 10, 95),
        np.clip(metrics["xt_p90"] / 2.2 * 100, 10, 95),
        np.clip((25.0 - metrics["ppda"]) / 18.0 * 100, 10, 95),
        np.clip((metrics["defensive_height_m"] - 31) / 22 * 100, 10, 95),
        np.clip(metrics["field_tilt"], 10, 95)
    ]
    colors = [OPTA_GREEN if s >= 70 else (OPTA_ORANGE if s >= 50 else "#ff5252") for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=scores,
        theta=categories,
        marker=dict(color=colors, line=dict(color="#1a1a1a", width=2)),
        opacity=0.88,
        hovertemplate="<b>%{theta}</b><br>Score: <b>%{r:.1f}/100</b><extra></extra>"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=OPTA_MUTED, size=9), gridcolor=OPTA_LINE, showline=False),
            angularaxis=dict(tickfont=dict(size=10, color=OPTA_TEXT, family="Arial"), linecolor=OPTA_LINE, gridcolor=OPTA_LINE),
            bgcolor=OPTA_CARD
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=45, r=45, t=25, b=25)
    )
    return fig

def plot_shot_map_mplsoccer(shots_df: pd.DataFrame, team_name: str) -> plt.Figure:
    """Mappa dei tiri su metà campo offensiva con cerchi proporzionali agli xG."""
    pitch = VerticalPitch(
        pitch_type='statsbomb',
        half=True,
        pitch_color=OPTA_BG,
        line_color=OPTA_LINE,
        linewidth=1.2,
        goal_type='box'
    )
    fig, ax = pitch.draw(figsize=(8, 6.5))
    fig.patch.set_facecolor(OPTA_CARD)

    if not shots_df.empty:
        for _, s in shots_df.iterrows():
            sx = s["x"]
            sy = s["y"]
            xg = float(s.get("shot_statsbomb_xg", 0.08))
            outcome = str(s.get("shot_outcome", "")).lower()

            size = max(xg * 650, 45)
            if outcome == "goal":
                color = OPTA_GREEN
                border = "#ffffff"
                alpha = 0.95
                zorder = 5
            elif outcome in ["saved", "blocked"]:
                color = OPTA_BLUE
                border = OPTA_BG
                alpha = 0.70
                zorder = 3
            else:
                color = "#475569"
                border = OPTA_BG
                alpha = 0.50
                zorder = 2

            pitch.scatter(sx, sy, ax=ax, s=size, color=color, edgecolors=border, linewidth=1.2, alpha=alpha, zorder=zorder)

    ax.set_title(f"MAPPA CONCLUSIONI & QUALITÀ TIRI (xG) — {team_name.upper()}", color=OPTA_TEXT, fontsize=11, pad=10, fontweight='bold')
    return fig

def plot_xt_threat_flow(df_threat: pd.DataFrame, team_name: str) -> go.Figure:
    """
    Renderizza la mappa Expected Threat (xT) interattiva con Plotly:
    - Livello 0: Geometria vettoriale del campo regolamentare FIFA (105m x 68m).
    - Livello 1: Heatmap xT (16x12) a layer indipendente e disattivabile.
    - Livello 2: Vettori chiave di progressione con tooltip dettagliati e frecce proporzionate.
    """
    fig = go.Figure()

    # -------------------------------------------------------------
    # LIVELLO 0: LINEE DI DEMARCAZIONE DEL CAMPO (105m x 68m)
    # -------------------------------------------------------------
    pitch_shapes = [
        # Perimetro esterno
        dict(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color=OPTA_LINE, width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Linea di metà campo
        dict(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color=OPTA_LINE, width=1.5)),
        # Cerchio di centrocampo
        dict(type="circle", x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15, line=dict(color=OPTA_LINE, width=1.5)),
        # Punto centrale
        dict(type="circle", x0=52.5 - 0.4, y0=34 - 0.4, x1=52.5 + 0.4, y1=34 + 0.4, fillcolor=OPTA_LINE, line=dict(color=OPTA_LINE, width=1)),
        # Area di rigore sinistra (difesa)
        dict(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color=OPTA_LINE, width=1.5)),
        # Area piccola sinistra
        dict(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color=OPTA_LINE, width=1.0)),
        # Area di rigore destra (attacco)
        dict(type="rect", x0=105 - 16.5, y0=13.84, x1=105, y1=54.16, line=dict(color=OPTA_LINE, width=1.5)),
        # Area piccola destra
        dict(type="rect", x0=105 - 5.5, y0=24.84, x1=105, y1=43.16, line=dict(color=OPTA_LINE, width=1.0)),
        # Linea degli ultimi 35 metri (trequarti offensiva)
        dict(type="line", x0=70.0, y0=0, x1=70.0, y1=68, line=dict(color=OPTA_BORDER, width=1.2, dash="dash"))
    ]
    fig.update_layout(shapes=pitch_shapes)

    # -------------------------------------------------------------
    # LIVELLO 1: HEATMAP EXPECTED THREAT (xT) INDIPENDENTE
    # -------------------------------------------------------------
    # Centri delle 16 colonne X e delle 12 righe Y in metri
    x_centers = np.linspace(105.0 / 32.0, 105.0 - (105.0 / 32.0), 16)
    y_centers = np.linspace(68.0 / 24.0, 68.0 - (68.0 / 24.0), 12)

    # Hover text personalizzato per la griglia xT
    hover_grid = []
    for r in range(12):
        row_txt = []
        for c in range(16):
            val = XT_GRID[r, c]
            row_txt.append(f"<b>Cella xT</b><br>Coordinate: {x_centers[c]:.1f}m, {y_centers[r]:.1f}m<br>Minaccia Base: <b>+{val:.3f}</b>")
        hover_grid.append(row_txt)

    fig.add_trace(go.Heatmap(
        z=XT_GRID,
        x=x_centers,
        y=y_centers,
        colorscale="Magma",
        opacity=0.30,
        showscale=False,
        name="Heatmap xT (16x12)",
        text=hover_grid,
        hoverinfo="text",
        hoverongaps=False
    ))

    # -------------------------------------------------------------
    # LIVELLO 2: VETTORI DI MINACCIA CHIAVE (ΔxT >= +0.025)
    # -------------------------------------------------------------
    if not df_threat.empty:
        top_threat = df_threat.sort_values("xt_gain", ascending=False).head(30)
        max_gain = max(top_threat["xt_gain"].max(), 0.025)

        # Tracciamo ogni traiettoria come annotazione con freccia proporzionata
        for _, r in top_threat.iterrows():
            sx, sy = r["x_start"], r["y_start"]
            ex, ey = r["x_end"], r["y_end"]
            gain = r["xt_gain"]

            norm_g = gain / max_gain
            arrow_w = 1.0 + (norm_g * 1.8)
            alpha = 0.50 + (norm_g * 0.45)

            # Freccia vettoriale calibrata
            fig.add_annotation(
                x=ex, y=ey,
                ax=sx, ay=sy,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=arrow_w,
                arrowcolor=f"rgba(0, 255, 133, {alpha:.2f})",
                opacity=alpha
            )

        # Traccia invisibile di appoggio per fornire il tooltip su ciascuna freccia
        hover_arrows = []
        for _, r in top_threat.iterrows():
            hover_arrows.append(
                f"<b>{r['player']}</b><br>"
                f"Traiettoria: ({r['x_start']:.1f}m, {r['y_start']:.1f}m) &rarr; ({r['x_end']:.1f}m, {r['y_end']:.1f}m)<br>"
                f"Minaccia Generata: <b>+{r['xt_gain']:.3f} &Delta;xT</b>"
            )

        mid_x = (top_threat["x_start"] + top_threat["x_end"]) / 2.0
        mid_y = (top_threat["y_start"] + top_threat["y_end"]) / 2.0

        fig.add_trace(go.Scatter(
            x=mid_x,
            y=mid_y,
            mode="markers",
            marker=dict(size=8, color=OPTA_GREEN, opacity=0.01),
            name="Vettori Chiave (ΔxT)",
            text=hover_arrows,
            hoverinfo="text"
        ))

    # -------------------------------------------------------------
    # LAYOUT & CONFIGURAZIONE PROPORZIONI RIGOROSE (105 x 68)
    # -------------------------------------------------------------
    fig.update_layout(
        xaxis=dict(
            range=[-2, 107],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[70, -2],
            autorange="reversed",  # Coordinate coerenti: Y=0 in alto
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",       # Blocca il rapporto d'aspetto 1:1
            scaleratio=1,
            fixedrange=True
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=OPTA_BG,
        showlegend=True,
        legend=dict(
            font=dict(color=OPTA_MUTED, size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=480
    )

    return fig

def plot_opta_style_matrix(direct_speed: float, passes_per_seq: float) -> go.Figure:
    """Punto B: Matrice di stile a 4 quadranti (Direct Speed vs Passes per Sequence)."""
    fig = go.Figure()

    med_speed = 1.60
    med_passes = 3.30

    fig.add_vline(x=med_passes, line=dict(color=OPTA_LINE, dash="dash", width=1.5))
    fig.add_hline(y=med_speed, line=dict(color=OPTA_LINE, dash="dash", width=1.5))

    fig.add_annotation(x=4.6, y=2.4, text="<b>FAST & DIRECT</b><br><span style='font-size:9px;color:#8e8e93;'>Transizioni lunghe ed immediate</span>", showarrow=False, font=dict(color=OPTA_ORANGE, size=10))
    fig.add_annotation(x=4.6, y=0.9, text="<b>SLOW & DIRECT</b><br><span style='font-size:9px;color:#8e8e93;'>Attacchi prolungati ma lineari</span>", showarrow=False, font=dict(color="#cbd5e1", size=10))
    fig.add_annotation(x=1.8, y=2.4, text="<b>TRANSIZIONI RAPIDE</b><br><span style='font-size:9px;color:#8e8e93;'>Pochi passaggi ad alta velocità</span>", showarrow=False, font=dict(color="#f87171", size=10))
    fig.add_annotation(x=1.8, y=0.9, text="<b>POSSESSO ELABORATO</b><br><span style='font-size:9px;color:#8e8e93;'>Dominio posizionale e pazienza</span>", showarrow=False, font=dict(color=OPTA_GREEN, size=10))

    fig.add_trace(go.Scatter(
        x=[passes_per_seq],
        y=[direct_speed],
        mode="markers+text",
        marker=dict(size=18, color=OPTA_BG, line=dict(width=3, color=OPTA_GREEN)),
        text=["<b>NOSTRA SQUADRA</b>"],
        textposition="top center",
        textfont=dict(color=OPTA_TEXT, size=11),
        hoverinfo="text",
        hovertext=f"Passaggi / Sequenza: <b>{passes_per_seq:.2f}</b><br>Direct Speed: <b>{direct_speed:.2f} m/s</b>"
    ))

    fig.update_layout(
        xaxis=dict(title=dict(text="Passaggi Medi per Sequenza", font=dict(color=OPTA_MUTED, size=11)), range=[1.2, 5.2], tickfont=dict(color=OPTA_MUTED, size=10), gridcolor=OPTA_BORDER),
        yaxis=dict(title=dict(text="Velocità Diretta d'Avanzamento (Direct Speed, m/s)", font=dict(color=OPTA_MUTED, size=11)), range=[0.6, 2.7], tickfont=dict(color=OPTA_MUTED, size=10), gridcolor=OPTA_BORDER),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=OPTA_BG,
        margin=dict(l=35, r=20, t=25, b=35)
    )
    return fig

def plot_field_tilt_comparison(field_tilt: float, nominal_poss: float) -> go.Figure:
    """Punto C: Confronto tra Possesso Nominale e Field Tilt (Quota Ultimi 35m)."""
    fig = go.Figure()

    labels = ["Possesso Nominale (Gara)", "Field Tilt (Ultimi 35m)"]
    our_values = [nominal_poss, field_tilt]
    opp_values = [round(100 - nominal_poss, 1), round(100 - field_tilt, 1)]

    fig.add_trace(go.Bar(
        y=labels,
        x=our_values,
        name="Nostra Squadra",
        orientation='h',
        marker=dict(color=OPTA_GREEN, line=dict(color=OPTA_CARD, width=1.5)),
        text=[f"<b>{v}%</b>" for v in our_values],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color="#121212", size=11, family="Arial")
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=opp_values,
        name="Avversari",
        orientation='h',
        marker=dict(color="#334155", line=dict(color=OPTA_CARD, width=1.5)),
        text=[f"{v}%" for v in opp_values],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color=OPTA_MUTED, size=10, family="Arial")
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(color=OPTA_TEXT, size=11)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=OPTA_MUTED, size=10), orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=25, b=10),
        height=180
    )
    return fig

# =====================================================================
# 3. VIEW PRINCIPALE STREAMLIT DEL MODULO 2
# =====================================================================

def render_style_profiler_view(events_df: pd.DataFrame, team_name: str):
    if events_df.empty:
        st.warning("Nessun evento disponibile per calcolare lo stile di gioco.")
        return

    metrics = compute_team_style_metrics(events_df, team_name, sample_size=5)
    score_data = compute_tactical_scores(metrics)

    # Header Card
    header_html = textwrap.dedent(f"""
        <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-left: 4px solid {OPTA_GREEN}; border-radius: 4px; padding: 18px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">OPTA STYLE PROFILE</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: {OPTA_TEXT}; margin: 4px 0;">{metrics['archetype']}</div>
                    <div style="font-size: 0.78rem; color: {OPTA_MUTED};">Campione: {metrics['matches_analyzed']} gare ufficiali | Dimensioni campo: 105m &times; 68m</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.72rem; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">OVERALL RATING</div>
                    <div style="font-size: 2rem; font-weight: 800; color: {OPTA_GREEN}; line-height: 1.1;">{score_data['overall_rating']}<span style="font-size: 1rem; color: {OPTA_MUTED}; font-weight: 400;">/100</span></div>
                </div>
            </div>
        </div>
    """).strip()
    st.markdown(header_html, unsafe_allow_html=True)

    # 4 Macro-KPI Box
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 16px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase;">FIELD TILT (TERRITORIO)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {OPTA_GREEN}; margin: 4px 0;">{metrics['field_tilt']}%</div>
                <div style="font-size: 0.75rem; color: {OPTA_MUTED};">Possesso nominale: {metrics['nominal_possession']}%</div>
            </div>
        """).strip(), unsafe_allow_html=True)

    with c2:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 16px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase;">EXPECTED THREAT (xT)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {OPTA_TEXT}; margin: 4px 0;">+{metrics['xt_p90']}</div>
                <div style="font-size: 0.75rem; color: {OPTA_GREEN};">Minaccia generata / 90m</div>
            </div>
        """).strip(), unsafe_allow_html=True)

    with c3:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 16px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase;">DIRECT SPEED MANOVRA</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {OPTA_TEXT}; margin: 4px 0;">{metrics['direct_speed_ms']} <span style="font-size: 0.9rem; color: {OPTA_MUTED};">m/s</span></div>
                <div style="font-size: 0.75rem; color: {OPTA_MUTED};">{metrics['avg_passes_per_sequence']} passaggi per sequenza</div>
            </div>
        """).strip(), unsafe_allow_html=True)

    with c4:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 16px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase;">COSTRUZIONE BASSA (X &le; 35m)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {OPTA_TEXT}; margin: 4px 0;">{metrics['buildup_score']}<span style="font-size: 0.9rem; color: {OPTA_MUTED};">/100</span></div>
                <div style="font-size: 0.75rem; color: {OPTA_GREEN};">Precisione uscita: {metrics['buildup_accuracy']}%</div>
            </div>
        """).strip(), unsafe_allow_html=True)

    st.write("")

    # Card Scoreboard Sintetica
    scoreboard_html = textwrap.dedent(f"""
        <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; margin-bottom: 20px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 15px;">PERFORMANCE BREAKDOWN (0–100)</div>
    """).strip()
    st.markdown(scoreboard_html, unsafe_allow_html=True)
    
    col_score_l, col_score_r = st.columns([1.2, 1])
    with col_score_l:
        for phase_name, score_val in score_data["scores"].items():
            color_bar = OPTA_GREEN if score_val >= 70 else (OPTA_ORANGE if score_val >= 50 else "#ff5252")
            st.markdown(textwrap.dedent(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; color: {OPTA_TEXT}; margin-bottom: 3px;">
                    <span>{phase_name}</span>
                    <span style="color: {color_bar}; font-weight: 700;">{score_val}/100</span>
                </div>
            """).strip(), unsafe_allow_html=True)
            st.progress(int(score_val))
            st.write("")

    with col_score_r:
        takeaway_html = textwrap.dedent(f"""
            <div style="background-color: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-left: 3px solid {OPTA_GREEN}; border-radius: 4px; padding: 14px; margin-bottom: 10px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_GREEN}; text-transform: uppercase;">FASE DI DOMINANZA TATTICA</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {OPTA_TEXT}; margin: 4px 0;">{score_data['strength'][0]} ({score_data['strength'][1]}/100)</div>
                <div style="font-size: 0.78rem; color: {OPTA_MUTED};">Massima efficienza e stabilità registrata nel campione.</div>
            </div>
            <div style="background-color: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-left: 3px solid {OPTA_ORANGE}; border-radius: 4px; padding: 14px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_ORANGE}; text-transform: uppercase;">AREA CRITICA DA POTENZIARE</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {OPTA_TEXT}; margin: 4px 0;">{score_data['weakness'][0]} ({score_data['weakness'][1]}/100)</div>
                <div style="font-size: 0.78rem; color: {OPTA_TEXT}; margin-top: 4px;"><b>Indicazione Operativa:</b> {score_data['prescription']}</div>
            </div>
        """).strip()
        st.markdown(takeaway_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================
    # RIGA AFFIANCATA 1: RADAR WHEEL & SHOT MAP (MPLSOCCER)
    # =====================================================================
    col_wheel, col_shots = st.columns([1.1, 1.1])

    with col_wheel:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">OPTA TACTICAL WHEEL (7 PARAMETRI)</div>
                <div style="font-size: 0.8rem; color: {OPTA_MUTED}; margin-bottom: 10px;">Distribuzione percentile normalizzata del modello di gioco</div>
        """).strip(), unsafe_allow_html=True)
        fig_wheel = plot_opta_style_wheel(metrics)
        st.plotly_chart(fig_wheel, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_shots:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">SHOT MAP & QUALITÀ CONCLUSIONI</div>
                    <div style="font-size: 0.75rem; color: {OPTA_GREEN};">xG/Tiro: {metrics['xg_per_shot']}</div>
                </div>
                <div style="font-size: 0.8rem; color: {OPTA_MUTED}; margin-bottom: 10px;">Verde: Gol | Blu: Parato/Respinto | Grigio: Fuori. Area proporzionale a xG.</div>
        """).strip(), unsafe_allow_html=True)
        fig_shots = plot_shot_map_mplsoccer(metrics["shots_df"], team_name)
        st.pyplot(fig_shots, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # =====================================================================
    # SEZIONE PUNTO A: EXPECTED THREAT (xT) FLOW INTERATTIVO PLOTLY
    # =====================================================================
    st.markdown(textwrap.dedent(f"""
        <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">PUNTO A: EXPECTED THREAT (xT) HEATMAP & TRACCIATURA VETTORI CHIAVE</div>
                <div style="font-size: 0.72rem; color: {OPTA_GREEN}; font-weight: 700;">LIVELLO INTERATTIVO PROPORZIONATO</div>
            </div>
            <div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 14px;">
                Pitch vettoriale $105\\text{{m}} \\times 68\\text{{m}}$ a rapporto fisso. Cliccare sulla legenda per isolare la heatmap o i vettori. Passare il mouse sulle frecce per visualizzare autore e valore netto di &Delta;xT generato.
            </div>
    """).strip(), unsafe_allow_html=True)

    fig_xt = plot_xt_threat_flow(metrics["df_high_threat"], team_name)
    st.plotly_chart(fig_xt, width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # =====================================================================
    # RIGA AFFIANCATA 2: PUNTI B & C (MATRICE OPTA + FIELD TILT)
    # =====================================================================
    col_mat, col_tilt = st.columns([1.1, 1.1])

    with col_mat:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">PUNTO B: MATRICE STILE DI POSSESSO OPTA</div>
                    <div style="font-size: 0.72rem; color: {OPTA_GREEN}; font-weight: 700;">4 QUADRANTI</div>
                </div>
                <div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 12px;">Classificazione tra velocità diretta verso la porta (m/s) e passaggi medi per sequenza.</div>
        """).strip(), unsafe_allow_html=True)
        fig_matrix = plot_opta_style_matrix(metrics["direct_speed_ms"], metrics["avg_passes_per_sequence"])
        st.plotly_chart(fig_matrix, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with col_tilt:
        st.markdown(textwrap.dedent(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">PUNTO C: CONTROLLO TERRITORIALE (FIELD TILT)</div>
                    <div style="font-size: 0.72rem; color: {OPTA_GREEN}; font-weight: 700;">ULTIMI 35m</div>
                </div>
                <div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 10px;">Rapporto percentuale tra i nostri passaggi nella trequarti offensiva e quelli concessi all'avversario.</div>
        """).strip(), unsafe_allow_html=True)
        fig_tilt = plot_field_tilt_comparison(metrics["field_tilt"], metrics["nominal_possession"])
        st.plotly_chart(fig_tilt, width='stretch')

        st.markdown(textwrap.dedent(f"""
            <div style="margin-top: 15px; border-top: 1px solid {OPTA_BORDER}; padding-top: 12px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; margin-bottom: 8px;">Ripartizione Canali d'Invasione</div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                    <div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 8px;">
                        <div style="font-size: 0.68rem; color: {OPTA_MUTED};">Fascia Sinistra</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: {OPTA_TEXT};">{metrics['channel_dist']['Corsia Sinistra']}%</div>
                    </div>
                    <div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 8px;">
                        <div style="font-size: 0.68rem; color: {OPTA_MUTED};">Corridoio Centrale</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: {OPTA_GREEN};">{metrics['channel_dist']['Corridoio Centrale']}%</div>
                    </div>
                    <div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 8px;">
                        <div style="font-size: 0.68rem; color: {OPTA_MUTED};">Fascia Destra</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: {OPTA_TEXT};">{metrics['channel_dist']['Corsia Destra']}%</div>
                    </div>
                </div>
            </div>
        """).strip(), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)