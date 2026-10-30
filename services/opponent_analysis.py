import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mplsoccer import Pitch, VerticalPitch
import textwrap
from typing import Dict, Any
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.validation import make_valid

from services.team_tactics import (
    get_dominant_formation_and_best_lineup,
    compute_lineup_passing_network,
    plot_best_xi_network,
    plot_team_voronoi_spacing
)

from services.tactical_simulator import (
    simulate_tactical_clash,
    simulate_tactical_clash_cached,
    get_available_team_formations,
    get_formation_line_groups,
    get_role_candidates,
    export_tactical_preset,
    parse_tactical_preset,
    FORMATION_TEMPLATES
)

from services.tactical_agent import generate_tactical_training_guidelines


# Costanti di conversione metriche FIFA 105m x 68m su coordinate StatsBomb (120yd x 80yd)[cite: 8]
CONV_X = 105.0 / 120.0
CONV_Y = 68.0 / 80.0
CONV_DIST = 0.9144

# Palette Opta Dark Uniformata[cite: 8]
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

# =====================================================================
# 1. ESTRAZIONE PROFILO COMPLETO SQUADRA (CAMPIONE 5 GARE)
# =====================================================================

def extract_team_profile(events_df: pd.DataFrame, team_name: str, sample_size: int = 5) -> Dict[str, Any]:
    team_full_ev = events_df[events_df["team"] == team_name].copy()
    available_matches = team_full_ev["match_id"].dropna().unique().tolist()
    sampled_matches = available_matches[:sample_size]
    n_matches = max(len(sampled_matches), 1)

    t_ev = team_full_ev[team_full_ev["match_id"].isin(sampled_matches)].copy()
    opp_ev = events_df[
        (events_df["match_id"].isin(sampled_matches)) & 
        (events_df["team"].notna()) & 
        (events_df["team"] != team_name)
    ].copy()

    # Coordinate metriche[cite: 8]
    t_ev["x_m"] = t_ev["x"] * CONV_X
    t_ev["y_m"] = t_ev["y"] * CONV_Y
    t_ev["end_x_m"] = t_ev["pass_end_x"] * CONV_X
    t_ev["end_y_m"] = t_ev["pass_end_y"] * CONV_Y

    opp_ev["x_m"] = opp_ev["x"] * CONV_X
    opp_ev["y_m"] = opp_ev["y"] * CONV_Y

    # Costruzione & Possesso[cite: 8]
    passes = t_ev[t_ev["type"] == "Pass"].copy()
    comp_passes = passes[passes["pass_outcome"].isna()].copy()
    total_passes = len(passes)
    pass_acc = (len(comp_passes) / total_passes * 100) if total_passes > 0 else 0.0

    raw_len = float(comp_passes["pass_length"].mean()) if "pass_length" in comp_passes.columns else 20.0
    avg_pass_len_m = raw_len * CONV_DIST

    comp_passes["dx_m"] = comp_passes["end_x_m"] - comp_passes["x_m"]
    prog_passes = comp_passes[comp_passes["dx_m"] >= 9.14]
    prog_ratio = (len(prog_passes) / len(comp_passes) * 100) if len(comp_passes) > 0 else 0.0

    # Build-up (X <= 35m)[cite: 8]
    buildup_passes = passes[passes["x_m"] <= 35.0]
    buildup_comp = comp_passes[comp_passes["x_m"] <= 35.0]
    buildup_passes_p90 = len(buildup_passes) / n_matches
    buildup_acc = (len(buildup_comp) / len(buildup_passes) * 100) if not buildup_passes.empty else 0.0
    score_buildup = 0.5 * normalize_metric(buildup_passes_p90, 80.0, 220.0) + 0.5 * normalize_metric(buildup_acc, 75.0, 92.0)

    # Tiri & Pericolosità[cite: 8]
    shots = t_ev[t_ev["type"] == "Shot"].copy()
    total_shots = len(shots)
    total_xg = float(shots["shot_statsbomb_xg"].sum()) if "shot_statsbomb_xg" in shots.columns else 0.0
    xg_per_shot = (total_xg / total_shots) if total_shots > 0 else 0.0

    # Difesa & Pressing[cite: 8]
    def_actions = t_ev[t_ev["type"].isin(["Pressure", "Tackle", "Interception", "Foul Committed", "Block"])].copy()
    raw_def_x = float(def_actions["x"].mean()) if not def_actions.empty else 50.0
    def_height_m = raw_def_x * CONV_X

    opp_passes_buildup = opp_ev[
        (opp_ev["type"] == "Pass") & 
        (opp_ev["pass_outcome"].isna()) & 
        (opp_ev["x"] * CONV_X <= 70.0)
    ]
    def_actions_high = def_actions[def_actions["x_m"] >= 35.0]
    ppda = (len(opp_passes_buildup) / len(def_actions_high)) if len(def_actions_high) > 0 else 0.0
    high_recov = len(t_ev[(t_ev["type"].isin(["Ball Recovery", "Interception"])) & (t_ev["x_m"] >= 70.0)])

    # Canali Trequarti (X >= 70m)[cite: 8]
    final_third_comp = comp_passes[comp_passes["end_x_m"] >= 70.0]
    l_ch = len(final_third_comp[final_third_comp["end_y_m"] < 22.6])
    c_ch = len(final_third_comp[final_third_comp["end_y_m"].between(22.6, 45.4)])
    r_ch = len(final_third_comp[final_third_comp["end_y_m"] > 45.4])
    tot_third = max(l_ch + c_ch + r_ch, 1)

    # Ingressi & Cross[cite: 8]
    if "pass_cross" in passes.columns:
        passes["is_cross"] = passes["pass_cross"].apply(_is_true)
    else:
        passes["is_cross"] = False

    box_p = passes[(passes["end_x_m"] >= 88.5) & (passes["end_y_m"].between(13.84, 54.16)) & (passes["x_m"] < 88.5)]
    cross_p = box_p[box_p["is_cross"]]
    cross_ratio = (len(cross_p) / len(box_p) * 100) if len(box_p) > 0 else 0.0

    # Calci d'Angolo Avanzati[cite: 8]
    corners_for = t_ev[(t_ev["type"] == "Pass") & (t_ev["pass_type"] == "Corner")].copy()
    corners_against = opp_ev[(opp_ev["type"] == "Pass") & (opp_ev["pass_type"] == "Corner")].copy()
    tot_c_for = max(len(corners_for), 1)

    inswing = len(corners_for[corners_for["pass_technique"].astype(str).str.contains("Inswinging", na=False)])
    outswing = len(corners_for[corners_for["pass_technique"].astype(str).str.contains("Outswinging", na=False)])
    straight = max(len(corners_for) - (inswing + outswing), 0)

    if not corners_for.empty and "pass_length" in corners_for.columns:
        c_lengths_m = corners_for["pass_length"] * CONV_DIST
        c_short = len(c_lengths_m[c_lengths_m < 18.0])
        c_box = len(c_lengths_m[(c_lengths_m >= 18.0) & (c_lengths_m <= 35.0)])
        c_deep = len(c_lengths_m[c_lengths_m > 35.0])
    else:
        c_short, c_box, c_deep = int(0.15 * tot_c_for), int(0.70 * tot_c_for), int(0.15 * tot_c_for)

    corner_gittata = {
        "Corti (<18m)": round((c_short / tot_c_for) * 100, 1),
        "In Area (18-35m)": round((c_box / tot_c_for) * 100, 1),
        "Oltre Area (>35m)": round((c_deep / tot_c_for) * 100, 1)
    }

    set_piece_patterns = ["From Corner", "From Free Kick", "From Throw In"]
    sp_shots = shots[shots["play_pattern"].isin(set_piece_patterns)]
    sp_xg_p90 = float(sp_shots["shot_statsbomb_xg"].sum()) / n_matches if "shot_statsbomb_xg" in sp_shots.columns else 0.0

    opp_shots = opp_ev[opp_ev["type"] == "Shot"].copy()
    opp_sp_shots = opp_shots[opp_shots["play_pattern"].isin(set_piece_patterns)]
    opp_sp_xg_p90 = float(opp_sp_shots["shot_statsbomb_xg"].sum()) / n_matches if "shot_statsbomb_xg" in opp_sp_shots.columns else 0.0

    # Metronomo e Verticalizzatore[cite: 8]
    pass_counts = comp_passes.groupby("player").size().sort_values(ascending=False)
    top_passer = pass_counts.index[0] if not pass_counts.empty else "N/D"

    prog_counts = prog_passes.groupby("player").size().sort_values(ascending=False)
    top_progressor = prog_counts.index[0] if not prog_counts.empty else "N/D"

    # Vulnerabilità: Palle perse nella propria metà campo (X <= 52.5m)[cite: 8]
    lost_passes = passes[passes["pass_outcome"].notna() & (passes["x_m"] <= 52.5)]
    lost_dribbles = t_ev[(t_ev["type"] == "Dribble") & (t_ev["dribble_outcome"].astype(str).str.lower() == "incomplete") & (t_ev["x_m"] <= 52.5)]
    disposs = t_ev[(t_ev["type"].isin(["Dispossessed", "Error"])) & (t_ev["x_m"] <= 52.5)]
    turnovers_own_half = pd.concat([lost_passes, lost_dribbles, disposs], ignore_index=True)

    return {
        "team_name": team_name,
        "matches_analyzed": n_matches,
        "passes_p90": round(total_passes / n_matches, 1),
        "pass_accuracy": round(pass_acc, 1),
        "avg_pass_length_m": round(avg_pass_len_m, 1),
        "progressive_ratio": round(prog_ratio, 1),
        "buildup_score": round(score_buildup, 1),
        "buildup_accuracy": round(buildup_acc, 1),
        "shots_p90": round(total_shots / n_matches, 1),
        "xg_p90": round(total_xg / n_matches, 2),
        "xg_per_shot": round(xg_per_shot, 3),
        "cross_ratio": round(cross_ratio, 1),
        "channels": {
            "Sinistra": round((l_ch / tot_third) * 100, 1),
            "Centro": round((c_ch / tot_third) * 100, 1),
            "Destra": round((r_ch / tot_third) * 100, 1)
        },
        "ppda": round(ppda, 2),
        "defensive_height_m": round(def_height_m, 1),
        "high_recoveries_p90": round(high_recov / n_matches, 1),
        "corners_for_p90": round(len(corners_for) / n_matches, 1),
        "corners_against_p90": round(len(corners_against) / n_matches, 1),
        "corner_trajectories": {"Rientrante": inswing, "Uscente": outswing, "Diretta": straight},
        "corner_gittata": corner_gittata,
        "sp_xg_for_p90": round(sp_xg_p90, 2),
        "sp_xg_against_p90": round(opp_sp_xg_p90, 2),
        "top_passer": top_passer,
        "top_progressor": top_progressor,
        "turnovers_own_half": turnovers_own_half,
        "shots_conceded": opp_shots
    }

# =====================================================================
# 2. GRAFICI: HEAD-TO-HEAD, DUAL CLASH, VULNERABILITÀ & SET PIECES
# =====================================================================

def plot_head_to_head_radar(p_our: dict, p_opp: dict) -> go.Figure:
    categories = [
        "Fase Costruzione",
        "Volume Possesso",
        "Verticalità",
        "Qualità Tiro",
        "Intensità Pressing",
        "Altezza Baricentro",
        "Recuperi Alti"
    ]

    def get_scores(p):
        return [
            p["buildup_score"],
            np.clip((p["passes_p90"] - 300) / 400 * 100, 10, 95),
            np.clip(p["progressive_ratio"] * 2.5, 10, 95),
            np.clip(p["xg_per_shot"] / 0.18 * 100, 10, 95),
            np.clip((25.0 - p["ppda"]) / 18.0 * 100, 10, 95),
            np.clip((p["defensive_height_m"] - 31) / 22 * 100, 10, 95),
            np.clip(p["high_recoveries_p90"] / 6.0 * 100, 10, 95)
        ]

    s_our = get_scores(p_our)
    s_opp = get_scores(p_opp)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=s_our + [s_our[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 255, 133, 0.22)',
        line=dict(color=OPTA_GREEN, width=2.5),
        name=f"{p_our['team_name']} (Nostra)",
        hoverinfo='name+r'
    ))

    fig.add_trace(go.Scatterpolar(
        r=s_opp + [s_opp[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(255, 87, 34, 0.22)',
        line=dict(color=OPTA_ORANGE, width=2.5),
        name=f"{p_opp['team_name']} (Avversaria)",
        hoverinfo='name+r'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=OPTA_MUTED, size=9), gridcolor=OPTA_LINE),
            angularaxis=dict(tickfont=dict(size=10, color=OPTA_TEXT), linecolor=OPTA_LINE, gridcolor=OPTA_LINE),
            bgcolor=OPTA_CARD
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=45, t=25, b=25),
        legend=dict(font=dict(color=OPTA_TEXT, size=10), orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig

def plot_channel_comparison(ch_our: dict, ch_opp: dict, our_name: str, opp_name: str) -> go.Figure:
    fig = go.Figure()
    zones = ["Sinistra", "Centro", "Destra"]
    v_our = [ch_our[z] for z in zones]
    v_opp = [ch_opp[z] for z in zones]

    fig.add_trace(go.Bar(
        name=our_name,
        x=zones,
        y=v_our,
        marker=dict(color=OPTA_GREEN, line=dict(color='#ffffff', width=0.8))
    ))
    fig.add_trace(go.Bar(
        name=opp_name,
        x=zones,
        y=v_opp,
        marker=dict(color=OPTA_ORANGE, line=dict(color='#ffffff', width=0.8))
    ))

    fig.update_layout(
        barmode='group',
        xaxis=dict(tickfont=dict(color=OPTA_TEXT, size=11), gridcolor=OPTA_BORDER),
        yaxis=dict(title=dict(text="% Ingressi Negli Ultimi 35m", font=dict(color=OPTA_MUTED, size=11)), tickfont=dict(color=OPTA_MUTED, size=10), gridcolor=OPTA_BORDER),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=OPTA_BG,
        margin=dict(l=35, r=15, t=20, b=30),
        legend=dict(font=dict(color=OPTA_TEXT, size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def _compute_bounded_voronoi(pts: np.ndarray, bbox_width: float = 105.0, bbox_height: float = 68.0):
    n_pts = len(pts)
    if n_pts == 0:
        return []

    pts_clean = np.copy(pts)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if np.allclose(pts_clean[i], pts_clean[j], atol=1e-3):
                pts_clean[j] += np.random.uniform(0.01, 0.03, size=2)

    pitch_box = box(0.0, 0.0, bbox_width, bbox_height)

    pts_left = np.copy(pts_clean); pts_left[:, 0] = -pts_left[:, 0]
    pts_right = np.copy(pts_clean); pts_right[:, 0] = 2.0 * bbox_width - pts_right[:, 0]
    pts_bottom = np.copy(pts_clean); pts_bottom[:, 1] = -pts_bottom[:, 1]
    pts_top = np.copy(pts_clean); pts_top[:, 1] = 2.0 * bbox_height - pts_top[:, 1]

    pts_extended = np.vstack([pts_clean, pts_left, pts_right, pts_bottom, pts_top])
    vor = Voronoi(pts_extended)

    polygons = []
    for i in range(n_pts):
        region_idx = vor.point_region[i]
        vert_indices = vor.regions[region_idx]
        
        if -1 not in vert_indices and len(vert_indices) >= 3:
            poly_coords = vor.vertices[vert_indices]
            raw_poly = Polygon(poly_coords)
            if not raw_poly.is_valid:
                raw_poly = make_valid(raw_poly)

            clipped = raw_poly.intersection(pitch_box)
            polygons.append(clipped)
        else:
            polygons.append(Polygon())

    return polygons

def plot_tactical_zones_clash(
    zone_matrix: np.ndarray,
    nodes_our: pd.DataFrame, 
    nodes_opp: pd.DataFrame, 
    our_name: str, 
    opp_name: str
):
    """
    Visualizza la Matrice Tattica Zonale a 15 Zone (3x5) con sovrapposizione dei nodi reali[cite: 8].
    Colorazione ad alto contrasto:
    - Verde (#00ff85): Superiorità nostra / Overload[cite: 8]
    - Neutro (#1a232c): Parità / Duello 1v1[cite: 8]
    - Arancione (#ff5722): Densità avversaria[cite: 8]
    """
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color=OPTA_BG,
        line_color=OPTA_LINE,
        linewidth=1.2,
        goal_type='box'
    )
    fig, ax = pitch.draw(figsize=(12, 7.5))
    fig.patch.set_facecolor(OPTA_CARD)

    x_bins_m = [0.0, 35.0, 70.0, 105.0]
    y_bins_m = [0.0, 13.6, 27.2, 40.8, 54.4, 68.0]

    x_bins_yd = [x / CONV_X for x in x_bins_m]
    y_bins_yd = [y / CONV_Y for y in y_bins_m]

    colors = ["#ff5722", "#131a22", "#00ff85"]
    cmap_zone = LinearSegmentedColormap.from_list("clash_cmap", colors, N=256)

    for i in range(3):
        for j in range(5):
            val = zone_matrix[i, j]
            norm_val = np.clip((val + 0.6) / 1.2, 0.0, 1.0)
            cell_color = cmap_zone(norm_val)

            x0, x1 = x_bins_yd[i], x_bins_yd[i+1]
            y0, y1 = y_bins_yd[j], y_bins_yd[j+1]

            ax.fill_between(
                [x0, x1], y0, y1,
                facecolor=cell_color,
                alpha=0.36,
                edgecolor="#2c3845",
                linewidth=1.0,
                zorder=1
            )

            delta_label = f"+{val:.2f}" if val > 0.05 else (f"{val:.2f}" if val < -0.05 else "1v1")
            text_col = OPTA_GREEN if val > 0.15 else (OPTA_ORANGE if val < -0.15 else "#8b949e")
            ax.text(
                (x0 + x1) / 2.0, (y0 + y1) / 2.0,
                delta_label,
                color=text_col,
                fontsize=8.5,
                ha='center',
                va='center',
                fontweight='bold',
                alpha=0.85,
                zorder=2
            )

    for y_val in y_bins_yd[1:-1]:
        ax.plot([0, 120], [y_val, y_val], color="#485969", linestyle=":", linewidth=0.9, alpha=0.6, zorder=2)
    for x_val in x_bins_yd[1:-1]:
        ax.plot([x_val, x_val], [0, 80], color="#485969", linestyle=":", linewidth=0.9, alpha=0.6, zorder=2)

    pts_our_x = nodes_our["x_mean"].values / CONV_X
    pts_our_y = nodes_our["y_mean"].values / CONV_Y
    pitch.scatter(pts_our_x, pts_our_y, ax=ax, s=260, color=OPTA_CARD, edgecolors=OPTA_GREEN, linewidth=2.2, zorder=4)

    for idx, r in nodes_our.iterrows():
        sigla = str(r.get("role", "MID"))
        sur = str(r.get("player", ""))
        px, py = pts_our_x[idx], pts_our_y[idx]
        ax.text(px, py, sigla, color=OPTA_TEXT, fontsize=7.5, ha='center', va='center', fontweight='bold', zorder=5)
        if sur and sur != sigla:
            ax.text(
                px, py - 3.2, sur,
                color=OPTA_GREEN, fontsize=6.8, ha='center', va='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.15", facecolor=OPTA_CARD, edgecolor=OPTA_GREEN, alpha=0.85, linewidth=0.6),
                zorder=5
            )

    pts_opp_x = nodes_opp["x_mean"].values / CONV_X
    pts_opp_y = nodes_opp["y_mean"].values / CONV_Y
    pitch.scatter(pts_opp_x, pts_opp_y, ax=ax, s=260, color=OPTA_CARD, edgecolors=OPTA_ORANGE, linewidth=2.2, zorder=4)

    for idx, r in nodes_opp.iterrows():
        sigla = str(r.get("role", "MID"))
        sur = str(r.get("player", ""))
        px, py = pts_opp_x[idx], pts_opp_y[idx]
        ax.text(px, py, sigla, color=OPTA_TEXT, fontsize=7.5, ha='center', va='center', fontweight='bold', zorder=5)
        if sur and sur != sigla:
            ax.text(
                px, py - 3.2, sur,
                color=OPTA_ORANGE, fontsize=6.8, ha='center', va='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.15", facecolor=OPTA_CARD, edgecolor=OPTA_ORANGE, alpha=0.85, linewidth=0.6),
                zorder=5
            )

    ax.set_title(
        f"TACTICAL MATCHUP MATRIX (15 ZONE)\n"
        f"Verde: Vantaggio {our_name.upper()} | Grigio: Duelli 1v1 | Arancione: Densità {opp_name.upper()}",
        color=OPTA_TEXT,
        fontsize=11.0,
        pad=14,
        fontweight='bold'
    )

    return fig

def plot_mplsoccer_opponent_turnovers(turnovers_df: pd.DataFrame, team_name: str):
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color=OPTA_BG,
        line_color=OPTA_LINE,
        linewidth=1.2,
        goal_type='box'
    )
    fig, ax = pitch.draw(figsize=(11, 7.0))
    fig.patch.set_facecolor(OPTA_CARD)

    if turnovers_df is None or turnovers_df.empty:
        ax.text(60, 40, f"Nessun turnover registrato per {team_name}", color=OPTA_TEXT, fontsize=12, ha='center', va='center', fontweight='bold')
        return fig

    x_col = "x" if "x" in turnovers_df.columns else ("location_x" if "location_x" in turnovers_df.columns else None)
    y_col = "y" if "y" in turnovers_df.columns else ("location_y" if "location_y" in turnovers_df.columns else None)

    if x_col is None or y_col is None:
        ax.text(60, 40, "Coordinate non disponibili nel DataFrame", color=OPTA_TEXT, fontsize=12, ha='center', va='center')
        return fig

    df_valid = turnovers_df[turnovers_df[x_col].notna() & turnovers_df[y_col].notna()].copy()
    defensive_turnovers = df_valid[df_valid[x_col] <= 60]
    attacking_turnovers = df_valid[df_valid[x_col] > 60]

    if not defensive_turnovers.empty:
        pitch.scatter(
            defensive_turnovers[x_col], defensive_turnovers[y_col],
            ax=ax, s=120, color="#ff5722", edgecolors="#ffffff", linewidth=1.2, alpha=0.85, zorder=4,
            label=f"Propria metà campo ({len(defensive_turnovers)})"
        )

    if not attacking_turnovers.empty:
        pitch.scatter(
            attacking_turnovers[x_col], attacking_turnovers[y_col],
            ax=ax, s=90, color="#ffb300", edgecolors=OPTA_BG, linewidth=1.0, alpha=0.75, zorder=3,
            label=f"Metà campo avversaria ({len(attacking_turnovers)})"
        )

    ax.axvline(60, color="#ffffff", linestyle="--", linewidth=1.0, alpha=0.35, zorder=2)
    ax.legend(facecolor=OPTA_CARD, edgecolor=OPTA_LINE, labelcolor=OPTA_TEXT, fontsize=9, loc="upper left")

    ax.set_title(
        f"MAPPA TURNOVERS & PALLE PERSE: {team_name.upper()}\n"
        f"Totale registrate: {len(df_valid)} (Pericolose in prima costruzione: {len(defensive_turnovers)})",
        color=OPTA_TEXT, fontsize=11, pad=12, fontweight='bold'
    )
    return fig

def plot_mplsoccer_opponent_shots_conceded(shots_df: pd.DataFrame, opp_name: str) -> plt.Figure:
    pitch = VerticalPitch(
        pitch_type='statsbomb',
        half=True,
        pitch_color=OPTA_BG,
        line_color=OPTA_LINE,
        linewidth=1.2,
        goal_type='box'
    )
    fig, ax = pitch.draw(figsize=(8, 5.8))
    fig.patch.set_facecolor(OPTA_CARD)

    if not shots_df.empty:
        for _, s in shots_df.iterrows():
            sx = s["x"]
            sy = s["y"]
            xg = float(s.get("shot_statsbomb_xg", 0.08))
            outcome = str(s.get("shot_outcome", "")).lower()

            size = max(xg * 650, 40)
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

    ax.set_title(
        f"TIRI E GOL CONCESSI DALL'AVVERSARIO (xG) — {opp_name.upper()}",
        color=OPTA_TEXT, fontsize=10, pad=10, fontweight='bold'
    )
    return fig

# =====================================================================
# SET PIECES INTELLIGENCE: ESTRAZIONE, CLUSTERING & PITCH
# =====================================================================

def extract_set_pieces_trajectories(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Estrae e normalizza le traiettorie dei piazzati offensivi:
    - Tutti i Calci d'Angolo (Corner).
    - SOLO i Calci di Punizione (Free Kick) il cui secondo tocco atterra all'interno dell'area di rigore.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame()

    target_clean = str(team_name).strip().lower()
    team_col = "team" if "team" in events_df.columns else ("team_name" if "team_name" in events_df.columns else None)
    if not team_col:
        return pd.DataFrame()

    t_ev = events_df[events_df[team_col].astype(str).str.strip().str.lower() == target_clean].copy()
    if t_ev.empty:
        t_ev = events_df[events_df[team_col].astype(str).str.strip().str.lower().str.contains(target_clean.split()[0], na=False)].copy()

    if t_ev.empty or "type" not in t_ev.columns:
        return pd.DataFrame()

    sp_mask = (t_ev["type"] == "Pass") & (t_ev.get("pass_type", "").isin(["Corner", "Free Kick"]))
    sp_ev = t_ev[sp_mask].copy()

    if sp_ev.empty:
        return pd.DataFrame()

    # Normalizzazione coordinate di partenza
    if "x" not in sp_ev.columns and "location" in sp_ev.columns:
        sp_ev["x"] = sp_ev["location"].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else np.nan)
        sp_ev["y"] = sp_ev["location"].apply(lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else np.nan)

    # Normalizzazione coordinate di arrivo (secondo tocco)
    if "pass_end_x" not in sp_ev.columns and "pass_end_location" in sp_ev.columns:
        sp_ev["pass_end_x"] = sp_ev["pass_end_location"].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else np.nan)
        sp_ev["pass_end_y"] = sp_ev["pass_end_location"].apply(lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else np.nan)

    valid_sp = sp_ev[sp_ev["x"].notna() & sp_ev["pass_end_x"].notna()].copy()
    if valid_sp.empty:
        return pd.DataFrame()

    records = []
    for _, row in valid_sp.iterrows():
        x0, y0 = float(row["x"]), float(row["y"])
        x1, y1 = float(row["pass_end_x"]), float(row["pass_end_y"])
        p_type = str(row.get("pass_type", "Corner")).strip()

        # Coordinate Area di Rigore Avversaria (StatsBomb 120x80): X >= 102.0, 18.0 <= Y <= 62.0
        is_in_box = (x1 >= 102.0) and (18.0 <= y1 <= 62.0)

        # REQUISITO RESTRITTIVO: Le punizioni NON da corner vengono tenute SOLO se cadono in area
        if p_type == "Free Kick" and not is_in_box:
            continue

        dx_m = (x1 - x0) * CONV_X
        dy_m = (y1 - y0) * CONV_Y
        dist_m = np.sqrt(dx_m**2 + dy_m**2)

        p_tech = str(row.get("pass_technique", "Standard")).strip()
        is_inswing = "Inswinging" in p_tech
        is_outswing = "Outswinging" in p_tech

        if p_type == "Free Kick":
            trajectory_type = "FreeKick_Box"
        else:
            trajectory_type = "Inswinging" if is_inswing else ("Outswinging" if is_outswing else "Straight")

        # Riconoscimento cluster target in area
        if dist_m < 18.0 and p_type == "Corner":
            cluster_name = "Scambio Corto"
        elif y1 < 33.0 and x1 >= 104.0:
            cluster_name = "Primo Palo"
        elif 33.0 <= y1 <= 47.0 and x1 >= 100.0:
            cluster_name = "Centro / Dischetto"
        elif y1 > 47.0 and x1 >= 104.0:
            cluster_name = "Secondo Palo"
        elif x1 < 102.0 and 26.0 <= y1 <= 54.0:
            cluster_name = "Rimorchio / Edge of Box"
        else:
            cluster_name = "Generica in Area"

        is_completed = pd.isna(row.get("pass_outcome"))
        is_shot_assist = bool(row.get("pass_shot_assist") == True or row.get("pass_goal_assist") == True)

        records.append({
            "player": str(row.get("player", "N/D")),
            "pass_type": p_type,
            "x": x0,
            "y": y0,
            "pass_end_x": x1,
            "pass_end_y": y1,
            "distance_m": round(dist_m, 1),
            "trajectory": trajectory_type,
            "cluster": cluster_name,
            "completed": is_completed,
            "shot_assist": is_shot_assist
        })

    return pd.DataFrame(records)

def plot_mplsoccer_set_piece_clusters(trajectories_df: pd.DataFrame, team_name: str) -> plt.Figure:
    """
    Renderizza il mezzo campo offensivo con distinzione cromatica:
    - ROSSO (#ff2a2a): Calci di punizione con secondo tocco in area di rigore.
    - VERDE (#00ff85): Corner rientranti.
    - CIANO (#00b0ff): Corner uscenti.
    - AMBRA/ARANCIO (#ff9100): Corner tesi/diretti.
    """
    pitch = VerticalPitch(
        pitch_type='statsbomb',
        half=True,
        pitch_color=OPTA_BG,
        line_color=OPTA_LINE,
        linewidth=1.2,
        goal_type='box'
    )
    fig, ax = pitch.draw(figsize=(9, 7.5))
    fig.patch.set_facecolor(OPTA_CARD)

    if trajectories_df is None or trajectories_df.empty:
        ax.text(40, 95, f"Nessun piazzato offensivo in area per {team_name}", color=OPTA_TEXT, fontsize=11, ha='center', va='center', fontweight='bold')
        return fig

    # Palette con Rosso esclusivo per le Punizioni in Area
    traj_colors = {
        "FreeKick_Box": "#ff2a2a",  # ROSSO VIVO per Punizioni dirette in area
        "Inswinging": "#00ff85",    # Verde Fluo per Corner Rientranti
        "Outswinging": "#00b0ff",   # Ciano Elettrico per Corner Uscenti
        "Straight": "#ff9100",      # Ambra / Arancio dorato per Corner Tesi
    }

    # 1. Tracciamento traiettorie
    for _, row in trajectories_df.iterrows():
        x0, y0 = row["x"], row["y"]
        x1, y1 = row["pass_end_x"], row["pass_end_y"]
        traj = row["trajectory"]
        is_assist = row["shot_assist"]
        c_line = traj_colors.get(traj, "#ff9100")
        
        alpha_val = 0.90 if is_assist else (0.65 if traj == "FreeKick_Box" else 0.45)
        width_val = 2.4 if is_assist else (1.8 if traj == "FreeKick_Box" else 1.2)

        pitch.lines(
            x0, y0, x1, y1,
            ax=ax,
            color=c_line,
            linewidth=width_val,
            alpha=alpha_val,
            transparent=True,
            zorder=4 if traj == "FreeKick_Box" else 3
        )

        # Marcatore del secondo tocco
        pitch.scatter(
            x1, y1,
            ax=ax,
            s=85 if is_assist else 45,
            color=c_line,
            edgecolors="#ffffff" if is_assist else OPTA_BG,
            linewidth=1.2 if is_assist else 0.8,
            alpha=0.95,
            zorder=5
        )

    # 2. Centroidi dei cluster con etichetta percentuale
    cluster_counts = trajectories_df["cluster"].value_counts(normalize=True) * 100
    cluster_coords = {
        "Primo Palo": (110.0, 28.0),
        "Centro / Dischetto": (104.0, 40.0),
        "Secondo Palo": (110.0, 52.0),
        "Rimorchio / Edge of Box": (94.0, 40.0),
        "Scambio Corto": (115.0, 15.0)
    }

    for c_name, pct in cluster_counts.items():
        if c_name in cluster_coords and pct >= 5.0:
            cx, cy = cluster_coords[c_name]
            ax.text(
                cy, cx,
                f"{c_name}\n({pct:.0f}%)",
                color="#ffffff",
                fontsize=7.8,
                ha='center',
                va='center',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#18222c", edgecolor=OPTA_BORDER, alpha=0.92, linewidth=0.8),
                zorder=6
            )

    # Conteggio per tipologia per titolo
    n_fk_box = len(trajectories_df[trajectories_df["trajectory"] == "FreeKick_Box"])
    n_corners = len(trajectories_df[trajectories_df["trajectory"] != "FreeKick_Box"])

    # Legenda cromatica chiara
    custom_legend = [
        patches.Patch(facecolor="#ff2a2a", edgecolor="none", label="Punizione Indiretta in Area"),
        patches.Patch(facecolor="#00ff85", edgecolor="none", label="Corner Rientrante (Inswinging)"),
        patches.Patch(facecolor="#00b0ff", edgecolor="none", label="Corner Uscente (Outswinging)"),
        patches.Patch(facecolor="#ff9100", edgecolor="none", label="Corner Teso / Diretto (Straight)")
    ]
    ax.legend(
        handles=custom_legend,
        facecolor=OPTA_CARD,
        edgecolor=OPTA_BORDER,
        labelcolor=OPTA_TEXT,
        fontsize=7.8,
        loc="lower left"
    )

    ax.set_title(
        f"SET PIECES IN AREA: {team_name.upper()}\n"
        f"Corner: {n_corners} | Punizioni in Area (Rosso): {n_fk_box} | Conclusioni generate: {int(trajectories_df['shot_assist'].sum())}",
        color=OPTA_TEXT,
        fontsize=10.5,
        pad=10,
        fontweight='bold'
    )
    return fig

# =====================================================================
# 3. ROSTER ANALYTICS & POSITIONAL PROFILE[cite: 8]
# =====================================================================

def get_squad_tactical_roster(events_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    from services.tactical_simulator import POSITION_ACRONYMS, CONV_X

    if events_df is None or events_df.empty:
        return pd.DataFrame()

    target_clean = str(team_name).strip().lower()
    team_col = "team" if "team" in events_df.columns else ("team_name" if "team_name" in events_df.columns else None)
    if not team_col:
        return pd.DataFrame()

    t_ev = events_df[
        events_df[team_col].astype(str).str.strip().str.lower() == target_clean
    ].copy()

    if t_ev.empty:
        t_ev = events_df[
            events_df[team_col].astype(str).str.strip().str.lower().str.contains(target_clean.split()[0], na=False)
        ].copy()

    if t_ev.empty:
        return pd.DataFrame()

    if "x" not in t_ev.columns:
        if "location_x" in t_ev.columns:
            t_ev["x"] = t_ev["location_x"]
        elif "location" in t_ev.columns:
            t_ev["x"] = t_ev["location"].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else np.nan)
        else:
            t_ev["x"] = 60.0

    t_ev["x_m"] = pd.to_numeric(t_ev["x"], errors="coerce") * CONV_X

    if "player" not in t_ev.columns and "player_name" in t_ev.columns:
        t_ev["player"] = t_ev["player_name"]
    elif "player" in t_ev.columns:
        t_ev["player"] = t_ev["player"].apply(
            lambda p: p.get("name") if isinstance(p, dict) else (str(p) if pd.notna(p) and str(p) != "nan" else np.nan)
        )

    t_ev = t_ev[t_ev["player"].notna()].copy()
    if t_ev.empty:
        return pd.DataFrame()

    player_roles = []

    if "tactics" in t_ev.columns:
        for tactics_dict in t_ev["tactics"].dropna():
            if isinstance(tactics_dict, dict) and "lineup" in tactics_dict:
                for item in tactics_dict["lineup"]:
                    p_name = item.get("player", {}).get("name") if isinstance(item.get("player"), dict) else item.get("player")
                    pos_val = item.get("position", {}).get("name") if isinstance(item.get("position"), dict) else item.get("position")
                    if p_name and pos_val:
                        sigla = POSITION_ACRONYMS.get(str(pos_val).strip(), str(pos_val).strip())
                        player_roles.append({"player": p_name, "sigla": sigla, "weight": 50})

    if "position" in t_ev.columns:
        for idx, row in t_ev[t_ev["position"].notna()].iterrows():
            pos_val = row["position"]
            p_name = row["player"]
            if isinstance(pos_val, dict):
                pos_str = pos_val.get("name", "")
            else:
                pos_str = str(pos_val)

            if pos_str and pos_str not in ["nan", "None", "Substitute"]:
                sigla = POSITION_ACRONYMS.get(pos_str.strip(), pos_str.strip())
                player_roles.append({"player": p_name, "sigla": sigla, "weight": 1})

    df_roles = pd.DataFrame(player_roles)

    reparto_order = {
        "GK": 1,
        "CB": 2, "RCB": 2, "LCB": 2, "RB": 2, "LB": 2, "RWB": 2, "LWB": 2,
        "DM": 3, "RDM": 3, "LDM": 3, "CM": 3, "RCM": 3, "LCM": 3, "RM": 3, "LM": 3, "AM": 3, "RAM": 3, "LAM": 3,
        "RW": 4, "LW": 4, "SS": 4, "CF": 4, "ST": 4, "RST": 4, "LST": 4
    }

    roster_rows = []
    player_stats = t_ev.groupby("player").agg(
        volume=("x_m", "count"),
        med_x=("x_m", "median")
    ).reset_index()

    for _, p_row in player_stats.iterrows():
        p_name = p_row["player"]
        tot_vol = int(p_row["volume"])
        avg_x = round(float(p_row["med_x"]), 1) if pd.notna(p_row["med_x"]) else 50.0

        if not df_roles.empty and p_name in df_roles["player"].values:
            p_df_roles = df_roles[df_roles["player"] == p_name]
            role_weights = p_df_roles.groupby("sigla")["weight"].sum().sort_values(ascending=False)
            primary_role = role_weights.index[0]

            sec_roles_list = []
            if len(role_weights) > 1:
                for r_sig, w in role_weights.iloc[1:].items():
                    sec_roles_list.append(f"{r_sig}")
            sec_roles = ", ".join(sec_roles_list) if sec_roles_list else "—"
        else:
            if avg_x < 15.0:
                primary_role = "GK"
            elif avg_x < 40.0:
                primary_role = "CB"
            elif avg_x < 65.0:
                primary_role = "CM"
            else:
                primary_role = "ST"
            sec_roles = "—"

        roster_rows.append({
            "player": p_name,
            "primary_role": primary_role,
            "rep_rank": reparto_order.get(primary_role, 5),
            "secondary_roles": sec_roles,
            "volume": tot_vol,
            "height_x": avg_x
        })

    df_roster = pd.DataFrame(roster_rows)
    if not df_roster.empty:
        df_roster = df_roster.sort_values(["rep_rank", "volume"], ascending=[True, False]).reset_index(drop=True)

    return df_roster

# =====================================================================
# 4. VIEW PRINCIPALE STREAMLIT DEL MODULO 3
# =====================================================================

def render_opponent_analysis_view(events_df: pd.DataFrame, our_team: str, opp_team: str):
    p_our = extract_team_profile(events_df, our_team, sample_size=5)
    p_opp = extract_team_profile(events_df, opp_team, sample_size=5)

    header_html = "".join([
        f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-left: 4px solid {OPTA_ORANGE}; border-radius: 4px; padding: 18px 22px; margin-bottom: 20px;">',
        '<div style="display: flex; justify-content: space-between; align-items: center;">',
        '<div>',
        f'<div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">OPTA COMPETITIVE ANALYSIS & TACTICAL SIMULATOR</div>',
        f'<div style="font-size: 1.35rem; font-weight: 800; color: {OPTA_TEXT}; margin: 4px 0;">',
        f'<span style="color: {OPTA_GREEN};">{our_team}</span> ',
        f'<span style="color: {OPTA_MUTED}; font-size: 1rem; margin: 0 6px;">VS</span> ',
        f'<span style="color: {OPTA_ORANGE};">{opp_team}</span>',
        '</div>',
        f'<div style="font-size: 0.78rem; color: {OPTA_MUTED};">Campione simmetrico sulle ultime 5 gare ufficiali. Dimensioni campo FIFA: 105m &times; 68m.</div>',
        '</div>',
        '<div style="text-align: right;">',
        f'<div style="display: inline-block; background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 6px 14px;">',
        f'<span style="font-size: 0.70rem; font-weight: 700; color: {OPTA_GREEN}; text-transform: uppercase;">MODULO 3 ATTIVO</span>',
        '</div>',
        '</div>',
        '</div>',
        '</div>'
    ])
    st.markdown(header_html, unsafe_allow_html=True)

    tab_dossier, tab_simulator = st.tabs([
        "📊 MATCH ANALYSIS & OPPONENT DOSSIER",
        "⚡ WHAT-IF TACTICAL SIMULATOR"
    ])

    # =====================================================================
    # TAB 1: MATCH ANALYSIS & DOSSIER PRE-GARA[cite: 8]
    # =====================================================================
    with tab_dossier:
        c_due1, c_due2, c_due3 = st.columns(3)
        with c_due1:
            st.markdown("".join([
                f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 16px; height: 100%;">',
                f'<div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_GREEN}; text-transform: uppercase; margin-bottom: 8px;">1. NOSTRA USCITA BASSA VS LORO PRESSING</div>',
                f'<div style="font-size: 0.85rem; color: {OPTA_TEXT}; margin-bottom: 4px;">Nostra Pulizia Bassa: <b>{p_our["buildup_accuracy"]}%</b> ({p_our["buildup_score"]}/100)</div>',
                f'<div style="font-size: 0.85rem; color: {OPTA_TEXT}; margin-bottom: 12px;">Loro Indice PPDA: <b>{p_opp["ppda"]}</b> (Baricentro: {p_opp["defensive_height_m"]}m)</div>',
                f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 10px; font-size: 0.75rem; color: {OPTA_MUTED}; line-height: 1.4;">',
                f'{"⚠️ <b>Allerta:</b> Pressing avversario molto aggressivo. Evitare passaggi orizzontali lenti; cercare subito il terzo uomo sul lato debole." if p_opp["ppda"] < 10.5 else "✅ <b>Opportunità:</b> Loro blocco medio/basso. Spazio per consolidare la prima impostazione e far salire i terzini."}',
                '</div>',
                '</div>'
            ]), unsafe_allow_html=True)

        with c_due2:
            pref_ch = max(p_our["channels"], key=p_our["channels"].get)
            st.markdown("".join([
                f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 16px; height: 100%;">',
                f'<div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_GREEN}; text-transform: uppercase; margin-bottom: 8px;">2. NOSTRA RIFINITURA VS STRUTTURA LORO</div>',
                f'<div style="font-size: 0.85rem; color: {OPTA_TEXT}; margin-bottom: 4px;">Nostro Canale Preferenziale: <b>{pref_ch} ({p_our["channels"][pref_ch]}%)</b></div>',
                f'<div style="font-size: 0.85rem; color: {OPTA_TEXT}; margin-bottom: 12px;">Propensione Cross: <b>{p_our["cross_ratio"]}%</b> (xG/tiro: {p_our["xg_per_shot"]})</div>',
                f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 10px; font-size: 0.75rem; color: {OPTA_MUTED}; line-height: 1.4;">',
                f'Sovraccaricare la corsia di <b>{pref_ch.lower()}</b> per attirare i loro mediani, favorendo l\'imbucata centrale e il rimorchio.',
                '</div>',
                '</div>'
            ]), unsafe_allow_html=True)

        with c_due3:
            st.markdown("".join([
                f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 16px; height: 100%;">',
                f'<div style="font-size: 0.70rem; font-weight: 700; color: {OPTA_ORANGE}; text-transform: uppercase; margin-bottom: 8px;">3. GABBIA SUL REGISTA AVVERSARIO</div>',
                f'<div style="font-size: 0.85rem; color: {OPTA_TEXT}; margin-bottom: 4px;">Metronomo: <b>{p_opp["top_passer"]}</b></div>',
                f'<div style="font-size: 0.85rem; color: {OPTA_TEXT}; margin-bottom: 12px;">Verticalizzatore: <b>{p_opp["top_progressor"]}</b></div>',
                f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 10px; font-size: 0.75rem; color: {OPTA_MUTED}; line-height: 1.4;">',
                f'<b>Direttiva:</b> Schermare la ricezione verso <b>{p_opp["top_passer"]}</b> ed aggredire il primo controllo di <b>{p_opp["top_progressor"]}</b>.',
                '</div>',
                '</div>'
            ]), unsafe_allow_html=True)

        st.write("")

        col_rad, col_ch = st.columns([1.2, 1])
        with col_rad:
            st.markdown("".join([
                f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">',
                f'<div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">RADAR TATTICO SOVRAPPOSTO (7 PARAMETRI)</div>',
                f'<div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 10px;">Verde: {our_team} | Arancione: {opp_team}</div>'
            ]), unsafe_allow_html=True)
            fig_radar = plot_head_to_head_radar(p_our, p_opp)
            st.plotly_chart(fig_radar, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ch:
            st.markdown("".join([
                f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">',
                f'<div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">CONFRONTO CANALI D\'INVASIONE TREQUARTI</div>',
                f'<div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 10px;">Distribuzione percentuale degli ingressi negli ultimi 35 metri</div>'
            ]), unsafe_allow_html=True)
            fig_ch = plot_channel_comparison(p_our["channels"], p_opp["channels"], our_team, opp_team)
            st.plotly_chart(fig_ch, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        st.markdown("".join([
            f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; margin-bottom: 20px;">',
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">',
            f'<div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">DOVE COLPIRLI: VULNERABILITÀ DIFENSIVE DELL\'AVVERSARIO (MPLSOCCER)</div>',
            f'<div style="font-size: 0.72rem; color: {OPTA_ORANGE}; font-weight: 700;">CAMPIONE GARE</div>',
            '</div>',
            f'<div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 14px;">A sinistra: zone in cui l\'avversario perde palla in fase di prima costruzione. A destra: mappatura dei tiri e gol subiti.</div>'
        ]), unsafe_allow_html=True)

        col_pl, col_pr = st.columns(2)
        with col_pl:
            fig_to = plot_mplsoccer_opponent_turnovers(p_opp["turnovers_own_half"], opp_team)
            st.pyplot(fig_to, width='stretch')
            plt.close(fig_to)
        with col_pr:
            fig_sc = plot_mplsoccer_opponent_shots_conceded(p_opp["shots_conceded"], opp_team)
            st.pyplot(fig_sc, width='stretch')
            plt.close(fig_sc)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        # 4. SET PIECES INTELLIGENCE (CLUSTERING TRAIETTORIE CORNER & PUNIZIONI)
        st.markdown("".join([
            f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; margin-bottom: 20px;">',
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">',
            f'<div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">SET PIECE INTELLIGENCE: CLUSTERING TRAIETTORIE & ZONE BATTUTA</div>',
            f'<div style="font-size: 0.72rem; color: {OPTA_GREEN}; font-weight: 700;">CORNER & PUNIZIONI</div>',
            '</div>',
            f'<div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 16px;">Analisi balistica delle traiettorie di battuta (rientranti, uscenti, tese) e clustering posizionale delle prime ricezioni in area avversaria.</div>',
            '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 16px;">',
            f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 12px; text-align: center;"><div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">Nostri Corner / 90m</div><div style="font-size: 1.5rem; font-weight: 800; color: {OPTA_GREEN};">{p_our["corners_for_p90"]}</div><div style="font-size: 0.72rem; color: {OPTA_TEXT};">xG gen: <b>{p_our["sp_xg_for_p90"]}</b></div></div>',
            f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 12px; text-align: center;"><div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">Loro Corner / 90m</div><div style="font-size: 1.5rem; font-weight: 800; color: {OPTA_ORANGE};">{p_opp["corners_for_p90"]}</div><div style="font-size: 0.72rem; color: {OPTA_TEXT};">xG gen: <b>{p_opp["sp_xg_for_p90"]}</b></div></div>',
            f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 12px; text-align: center;"><div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">Nostri Corner Concessi</div><div style="font-size: 1.5rem; font-weight: 800; color: {OPTA_GREEN};">{p_our["corners_against_p90"]}</div><div style="font-size: 0.72rem; color: {OPTA_TEXT};">xG sub: <b>{p_our["sp_xg_against_p90"]}</b></div></div>',
            f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 12px; text-align: center;"><div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">Loro Corner Concessi</div><div style="font-size: 1.5rem; font-weight: 800; color: {OPTA_ORANGE};">{p_opp["corners_against_p90"]}</div><div style="font-size: 0.72rem; color: {OPTA_TEXT};">xG sub: <b>{p_opp["sp_xg_against_p90"]}</b></div></div>',
            '</div>'
        ]), unsafe_allow_html=True)

        sp_our_df = extract_set_pieces_trajectories(events_df, our_team)
        sp_opp_df = extract_set_pieces_trajectories(events_df, opp_team)

        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            fig_sp_our = plot_mplsoccer_set_piece_clusters(sp_our_df, our_team)
            st.pyplot(fig_sp_our, width='stretch')
            plt.close(fig_sp_our)
        with col_sp2:
            fig_sp_opp = plot_mplsoccer_set_piece_clusters(sp_opp_df, opp_team)
            st.pyplot(fig_sp_opp, width='stretch')
            plt.close(fig_sp_opp)

        if not sp_opp_df.empty:
            cluster_recap = sp_opp_df.groupby("cluster").agg(
                Battuti=("distance_m", "count"),
                Gittata_Media=("distance_m", "mean"),
                Assist_Tiro=("shot_assist", "sum")
            ).reset_index()
            cluster_recap["Frequenza"] = (cluster_recap["Battuti"] / len(sp_opp_df) * 100).round(1).astype(str) + "%"
            cluster_recap["Gittata_Media"] = cluster_recap["Gittata_Media"].round(1).astype(str) + " m"

            st.markdown(
                f'<div style="font-size: 0.75rem; font-weight: 700; color: {OPTA_ORANGE}; text-transform: uppercase; margin-top: 10px; margin-bottom: 6px;">'
                f'Distribuzione Obiettivi in Area Avversaria ({opp_team})'
                f'</div>',
                unsafe_allow_html=True
            )
            st.dataframe(cluster_recap, width='stretch', hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")

        # 5. ROSTER ANALYTICS & POSITIONAL PROFILE[cite: 8]
        st.markdown(
            f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-left: 4px solid {OPTA_ORANGE}; border-radius: 4px; padding: 14px 18px; margin-bottom: 14px;">'
            f'<div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_ORANGE}; text-transform: uppercase; letter-spacing: 0.08em;">ROSTER ANALYTICS & POSITIONAL PROFILE</div>'
            f'<div style="font-size: 1.05rem; font-weight: 800; color: {OPTA_TEXT}; margin-top: 2px;">Organico e Polivalenza Tattica: {opp_team.upper()}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        roster_df = get_squad_tactical_roster(events_df, opp_team)

        if roster_df is not None and not roster_df.empty:
            def get_role_badge_style(role: str):
                role_str = str(role).upper()
                if role_str == "GK":
                    return "background: #263238; color: #90a4ae; border: 1px solid #37474f;"
                elif any(k in role_str for k in ["B", "CB", "WB"]):
                    return "background: rgba(33, 150, 243, 0.15); color: #29b6f6; border: 1px solid #0288d1;"
                elif any(k in role_str for k in ["M", "DM", "CM", "AM"]):
                    return "background: rgba(0, 255, 133, 0.15); color: #00ff85; border: 1px solid #00c853;"
                else:
                    return "background: rgba(255, 87, 34, 0.15); color: #ff5722; border: 1px solid #e64a19;"

            table_rows = []
            for _, row in roster_df.iterrows():
                badge_style = get_role_badge_style(row["primary_role"])
                p_name = str(row["player"])
                p_role = str(row["primary_role"])
                sec_roles = str(row["secondary_roles"])
                vol = str(row["volume"])
                hx = f"{row['height_x']} m"

                row_html = (
                    f'<tr style="border-bottom: 1px solid #1e2630;">'
                    f'<td style="padding: 10px 14px; font-weight: 700; color: {OPTA_TEXT};">{p_name}</td>'
                    f'<td style="padding: 10px 14px; text-align: center;">'
                    f'<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 800; {badge_style}">{p_role}</span>'
                    f'</td>'
                    f'<td style="padding: 10px 14px; color: #8b949e; font-size: 0.80rem;">{sec_roles}</td>'
                    f'<td style="padding: 10px 14px; text-align: center; color: {OPTA_TEXT}; font-weight: 600;">{vol}</td>'
                    f'<td style="padding: 10px 14px; text-align: center; color: {OPTA_GREEN}; font-weight: 700;">{hx}</td>'
                    f'</tr>'
                )
                table_rows.append(row_html)

            tbody_content = "".join(table_rows)

            html_table = (
                f'<div style="background: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 6px; overflow-x: auto; margin-bottom: 20px;">'
                f'<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.86rem; margin: 0; padding: 0;">'
                f'<thead>'
                f'<tr style="background: #18222c; border-bottom: 1.5px solid {OPTA_BORDER}; font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;">'
                f'<th style="padding: 10px 14px;">Calciatore</th>'
                f'<th style="padding: 10px 14px; text-align: center;">Ruolo Principale</th>'
                f'<th style="padding: 10px 14px;">Ruoli Secondari (Eventi)</th>'
                f'<th style="padding: 10px 14px; text-align: center;">Coinvolgimento (Eventi)</th>'
                f'<th style="padding: 10px 14px; text-align: center;">Baricentro Medio (X)</th>'
                f'</tr>'
                f'</thead>'
                f'<tbody>'
                f'{tbody_content}'
                f'</tbody>'
                f'</table>'
                f'</div>'
            )

            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.info(f"Nessun dato di ruolo dettagliato disponibile per {opp_team}.")

        st.write("")

        # 6. AI TACTICAL COPILOT[cite: 8]
        ai_advice = generate_tactical_training_guidelines(p_our, p_opp, {})

        st.markdown("".join([
            f'<div style="background-color: {OPTA_CARD}; border: 1.5px solid {OPTA_BLUE}; border-left: 5px solid {OPTA_BLUE}; border-radius: 4px; padding: 14px 18px; margin-bottom: 14px;">',
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">',
            '<div style="display: align-items: center; gap: 8px;">',
            '<span style="font-size: 1.15rem;">🤖</span>',
            '<div>',
            f'<div style="font-size: 0.70rem; font-weight: 800; color: {OPTA_BLUE}; text-transform: uppercase; letter-spacing: 0.08em;">AI TACTICAL COPILOT | DIRETTIVE METODOLOGICHE</div>',
            f'<div style="font-size: 1.0rem; font-weight: 800; color: {OPTA_TEXT};">Indirizzo per la Programmazione del Microciclo</div>',
            '</div>',
            '</div>',
            f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 4px 10px; font-size: 0.68rem; font-weight: 800; color: {OPTA_BLUE};">TACTICAL ADVISORY</div>',
            '</div>',
            f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 10px 14px;">',
            f'<div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase; font-weight: 800;">Scenario di Gara Diagnosticato</div>',
            f'<div style="font-size: 0.85rem; font-weight: 800; color: {OPTA_TEXT}; margin: 2px 0;">{ai_advice["scenario"]}</div>',
            f'<div style="font-size: 0.76rem; color: {OPTA_GREEN}; margin-top: 3px;"><span style="font-weight: 800;">DIRETTRICE OPERATIVA:</span> {ai_advice["core_theme"]}</div>',
            '</div>',
            '</div>'
        ]), unsafe_allow_html=True)

        c_g1, c_g2 = st.columns(2)
        for idx, g in enumerate(ai_advice["guidelines"]):
            col_target = c_g1 if idx % 2 == 0 else c_g2
            with col_target:
                st.markdown("".join([
                    f'<div style="background: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px 16px; margin-bottom: 12px; min-height: 250px;">',
                    '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">',
                    f'<span style="font-size: 0.72rem; font-weight: 800; color: {OPTA_BLUE}; text-transform: uppercase; letter-spacing: 0.05em;">{g["giorno"]}</span>',
                    f'<span style="font-size: 0.68rem; color: {OPTA_MUTED}; font-weight: 700;">{g["fase"]}</span>',
                    '</div>',
                    f'<div style="font-size: 0.82rem; color: {OPTA_TEXT}; line-height: 1.45; margin-bottom: 10px;"><span style="font-weight: 800; color: {OPTA_GREEN};">PRINCIPIO:</span> {g["principio_guida"]}</div>',
                    f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 10px; display: flex; flex-direction: column; gap: 6px;">',
                    f'<div style="font-size: 0.72rem; color: {OPTA_TEXT}; line-height: 1.3;"><span style="font-weight: 800; color: {OPTA_MUTED};">Spazi & Struttura:</span> {g["sviluppo_spaziale"]}</div>',
                    f'<div style="font-size: 0.72rem; color: {OPTA_TEXT}; line-height: 1.3;"><span style="font-weight: 800; color: {OPTA_ORANGE};">Innesco Situazionale:</span> {g["innesco_situazionale"]}</div>',
                    f'<div style="font-size: 0.72rem; color: {OPTA_TEXT}; line-height: 1.3;"><span style="font-weight: 800; color: {OPTA_BLUE};">Compiti Reparto:</span> {g["obiettivo_reparto"]}</div>',
                    '</div>',
                    '</div>'
                ]), unsafe_allow_html=True)

    # =====================================================================
    # TAB 2: WHAT-IF TACTICAL SIMULATOR (TIMELINE, PRESETS & CACHING)[cite: 8]
    # =====================================================================
    with tab_simulator:
        from collections import Counter

        st.markdown("".join([
            f'<div style="background-color: {OPTA_CARD}; border: 1.5px solid {OPTA_GREEN}; border-radius: 4px; padding: 16px 20px; margin-bottom: 16px;">',
            '<div style="display: flex; justify-content: space-between; align-items: center;">',
            '<div>',
            f'<div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_GREEN}; text-transform: uppercase; letter-spacing: 0.08em;">WHAT-IF PREDICTIVE ENGINE (TACTICAL BOARD)</div>',
            f'<div style="font-size: 1.15rem; font-weight: 800; color: {OPTA_TEXT}; margin-top: 2px;">Simulatore Dinamico: In-Game Timeline, Game State & Preset Scenari</div>',
            '</div>',
            f'<div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 4px 10px; font-size: 0.70rem; color: {OPTA_MUTED};">DETERMINISTIC CACHE ENGINE</div>',
            '</div>',
            '</div>'
        ]), unsafe_allow_html=True)

        # 1. Selettori del Sistema di Gioco[cite: 8]
        real_forms_our = get_available_team_formations(events_df, our_team)
        real_forms_opp = get_available_team_formations(events_df, opp_team)

        c_mod1, c_mod2 = st.columns(2)
        with c_mod1:
            sel_our_form = st.selectbox(
                f"SISTEMA DI GIOCO: {our_team.upper()}",
                real_forms_our,
                key=f"tboard_form_our_{our_team}"
            )
        with c_mod2:
            sel_opp_form = st.selectbox(
                f"SISTEMA DI GIOCO: {opp_team.upper()}",
                real_forms_opp,
                key=f"tboard_form_opp_{opp_team}"
            )

        # 2. BARRA CONTESTO DI GARA: TIMELINE & GAME STATE[cite: 8]
        c_time, c_state = st.columns(2)
        with c_time:
            st.markdown("<div style='font-size: 0.72rem; font-weight: 800; color: #8e8e93; text-transform: uppercase; margin-bottom: 4px;'>1. Finestra Temporale / Impatto Sostituzioni</div>", unsafe_allow_html=True)
            sel_time_phase = st.segmented_control(
                "Fase Temporale",
                options=["Intero Match (0'-90')", "Primo Tempo (0'-45')", "Fase Centrale (0'-60')", "Fase Finale & Cambi (60'-90+')"],
                default="Intero Match (0'-90')",
                label_visibility="collapsed",
                key=f"sim_time_phase_{our_team}"
            )

        with c_state:
            st.markdown("<div style='font-size: 0.72rem; font-weight: 800; color: #8e8e93; text-transform: uppercase; margin-bottom: 4px;'>2. Inerzia di Gara (Game State)</div>", unsafe_allow_html=True)
            sel_game_state = st.segmented_control(
                "Stato Punteggio",
                options=["Tutti gli Stati", "In Parità (0-0)", "In Vantaggio (+1)", "In Svantaggio (-1)"],
                default="Tutti gli Stati",
                label_visibility="collapsed",
                key=f"sim_game_state_{our_team}"
            )

        time_map = {
            "Intero Match (0'-90')": (0, 100),
            "Primo Tempo (0'-45')": (0, 45),
            "Fase Centrale (0'-60')": (0, 60),
            "Fase Finale & Cambi (60'-90+')": (60, 100)
        }
        state_map = {
            "Tutti gli Stati": "ALL",
            "In Parità (0-0)": "TIED",
            "In Vantaggio (+1)": "LEADING",
            "In Svantaggio (-1)": "TRAILING"
        }
        active_minute_range = time_map.get(sel_time_phase, (0, 100))
        active_game_state = state_map.get(sel_game_state, "ALL")

        key_commit_our = f"committed_lineup_our_{our_team}_{sel_our_form}"
        key_commit_opp = f"committed_lineup_opp_{opp_team}_{sel_opp_form}"

        if key_commit_our not in st.session_state:
            st.session_state[key_commit_our] = {}
        if key_commit_opp not in st.session_state:
            st.session_state[key_commit_opp] = {}
        if "tactical_presets" not in st.session_state:
            st.session_state["tactical_presets"] = {}

        # 3. GESTIONE PRESET TATTICI DELLO STAFF[cite: 8]
        with st.expander("💾 Gestione Scenari Tattici Salvati (Preset Staff)", expanded=False):
            c_p_left, c_p_right = st.columns([1.2, 1])
            with c_p_left:
                st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#8e8e93; text-transform:uppercase; margin-bottom:4px;'>Salva Configurazione Corrente</div>", unsafe_allow_html=True)
                new_preset_name = st.text_input("Nome Scenario", placeholder="es. Assalto Finale 4-2-4", key=f"txt_preset_{our_team}_{sel_our_form}")
                if st.button("💾 SALVA SCENARIO", key=f"btn_save_preset_{our_team}", width='stretch'):
                    if new_preset_name.strip():
                        st.session_state["tactical_presets"][new_preset_name.strip()] = {
                            "our_formation": sel_our_form,
                            "opp_formation": sel_opp_form,
                            "our_lineup": dict(st.session_state[key_commit_our]),
                            "opp_lineup": dict(st.session_state[key_commit_opp]),
                            "time_phase": sel_time_phase,
                            "game_state": sel_game_state
                        }
                        st.toast(f"Scenario '{new_preset_name.strip()}' memorizzato!", icon="✅")
                    else:
                        st.toast("Inserisci un nome valido prima di salvare.", icon="⚠️")

            with c_p_right:
                st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#8e8e93; text-transform:uppercase; margin-bottom:4px;'>Richiama Scenario Salvato</div>", unsafe_allow_html=True)
                saved_keys = list(st.session_state["tactical_presets"].keys())
                if saved_keys:
                    chosen_preset = st.selectbox("Seleziona Scenario", saved_keys, key=f"sel_saved_preset_{our_team}")
                    if st.button("⚡ CARICA SCENARIO", key=f"btn_load_preset_{our_team}", width='stretch'):
                        cfg = st.session_state["tactical_presets"][chosen_preset]
                        st.session_state[key_commit_our] = cfg.get("our_lineup", {})
                        st.session_state[key_commit_opp] = cfg.get("opp_lineup", {})
                        st.toast(f"Scenario '{chosen_preset}' caricato con successo!", icon="🚀")
                        st.rerun()
                else:
                    st.caption("Nessuno scenario personalizzato salvato nella sessione.")

        # 4. TACTICAL BOARD (CHECKBOX PER REPARTO & BATCH COMMIT)[cite: 8]
        with st.expander("📋 Tactical Board: Personalizza Titolari per Reparto", expanded=False):
            st.markdown(
                f"<div style='font-size:0.80rem; color:{OPTA_MUTED}; margin-bottom:12px;'>"
                "Spunta il calciatore desiderato per ciascun ruolo. Se non spunti nessun nome per un ruolo, "
                "il sistema selezionerà automaticamente il titolare in base ai dati storici. "
                "<b>È vietato selezionare lo stesso calciatore in più ruoli contemporaneamente.</b>"
                "</div>",
                unsafe_allow_html=True
            )

            with st.form(key=f"form_lineup_batch_{sel_our_form}_{sel_opp_form}"):
                tab_team_our, tab_team_opp = st.tabs([f"🟢 {our_team.upper()}", f"🟠 {opp_team.upper()}"])

                draft_our = {}
                draft_opp = {}

                # --- NOSTRA SQUADRA ---[cite: 8]
                with tab_team_our:
                    groups_our = get_formation_line_groups(sel_our_form)
                    for rep_name, roles_list in groups_our.items():
                        st.markdown(
                            f"<div style='font-size:0.82rem; font-weight:800; color:{OPTA_GREEN}; text-transform:uppercase; margin-top:8px; margin-bottom:6px;'>{rep_name}</div>", 
                            unsafe_allow_html=True
                        )
                        cols = st.columns(len(roles_list))
                        for idx_r, (r_code, r_label) in enumerate(roles_list):
                            with cols[idx_r]:
                                candidates = get_role_candidates(events_df, our_team, r_code)
                                current_val = st.session_state[key_commit_our].get(r_code, None)

                                st.markdown(
                                    f"<div style='background:{OPTA_BG}; border:1px solid {OPTA_BORDER}; border-radius:3px; padding:4px 6px; font-size:0.75rem; font-weight:700; color:{OPTA_TEXT}; text-align:center; margin-bottom:6px;'>{r_code}</div>", 
                                    unsafe_allow_html=True
                                )

                                picked_players = []
                                for cand in candidates:
                                    is_active = (current_val == cand)
                                    cb_cand = st.checkbox(
                                        cand,
                                        value=is_active,
                                        key=f"chk_our_{sel_our_form}_{r_code}_{cand}"
                                    )
                                    if cb_cand:
                                        picked_players.append(cand)

                                draft_our[r_code] = picked_players[-1] if picked_players else None

                # --- SQUADRA AVVERSARIA ---[cite: 8]
                with tab_team_opp:
                    groups_opp = get_formation_line_groups(sel_opp_form)
                    for rep_name, roles_list in groups_opp.items():
                        st.markdown(
                            f"<div style='font-size:0.82rem; font-weight:800; color:{OPTA_ORANGE}; text-transform:uppercase; margin-top:8px; margin-bottom:6px;'>{rep_name}</div>", 
                            unsafe_allow_html=True
                        )
                        cols = st.columns(len(roles_list))
                        for idx_r, (r_code, r_label) in enumerate(roles_list):
                            with cols[idx_r]:
                                candidates = get_role_candidates(events_df, opp_team, r_code)
                                current_val = st.session_state[key_commit_opp].get(r_code, None)

                                st.markdown(
                                    f"<div style='background:{OPTA_BG}; border:1px solid {OPTA_BORDER}; border-radius:3px; padding:4px 6px; font-size:0.75rem; font-weight:700; color:{OPTA_TEXT}; text-align:center; margin-bottom:6px;'>{r_code}</div>", 
                                    unsafe_allow_html=True
                                )

                                picked_players_opp = []
                                for cand in candidates:
                                    is_active = (current_val == cand)
                                    cb_cand = st.checkbox(
                                        cand,
                                        value=is_active,
                                        key=f"chk_opp_{sel_opp_form}_{r_code}_{cand}"
                                    )
                                    if cb_cand:
                                        picked_players_opp.append(cand)

                                draft_opp[r_code] = picked_players_opp[-1] if picked_players_opp else None

                st.write("")
                btn_commit = st.form_submit_button(
                    "⚡ APPLICA FORMAZIONI E RICALCOLA MATCHUP",
                    use_container_width=True
                )

                if btn_commit:
                    selected_our_names = [p for p in draft_our.values() if p is not None and str(p).strip() != ""]
                    selected_opp_names = [p for p in draft_opp.values() if p is not None and str(p).strip() != ""]

                    counts_our = Counter(selected_our_names)
                    counts_opp = Counter(selected_opp_names)

                    duplicates_our = [p for p, c in counts_our.items() if c > 1]
                    duplicates_opp = [p for p, c in counts_opp.items() if c > 1]

                    if duplicates_our or duplicates_opp:
                        if duplicates_our:
                            for dup in duplicates_our:
                                roles_inv = [r for r, p in draft_our.items() if p == dup]
                                st.toast(f"Duplicato {our_team}: {dup} inserito in {', '.join(roles_inv)}", icon="⚠️")
                        if duplicates_opp:
                            for dup in duplicates_opp:
                                roles_inv = [r for r, p in draft_opp.items() if p == dup]
                                st.toast(f"Duplicato {opp_team}: {dup} inserito in {', '.join(roles_inv)}", icon="⚠️")
                        st.stop()
                    else:
                        st.session_state[key_commit_our] = draft_our
                        st.session_state[key_commit_opp] = draft_opp
                        st.toast("Formazioni convalidate e mappa aggiornata!", icon="✅")

        # 5. ESECUZIONE DELLA SIMULAZIONE CON CACHING DETERMINISTICO[cite: 8]
        p_our_tuple = tuple((k, tuple(v.items()) if isinstance(v, dict) else v) for k, v in p_our.items() if not isinstance(v, (pd.DataFrame, list)))
        p_opp_tuple = tuple((k, tuple(v.items()) if isinstance(v, dict) else v) for k, v in p_opp.items() if not isinstance(v, (pd.DataFrame, list)))
        custom_our_tuple = tuple(sorted(st.session_state[key_commit_our].items()))
        custom_opp_tuple = tuple(sorted(st.session_state[key_commit_opp].items()))

        sim_res = simulate_tactical_clash_cached(
            events_df=events_df,
            our_team=our_team,
            opp_team=opp_team,
            our_formation=sel_our_form,
            opp_formation=sel_opp_form,
            p_our_tuple=p_our_tuple,
            p_opp_tuple=p_opp_tuple,
            custom_our_items=custom_our_tuple,
            custom_opp_items=custom_opp_tuple,
            minute_range=active_minute_range,
            game_state=active_game_state
        )

        # 6. RENDERING PITCH TATTICO ZONALE[cite: 8]
        fig_clash = plot_tactical_zones_clash(
            sim_res["zone_matrix"],
            sim_res["nodes_our"],
            sim_res["nodes_opp"],
            f"{our_team} ({sel_our_form})",
            f"{opp_team} ({sel_opp_form})"
        )
        st.pyplot(fig_clash, width='stretch')
        plt.close(fig_clash)

        st.write("")

        # 7. SCOREBOARD PREDITTIVO[cite: 8]
        c_p1, c_p2, c_p3, c_p4 = st.columns(4)
        with c_p1:
            st.markdown("".join([
                f'<div style="background: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px; text-align: center;">',
                f'<div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">PROBABILITÀ VITTORIA</div>',
                f'<div style="font-size: 1.6rem; font-weight: 800; color: {OPTA_GREEN};">{sim_res["prob_win"]}%</div>',
                f'<div style="font-size: 0.72rem; color: {OPTA_MUTED};">Pareggio: {sim_res["prob_draw"]}% | Sconfitta: {sim_res["prob_loss"]}%</div>',
                '</div>'
            ]), unsafe_allow_html=True)

        with c_p2:
            st.markdown("".join([
                f'<div style="background: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px; text-align: center;">',
                f'<div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">xG GENERATI PREVISTI</div>',
                f'<div style="font-size: 1.6rem; font-weight: 800; color: {OPTA_GREEN};">{sim_res["expected_our_xg"]}</div>',
                f'<div style="font-size: 0.72rem; color: {OPTA_MUTED};">Base: {p_our["xg_p90"]}</div>',
                '</div>'
            ]), unsafe_allow_html=True)

        with c_p3:
            st.markdown("".join([
                f'<div style="background: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px; text-align: center;">',
                f'<div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">xG CONCESSI PREVISTI</div>',
                f'<div style="font-size: 1.6rem; font-weight: 800; color: {OPTA_ORANGE};">{sim_res["expected_opp_xg"]}</div>',
                f'<div style="font-size: 0.72rem; color: {OPTA_MUTED};">Base: {p_opp["xg_p90"]}</div>',
                '</div>'
            ]), unsafe_allow_html=True)

        with c_p4:
            st.markdown("".join([
                f'<div style="background: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px; text-align: center;">',
                f'<div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase;">FIELD TILT PREVISTO</div>',
                f'<div style="font-size: 1.6rem; font-weight: 800; color: {OPTA_TEXT};">{sim_res["predicted_field_tilt"]}%</div>',
                f'<div style="font-size: 0.72rem; color: {OPTA_GREEN};">Quota territorio ({sel_time_phase})</div>',
                '</div>'
            ]), unsafe_allow_html=True)

        st.write("")

        # 8. DIAGNOSI PER CANALI TATTICI[cite: 8]
        st.markdown("".join([
            f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-left: 4px solid {OPTA_GREEN}; border-radius: 4px; padding: 16px 20px; margin-bottom: 16px;">',
            f'<div style="font-size: 0.74rem; font-weight: 800; color: {OPTA_GREEN}; text-transform: uppercase;">ANALISI SCONTRO TATTICO SUI 5 CANALI VERTICALI ({sel_our_form} VS {sel_opp_form} — {sel_game_state.upper()})</div>',
            '<div style="margin-top: 10px; font-size: 0.88rem; color: #f8f9fa; line-height: 1.6;">',
            "".join([f'<div style="margin-bottom: 6px;">• {item}</div>' for item in sim_res["mismatch_info"]["mismatches"]]),
            '</div>',
            '</div>'
        ]), unsafe_allow_html=True)

def plot_dual_team_spatial_clash(nodes_our: pd.DataFrame, nodes_opp: pd.DataFrame, our_name: str, opp_name: str):
    """
    Funzione di retrocompatibilità: mappa la vecchia firma sui 22 nodi 
    sulla nuova Matrice Tattica Zonale a 15 zone[cite: 8].
    Gestisce la presenza/assenza della colonna 'role' e l'orientamento di campo[cite: 8].
    """
    from services.tactical_simulator import calculate_tactical_zones_matrix

    df_our = nodes_our.copy() if nodes_our is not None else pd.DataFrame()
    df_opp = nodes_opp.copy() if nodes_opp is not None else pd.DataFrame()

    if "role" not in df_our.columns:
        df_our["role"] = df_our.get("position", "MID")
    if "role" not in df_opp.columns:
        df_opp["role"] = df_opp.get("position", "MID")

    if not df_opp.empty and "x_mean" in df_opp.columns:
        if df_opp["x_mean"].mean() < 52.5:
            df_opp["x_mean"] = 105.0 - df_opp["x_mean"]
            df_opp["y_mean"] = 68.0 - df_opp["y_mean"]

    zone_mat, mismatch_info = calculate_tactical_zones_matrix(df_our, df_opp)
    fig = plot_tactical_zones_clash(zone_mat, df_our, df_opp, our_name, opp_name)

    clash_kpi = {
        "zone_matrix": zone_mat,
        "mismatches": mismatch_info.get("mismatches", [])
    }
    return fig, clash_kpi