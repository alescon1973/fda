import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from shapely.geometry import Polygon as ShapelyPolygon
from scipy.spatial import Voronoi
import ast
from typing import Dict, Any
import textwrap


# Parametri metrici FIFA 105m x 68m su coordinate StatsBomb (120yd x 80yd)
CONV_X = 105.0 / 120.0
CONV_Y = 68.0 / 80.0

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

POSITION_ACRONYMS = {
    "Goalkeeper": "GK", "Right Back": "RB", "Right Center Back": "RCB",
    "Center Back": "CB", "Left Center Back": "LCB", "Left Back": "LB",
    "Right Wing Back": "RWB", "Left Wing Back": "LWB",
    "Right Defensive Midfield": "RDM", "Center Defensive Midfield": "DM", "Left Defensive Midfield": "LDM",
    "Right Midfield": "RM", "Right Center Midfield": "RCM", "Center Midfield": "CM",
    "Left Center Midfield": "LCM", "Left Midfield": "LM",
    "Right Attacking Midfield": "RAM", "Center Attacking Midfield": "AM", "Left Attacking Midfield": "LAM",
    "Right Wing": "RW", "Left Wing": "LW", "Second Striker": "SS",
    "Center Forward": "CF", "Striker": "ST", "Right Center Forward": "RST", "Left Center Forward": "LST"
}

# =====================================================================
# FORMATTAZIONE DEI BADGE PER VISUALIZZAZIO TOP XI
# =====================================================================
def get_role_badge_style(role: str) -> str:
    role_str = str(role).upper()
    if role_str == "GK":
        return "background: #263238; color: #90a4ae; border: 1px solid #37474f;"
    elif any(k in role_str for k in ["B", "CB", "WB"]):
        return "background: rgba(33, 150, 243, 0.15); color: #29b6f6; border: 1px solid #0288d1;"
    elif any(k in role_str for k in ["M", "DM", "CM", "AM"]):
        return "background: rgba(0, 255, 133, 0.15); color: #00ff85; border: 1px solid #00c853;"
    else:  # Attaccanti ed esterni offensivi
        return "background: rgba(255, 87, 34, 0.15); color: #ff5722; border: 1px solid #e64a19;"
    
# =====================================================================
# 1. IDENTIFICAZIONE MODULO PREDOMINANTE & BEST XI PER RUOLO
# =====================================================================

def get_dominant_formation_and_best_lineup(events_df: pd.DataFrame, team_name: str) -> Dict[str, Any]:
    team_ev = events_df[events_df["team"] == team_name].copy()
    tactical_events = team_ev[team_ev["type"].isin(["Starting XI", "Tactical Shift"])].sort_values(["match_id", "minute"])

    formation_minutes = {}
    player_role_minutes = {}

    for match_id, match_group in team_ev.groupby("match_id"):
        match_tactics = tactical_events[tactical_events["match_id"] == match_id].sort_values("minute")
        match_max_min = int(match_group["minute"].max())

        shifts = []
        for _, row in match_tactics.iterrows():
            tactics_val = row.get("tactics")
            if pd.isna(tactics_val):
                continue
            if isinstance(tactics_val, str):
                try:
                    tactics_dict = ast.literal_eval(tactics_val)
                except Exception:
                    continue
            elif isinstance(tactics_val, dict):
                tactics_dict = tactics_val
            else:
                continue

            raw_form = str(tactics_dict.get("formation", "N/D"))
            if len(raw_form) == 3:
                f_name = f"{raw_form[0]}-{raw_form[1]}-{raw_form[2]}"
            elif len(raw_form) == 4:
                f_name = f"{raw_form[0]}-{raw_form[1]}-{raw_form[2]}-{raw_form[3]}"
            else:
                f_name = raw_form

            shifts.append({
                "minute": int(row["minute"]),
                "formation": f_name,
                "lineup": tactics_dict.get("lineup", [])
            })

        for i in range(len(shifts)):
            curr = shifts[i]
            next_min = shifts[i + 1]["minute"] if i + 1 < len(shifts) else match_max_min
            dur = max(next_min - curr["minute"], 1)

            f_name = curr["formation"]
            formation_minutes[f_name] = formation_minutes.get(f_name, 0) + dur

            if f_name not in player_role_minutes:
                player_role_minutes[f_name] = {}

            for p_info in curr["lineup"]:
                p_name = p_info["player"]["name"]
                pos_name = p_info["position"]["name"]

                if pos_name not in player_role_minutes[f_name]:
                    player_role_minutes[f_name][pos_name] = {}

                player_role_minutes[f_name][pos_name][p_name] = (
                    player_role_minutes[f_name][pos_name].get(p_name, 0) + dur
                )

    if not formation_minutes:
        return {"dominant_formation": "N/D", "best_lineup": {}, "total_minutes": 0}

    dominant_formation = max(formation_minutes, key=formation_minutes.get)
    total_form_mins = formation_minutes[dominant_formation]

    best_lineup = {}
    assigned_players = set()
    role_dict = player_role_minutes.get(dominant_formation, {})

    for pos, players_count in role_dict.items():
        sorted_candidates = sorted(players_count.items(), key=lambda x: x[1], reverse=True)
        chosen = None
        for cand, mins in sorted_candidates:
            if cand not in assigned_players:
                chosen = (cand, mins)
                assigned_players.add(cand)
                break
        if not chosen and sorted_candidates:
            chosen = sorted_candidates[0]
        
        if chosen:
            best_lineup[pos] = {"player": chosen[0], "minutes": chosen[1]}

    return {
        "dominant_formation": dominant_formation,
        "best_lineup": best_lineup,
        "total_minutes": total_form_mins
    }


# =====================================================================
# 2. CALCOLO PASSING NETWORK & METRICHE NORMALIZE
# =====================================================================

def compute_lineup_passing_network(events_df: pd.DataFrame, team_name: str, best_lineup: dict, min_pass_threshold: int = 3) -> Dict[str, Any]:
    ideal_players = [info["player"] for info in best_lineup.values()]
    if not ideal_players:
        return {"nodes": pd.DataFrame(), "edges": pd.DataFrame()}

    team_ev = events_df[events_df["team"] == team_name].copy()

    passes = team_ev[
        (team_ev["type"] == "Pass") & 
        (team_ev["pass_outcome"].isna()) & 
        (team_ev["player"].isin(ideal_players)) & 
        (team_ev["pass_recipient"].isin(ideal_players))
    ].copy()

    if passes.empty:
        return {"nodes": pd.DataFrame(), "edges": pd.DataFrame()}

    # Conversione in metri regolamentari
    passes["x_m"] = passes["x"] * CONV_X
    passes["y_m"] = passes["y"] * CONV_Y
    passes["end_x_m"] = passes["pass_end_x"] * CONV_X
    passes["end_y_m"] = passes["pass_end_y"] * CONV_Y

    # Nodi
    pass_made = passes.groupby("player").agg(
        x_mean=("x_m", "mean"),
        y_mean=("y_m", "mean"),
        passes_sent=("id", "count")
    )
    pass_received = passes.groupby("pass_recipient").agg(
        passes_received=("id", "count")
    )

    passes["is_prog"] = (passes["end_x_m"] - passes["x_m"]) >= 9.14
    prog_passes = passes[passes["is_prog"]].groupby("player").agg(progressive_sent=("id", "count"))

    nodes = pass_made.join(pass_received, how="outer").join(prog_passes, how="outer").fillna(0)
    nodes["total_involvement"] = nodes["passes_sent"] + nodes["passes_received"]

    player_to_pos = {v["player"]: k for k, v in best_lineup.items()}
    nodes["position"] = nodes.index.map(player_to_pos).fillna("N/D")

    # Archi
    pass_pairs = passes.groupby(["player", "pass_recipient"]).size().reset_index(name="pass_count")
    pass_pairs = pass_pairs[pass_pairs["pass_count"] >= min_pass_threshold]

    pass_pairs = pass_pairs.merge(nodes[["x_mean", "y_mean"]], left_on="player", right_index=True)
    pass_pairs = pass_pairs.merge(nodes[["x_mean", "y_mean"]], left_on="pass_recipient", right_index=True, suffixes=('_start', '_end'))

    return {
        "nodes": nodes.reset_index(),
        "edges": pass_pairs
    }


# =====================================================================
# 3. PASSING NETWORK CON PITCH MPLSOCCER
# =====================================================================

def plot_best_xi_network(network_data: Dict[str, Any], team_name: str, dominant_formation: str) -> plt.Figure:
    """
    Renderizza la Passing Network in stile Opta Dark:
    - Diametro dei nodi scalato proporzionalmente alla somma di passaggi effettuati e ricevuti.
    - Connessioni disegnate con frecce direzionali, spessore e trasparenza scalati sul volume di scambi.
    """
    nodes = network_data["nodes"]
    edges = network_data["edges"]

    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color=OPTA_BG,
        line_color=OPTA_LINE,
        linewidth=1.2,
        goal_type='box'
    )
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.patch.set_facecolor(OPTA_CARD)

    # 1. Tracciamento delle Connessioni (Frecce graduate con pitch.arrows)
    if not edges.empty:
        max_edge_passes = max(edges["pass_count"].max(), 1)

        for _, row in edges.iterrows():
            x_start = row["x_mean_start"] / CONV_X
            y_start = row["y_mean_start"] / CONV_Y
            x_end = row["x_mean_end"] / CONV_X
            y_end = row["y_mean_end"] / CONV_Y

            # Normalizzazione volume di scambio (0.0 a 1.0)
            norm_vol = row["pass_count"] / max_edge_passes

            # Spessore ed opacità crescenti
            line_width = 1.0 + (norm_vol * 5.0)
            alpha_val = 0.25 + (norm_vol * 0.65)

            # Freccia direzionale sottile
            pitch.arrows(
                x_start, y_start, x_end, y_end,
                ax=ax,
                color=OPTA_GREEN,
                width=line_width,
                headwidth=3.5,
                headlength=4.5,
                headaxislength=3.5,
                alpha=alpha_val,
                zorder=2
            )

    # 2. Tracciamento dei Nodi Giocatori (Scalati sul Volume Totale di Coinvolgimento)
    if not nodes.empty:
        pts_x = nodes["x_mean"] / CONV_X
        pts_y = nodes["y_mean"] / CONV_Y

        # Coinvolgimento Totale: Sent + Received
        inv_tot = nodes["total_involvement"].to_numpy()
        min_inv = inv_tot.min()
        max_inv = inv_tot.max()

        # Evitiamo divisioni per zero se il dataset è molto ridotto
        if max_inv == min_inv:
            node_sizes = np.full_like(inv_tot, 280.0)
        else:
            # Scala di grandezza del marcatore (area 's' in pt^2 da 160 a 650)
            norm_inv = (inv_tot - min_inv) / (max_inv - min_inv)
            node_sizes = 160.0 + (norm_inv * 490.0)

        # Disegno nodi Opta Dark (sfondo antracite scuro con perimetro verde fluo)
        pitch.scatter(
            pts_x,
            pts_y,
            ax=ax,
            s=node_sizes,
            color=OPTA_BG,
            edgecolors=OPTA_GREEN,
            linewidth=2.4,
            zorder=4
        )

        # Etichette con Cognome e Volume Coinvolgimento (es. "Demirović (42)")
        for i, row in nodes.iterrows():
            surname = row["player"].split()[-1]
            tot_p = int(row["total_involvement"])
            px = row["x_mean"] / CONV_X
            py = row["y_mean"] / CONV_Y

            # Cognome in bianco grassetto
            ax.text(
                px,
                py - 2.5,
                surname,
                color=OPTA_TEXT,
                fontsize=8.5,
                ha='center',
                va='top',
                fontweight='bold',
                zorder=5
            )
            # Volume complessivo tra parentesi in tonalità secondaria
            ax.text(
                px,
                py - 4.6,
                f"({tot_p})",
                color=OPTA_MUTED,
                fontsize=7.2,
                ha='center',
                va='top',
                fontweight='600',
                zorder=5
            )

    ax.set_title(
        f"PASSING NETWORK ({dominant_formation}) — {team_name.upper()}",
        color=OPTA_TEXT,
        fontsize=11,
        pad=10,
        fontweight='bold'
    )
    return fig


# =====================================================================
# 4. VORONOI DIAGRAM CON PITCH MPLSOCCER & ZONE COLORATE
# =====================================================================

def plot_team_voronoi_spacing(nodes_df: pd.DataFrame, team_name: str, dominant_formation: str):
    """
    Costruisce la mappa di Dominanza Spaziale e Pitch Control Probabilistico
    basata su tempi di intercetto differenziati per ruolo e tempo di reazione neuromuscolare.
    """
    pitch = Pitch(
        pitch_type='statsbomb',
        pitch_color=OPTA_BG,
        line_color=OPTA_LINE,
        linewidth=1.2,
        goal_type='box'
    )
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.patch.set_facecolor(OPTA_CARD)

    # 1. Coordinate reali dei calciatori in metri FIFA
    pts_x_m = nodes_df["x_mean"].to_numpy()
    pts_y_m = nodes_df["y_mean"].to_numpy()
    n_players = len(nodes_df)

    if n_players < 4:
        st.warning("Dati posizionali insufficienti per il calcolo del Pitch Control.")
        return fig, pd.DataFrame()

    # 2. Assegnazione velocità massima stimata e tempo di reazione in base al ruolo
    speeds = []
    t_reactions = []

    for _, row in nodes_df.iterrows():
        pos = str(row.get("position", "")).lower()
        if "goalkeeper" in pos or "gk" in pos:
            v_max = 5.2   # m/s
            t_react = 0.25 # Riflesso da fermo
        elif any(k in pos for k in ["wing", "back", "wb", "wide"]):
            v_max = 8.0   # Cursori di fascia ad alta velocità
            t_react = 0.45
        elif any(k in pos for k in ["midfield", "center mid", "dm", "am"]):
            v_max = 7.0   # Raccordo e mobilità a centrocampo
            t_react = 0.40
        else:
            v_max = 6.4   # Difensori centrali e prime punte
            t_react = 0.50

        speeds.append(v_max)
        t_reactions.append(t_react)

    speeds = np.array(speeds)
    t_reactions = np.array(t_reactions)

    # 3. Creazione griglia di calcolo ad alta risoluzione (105m x 68m)
    grid_x = np.linspace(0, 105, 140)
    grid_y = np.linspace(0, 68, 90)
    GX, GY = np.meshgrid(grid_x, grid_y)

    # 4. Calcolo matrice del tempo di intercetto (Time-to-Intercept: T_i = t_react + d / v)
    times = np.zeros((n_players, GY.shape[0], GY.shape[1]))
    for i in range(n_players):
        dist = np.sqrt((GX - pts_x_m[i])**2 + (GY - pts_y_m[i])**2)
        times[i] = t_reactions[i] + (dist / speeds[i])

    # 5. Modello Probabilistico Softmax per Dominanza Spaziale
    lambda_param = 1.3
    exp_neg_times = np.exp(-lambda_param * times)
    sum_exp = np.sum(exp_neg_times, axis=0) + 1e-9
    probs = exp_neg_times / sum_exp  # Probabilità continua di controllo [0, 1]

    # Identificazione del giocatore dominante per ogni cella
    dominant_player_idx = np.argmax(probs, axis=0)
    max_prob_map = np.max(probs, axis=0)

    # 6. Mappatura Cromatica Opta Semantica per Zona di Campo
    # - Verde fluo: Zona di rifinitura offensiva (X >= 70m)
    # - Ciano: Zona di sviluppo e centrocampo (35m <= X < 70m)
    # - Grigio/Slate: Prima costruzione difensiva (X < 35m)
    zone_matrix = np.zeros_like(dominant_player_idx, dtype=float)
    for i in range(n_players):
        xm = pts_x_m[i]
        mask = (dominant_player_idx == i)
        if xm >= 70.0:
            val_color = 2.0  # Offensivo (Verde)
        elif xm >= 35.0:
            val_color = 1.0  # Centrocampo (Ciano)
        else:
            val_color = 0.0  # Difesa (Grigio)
        zone_matrix[mask] = val_color

    # Conversione griglia da metri a coordinate StatsBomb (120 x 80) per il disegno
    GX_yd = GX / CONV_X
    GY_yd = GY / CONV_Y

    # Creazione Colormap personalizzata Opta Dark
    from matplotlib.colors import ListedColormap
    opta_cmap = ListedColormap(["#263238", "#0284c7", "#00ff85"])

    # Disegno della superficie probabilistica
    ax.contourf(
        GX_yd, GY_yd, zone_matrix,
        levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=opta_cmap,
        alpha=0.26,
        zorder=1
    )

    # Curve di livello e linee di confine tra i territori
    ax.contour(
        GX_yd, GY_yd, dominant_player_idx,
        levels=np.arange(n_players),
        colors="#383838",
        linewidths=0.8,
        alpha=0.6,
        zorder=2
    )

    # 7. Calcolo delle Superficie Effettiva Controllata in m²
    cell_area_sqm = (105.0 / 140.0) * (68.0 / 90.0)
    pitch_total_sqm = 105.0 * 68.0
    spatial_stats = []

    for i in range(n_players):
        player_row = nodes_df.iloc[i]
        # Ponderazione probabilistica integrale della superficie
        player_area_sqm = np.sum(probs[i]) * cell_area_sqm

        spatial_stats.append({
            "Calciatore": player_row["player"],
            "Ruolo": player_row["position"],
            "Velocità Max Stimata": f"{speeds[i]:.1f} m/s ({speeds[i]*3.6:.0f} km/h)",
            "Baricentro (X, Y)": f"{player_row['x_mean']:.1f}m, {player_row['y_mean']:.1f}m",
            "Superficie Controllata (m²)": round(player_area_sqm, 1),
            "Quota Campo (%)": f"{(player_area_sqm / pitch_total_sqm * 100):.1f}%"
        })

    # 8. Marcatori dei Calciatori (Coordinate StatsBomb)
    pts_x_yd = pts_x_m / CONV_X
    pts_y_yd = pts_y_m / CONV_Y

    pitch.scatter(
        pts_x_yd, 
        pts_y_yd, 
        ax=ax, 
        s=190, 
        color=OPTA_BG, 
        edgecolors=OPTA_GREEN, 
        linewidth=2.2, 
        zorder=4
    )

    # Cognomi sotto il marcatore
    for i in range(n_players):
        surname = nodes_df.iloc[i]["player"].split()[-1]
        ax.text(
            pts_x_yd[i], 
            pts_y_yd[i] - 2.8, 
            surname, 
            color=OPTA_TEXT, 
            fontsize=8.5, 
            ha='center', 
            va='top', 
            fontweight='bold', 
            zorder=5
        )

    ax.set_title(
        f"PITCH CONTROL PROBABILISTICO ({dominant_formation}) — {team_name.upper()}", 
        color=OPTA_TEXT, 
        fontsize=11, 
        pad=10, 
        fontweight='bold'
    )

    df_pitch_control_summary = pd.DataFrame(spatial_stats).sort_values("Superficie Controllata (m²)", ascending=False)
    return fig, df_pitch_control_summary


# =====================================================================
# 5. MATRICE DI PROGRESSIONE & INFLUENCE RANKING (PLOTLY)
# =====================================================================

def plot_player_importance_matrix(nodes_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    avg_volume = nodes_df["passes_sent"].mean()
    avg_prog = nodes_df["progressive_sent"].mean()

    hover_texts = []
    for _, r in nodes_df.iterrows():
        hover_texts.append(
            f"<b>{r['player']}</b> ({r['position']})<br>"
            f"Passaggi Completati: {int(r['passes_sent'])}<br>"
            f"Verticalizzazioni: {int(r['progressive_sent'])}<br>"
            f"Volume Totale: {int(r['total_involvement'])}"
        )

    fig.add_trace(go.Scatter(
        x=nodes_df["passes_sent"],
        y=nodes_df["progressive_sent"],
        mode="markers+text",
        text=[p.split()[-1] for p in nodes_df["player"]],
        textposition="top right",
        textfont=dict(color=OPTA_MUTED, size=10),
        marker=dict(
            size=nodes_df["total_involvement"] / nodes_df["total_involvement"].max() * 26 + 10,
            color=nodes_df["progressive_sent"],
            colorscale="Greens",
            showscale=False,
            line=dict(width=1, color=OPTA_GREEN)
        ),
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.add_vline(x=avg_volume, line=dict(color=OPTA_LINE, dash="dash", width=1))
    fig.add_hline(y=avg_prog, line=dict(color=OPTA_LINE, dash="dash", width=1))

    max_x = nodes_df["passes_sent"].max()
    max_y = nodes_df["progressive_sent"].max()

    fig.add_annotation(x=max_x, y=max_y, text="REGIA VERTICALE", showarrow=False, font=dict(color=OPTA_GREEN, size=9))
    fig.add_annotation(x=max_x, y=0, text="CONSOLIDAMENTO", showarrow=False, font=dict(color=OPTA_MUTED, size=9))
    fig.add_annotation(x=nodes_df["passes_sent"].min(), y=max_y, text="RIFINITURA / ATTACCO SPAZI", showarrow=False, font=dict(color=OPTA_ORANGE, size=9))

    fig.update_layout(
        xaxis=dict(title=dict(text="Volume Passaggi Completati", font=dict(color=OPTA_MUTED, size=11)), tickfont=dict(color=OPTA_MUTED, size=10), gridcolor=OPTA_BORDER),
        yaxis=dict(title=dict(text="Passaggi Progressivi (> 9.14m)", font=dict(color=OPTA_MUTED, size=11)), tickfont=dict(color=OPTA_MUTED, size=10), gridcolor=OPTA_BORDER),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=OPTA_BG,
        margin=dict(l=35, r=20, t=20, b=35)
    )
    return fig


def plot_player_influence_ranking(nodes_df: pd.DataFrame) -> go.Figure:
    df = nodes_df.copy()
    score = (
        (df["passes_sent"] / df["passes_sent"].max() * 40) +
        (df["passes_received"] / df["passes_received"].max() * 25) +
        (df["progressive_sent"] / df["progressive_sent"].max() * 35)
    )
    df["influence_score"] = score.round(1)
    df = df.sort_values("influence_score", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["influence_score"],
        y=[f"{p.split()[-1]} ({r})" for p, r in zip(df["player"], df["position"])],
        orientation="h",
        marker=dict(color=OPTA_GREEN, line=dict(color="#ffffff", width=0.5)),
        hovertemplate="<b>%{y}</b><br>Score Centralità: %{x}/100<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 105], tickfont=dict(color=OPTA_MUTED, size=10), gridcolor=OPTA_BORDER),
        yaxis=dict(tickfont=dict(color=OPTA_TEXT, size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=OPTA_BG,
        margin=dict(l=20, r=20, t=10, b=20)
    )
    return fig


# Griglia di probabilità empirica di segnare in base alle coordinate X in metri [0, 105]
def _spatial_score_prob(x_m: float, y_m: float) -> float:
    """Stima empirica della probabilità di gol generata dalla posizione palla."""
    # Distanza dal centro porta avversaria (105, 34)
    dist_goal = np.sqrt((105.0 - x_m)**2 + (34.0 - y_m)**2)
    # Sigmoide invertita: molto alta dentro l'area (<16m), bassa sotto i 50m
    prob = 0.40 / (1.0 + np.exp((dist_goal - 16.0) / 7.0))
    return float(prob)

def _spatial_concede_prob(x_m: float, y_m: float) -> float:
    """Stima empirica del rischio di subire gol in caso di palla persa alle coordinate x, y."""
    # Distanza dalla propria porta (0, 34)
    dist_own_goal = np.sqrt(x_m**2 + (34.0 - y_m)**2)
    risk = 0.35 / (1.0 + np.exp((dist_own_goal - 20.0) / 8.0))
    return float(risk)

def compute_team_vaep(events_df: pd.DataFrame, team_name: str, target_players: list) -> pd.DataFrame:
    """
    Calcola il punteggio VAEP (Offensive, Defensive, Net) per ogni calciatore dell'undici titolare.
    """
    team_ev = events_df[
        (events_df["team"] == team_name) & 
        (events_df["player"].isin(target_players))
    ].copy()

    if team_ev.empty:
        return pd.DataFrame()

    team_ev["x_m"] = team_ev["x"] * CONV_X
    team_ev["y_m"] = team_ev["y"] * CONV_Y
    team_ev["end_x_m"] = team_ev["pass_end_x"].fillna(team_ev["x"]) * CONV_X
    team_ev["end_y_m"] = team_ev["pass_end_y"].fillna(team_ev["y"]) * CONV_Y

    records = []
    for _, row in team_ev.iterrows():
        action_type = row.get("type", "")
        player = row.get("player")
        is_success = pd.isna(row.get("pass_outcome")) if action_type == "Pass" else True
        
        # Gestione fallimenti duelli / dribbling
        if action_type == "Dribble" and str(row.get("dribble_outcome", "")).lower() == "incomplete":
            is_success = False

        x1, y1 = row["x_m"], row["y_m"]
        x2, y2 = row["end_x_m"], row["end_y_m"]

        p_score_start = _spatial_score_prob(x1, y1)
        p_concede_start = _spatial_concede_prob(x1, y1)

        p_score_end = _spatial_score_prob(x2, y2) if is_success else 0.0
        p_concede_end = _spatial_concede_prob(x2, y2) if is_success else _spatial_concede_prob(x1, y1) * 1.5

        off_vaep = 0.0
        def_vaep = 0.0

        # 1. Passaggi e Conduzioni
        if action_type in ["Pass", "Carry"]:
            if is_success:
                off_vaep = max(p_score_end - p_score_start, 0.0)
                # Assist / Key Pass bonus
                if row.get("pass_goal_assist") == True:
                    off_vaep += 0.85
                elif row.get("pass_shot_assist") == True:
                    off_vaep += 0.25
            else:
                # Palla persa: pesante penalità difensiva se persa nella propria metà campo
                off_vaep = -0.015
                def_vaep = -(p_concede_end - p_concede_start)

        # 2. Tiri
        elif action_type == "Shot":
            xg = float(row.get("shot_statsbomb_xg", 0.1))
            is_goal = str(row.get("shot_outcome", "")).lower() == "goal"
            off_vaep = (1.0 - xg) if is_goal else (xg - p_score_start)

        # 3. Dribbling
        elif action_type == "Dribble":
            if is_success:
                off_vaep = 0.04
            else:
                off_vaep = -0.03
                def_vaep = -0.02

        # 4. Azioni Difensive (Tackle, Intercettazione, Recupero, Blocco, Spazzata)
        elif action_type in ["Tackle", "Interception", "Block", "Clearance"]:
            # Riduce la probabilità di subire gol (def_vaep positivo)
            def_vaep = p_concede_start * 0.85
        elif action_type == "Ball Recovery":
            def_vaep = p_concede_start * 0.50
            off_vaep = p_score_start * 0.30
        elif action_type == "Foul Won":
            off_vaep = 0.02
        elif action_type == "Foul Committed":
            def_vaep = -0.035

        records.append({
            "player": player,
            "off_vaep": off_vaep,
            "def_vaep": def_vaep,
            "total_vaep": off_vaep + def_vaep
        })

    df_actions = pd.DataFrame(records)
    if df_actions.empty:
        return pd.DataFrame()

    vaep_summary = df_actions.groupby("player").agg(
        azioni_valutate=("total_vaep", "count"),
        vaep_offensivo=("off_vaep", "sum"),
        vaep_difensivo=("def_vaep", "sum"),
        vaep_totale=("total_vaep", "sum")
    ).reset_index()

    vaep_summary["vaep_per_azione"] = vaep_summary["vaep_totale"] / vaep_summary["azioni_valutate"]
    return vaep_summary.sort_values("vaep_totale", ascending=False)

# =====================================================================
# 6. VIEW PRINCIPALE STREAMLIT
# =====================================================================

def render_team_tactics_view(events_df: pd.DataFrame, team_name: str):
    formation_info = get_dominant_formation_and_best_lineup(events_df, team_name)
    dominant_form = formation_info["dominant_formation"]
    best_lineup = formation_info["best_lineup"]

    # -------------------------------------------------------------
    # Riga 1: Modulo Base & Titolari Congeniali
    # -------------------------------------------------------------
    c_form, c_lineup = st.columns([1, 2.3])
    with c_form:
        st.markdown(
            f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-left: 4px solid {OPTA_GREEN}; border-radius: 4px; padding: 20px; height: 100%;">'
            f'<div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">SCHIERAMENTO BASE</div>'
            f'<div style="font-size: 2.3rem; font-weight: 800; color: {OPTA_GREEN}; margin: 8px 0;">{dominant_form}</div>'
            f'<div style="font-size: 0.85rem; color: {OPTA_TEXT};">Minutaggio cumulato: <b>{formation_info["total_minutes"]} minuti</b></div>'
            f'<div style="font-size: 0.78rem; color: {OPTA_MUTED}; margin-top: 6px;">Schema tattico predominante rilevato nell\'archivio gare analizzato.</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c_lineup:
        st.markdown(
            f'<div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-left: 4px solid {OPTA_GREEN}; border-radius: 4px; padding: 14px 18px; margin-bottom: 12px;">'
            f'<div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_GREEN}; text-transform: uppercase; letter-spacing: 0.08em;">STARTING XI TACTICAL FIT</div>'
            f'<div style="font-size: 1.05rem; font-weight: 800; color: {OPTA_TEXT}; margin-top: 2px;">Undici di Riferimento (Congeniale per Ruolo)</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        if best_lineup:
            # Calcolo metrico integrato degli eventi e del baricentro X reale per titolare
            t_ev = events_df[events_df["team"] == team_name].copy()
            if "x" in t_ev.columns:
                t_ev["x_m"] = pd.to_numeric(t_ev["x"], errors="coerce") * CONV_X
            else:
                t_ev["x_m"] = 50.0

            reparto_order = {
                "GK": 1,
                "CB": 2, "RCB": 2, "LCB": 2, "RB": 2, "LB": 2, "RWB": 2, "LWB": 2,
                "DM": 3, "RDM": 3, "LDM": 3, "CM": 3, "RCM": 3, "LCM": 3, "RM": 3, "LM": 3, "AM": 3, "RAM": 3, "LAM": 3,
                "RW": 4, "LW": 4, "SS": 4, "CF": 4, "ST": 4, "RST": 4, "LST": 4
            }

            lineup_rows = []
            for pos_name, info in best_lineup.items():
                p_name = info["player"]
                p_mins = f"{info['minutes']}'"
                sigla = POSITION_ACRONYMS.get(pos_name, pos_name)

                p_ev = t_ev[t_ev["player"] == p_name]
                vol_events = len(p_ev)
                med_x = float(p_ev["x_m"].median()) if not p_ev.empty and p_ev["x_m"].notna().any() else 50.0

                lineup_rows.append({
                    "player": p_name,
                    "pos_name": pos_name,
                    "sigla": sigla,
                    "minutes": p_mins,
                    "volume": vol_events,
                    "height_x": round(med_x, 1),
                    "rep_rank": reparto_order.get(sigla, 5)
                })

            df_table_xi = pd.DataFrame(lineup_rows).sort_values(["rep_rank", "volume"], ascending=[True, False])

            # Costruzione righe HTML compatte
            table_rows_html = []
            for _, r in df_table_xi.iterrows():
                badge_style = get_role_badge_style(r["sigla"])
                row_html = (
                    f'<tr style="border-bottom: 1px solid #1e2630;">'
                    f'<td style="padding: 8px 12px; text-align: center;">'
                    f'<span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 800; {badge_style}">'
                    f'{r["sigla"]}'
                    f'</span>'
                    f'</td>'
                    f'<td style="padding: 8px 12px; font-weight: 700; color: {OPTA_TEXT};">'
                    f'{r["player"]}'
                    f'<div style="font-size: 0.70rem; color: {OPTA_MUTED}; font-weight: 400;">{r["pos_name"]}</div>'
                    f'</td>'
                    f'<td style="padding: 8px 12px; text-align: center; color: #8b949e; font-size: 0.82rem;">{r["minutes"]}</td>'
                    f'<td style="padding: 8px 12px; text-align: center; color: {OPTA_TEXT}; font-weight: 600;">{r["volume"]}</td>'
                    f'<td style="padding: 8px 12px; text-align: center; color: {OPTA_GREEN}; font-weight: 700;">{r["height_x"]} m</td>'
                    f'</tr>'
                )
                table_rows_html.append(row_html)

            html_table = (
                f'<div style="background: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 6px; overflow-x: auto; margin-bottom: 10px;">'
                f'<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.84rem; margin: 0; padding: 0;">'
                f'<thead>'
                f'<tr style="background: #18222c; border-bottom: 1.5px solid {OPTA_BORDER}; font-size: 0.70rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;">'
                f'<th style="padding: 8px 12px; text-align: center;">Ruolo</th>'
                f'<th style="padding: 8px 12px;">Calciatore</th>'
                f'<th style="padding: 8px 12px; text-align: center;">Impiego</th>'
                f'<th style="padding: 8px 12px; text-align: center;">Eventi</th>'
                f'<th style="padding: 8px 12px; text-align: center;">Baricentro (X)</th>'
                f'</tr>'
                f'</thead>'
                f'<tbody>'
                f'{"".join(table_rows_html)}'
                f'</tbody>'
                f'</table>'
                f'</div>'
            )
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.info("Nessuna formazione congeniale rilevata.")

    st.write("")

    # -------------------------------------------------------------
    # Calcolo Dati Rete e Definizione della variabile nodes_df
    # -------------------------------------------------------------
    network_data = compute_lineup_passing_network(events_df, team_name, best_lineup, min_pass_threshold=3)
    nodes_df = network_data["nodes"]

    # -------------------------------------------------------------
    # Riga 2: Passing Network & Pitch Control Probabilistico
    # -------------------------------------------------------------
    col_net, col_voro = st.columns(2)

    with col_net:
        st.markdown(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">RETE DEI PASSAGGI (MPLSOCCER)</div>
                    <div style="font-size: 0.75rem; color: {OPTA_GREEN};">Pitch 105m &times; 68m</div>
                </div>
                <div style="font-size: 0.78rem; color: {OPTA_MUTED}; margin-bottom: 12px;">Spessore linee proporzionale alla frequenza degli scambi tra i titolari.</div>
        """, unsafe_allow_html=True)
        if not nodes_df.empty:
            fig_network = plot_best_xi_network(network_data, team_name, dominant_form)
            st.pyplot(fig_network, width='stretch')
        else:
            st.warning("Dati insufficienti per generare la rete dei passaggi.")
        st.markdown("</div>", unsafe_allow_html=True)

    df_voronoi_res = pd.DataFrame()
    with col_voro:
        st.markdown(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">PITCH CONTROL PROBABILISTICO</div>
                    <div style="font-size: 0.75rem; color: {OPTA_GREEN};">Modello Time-to-Intercept</div>
                </div>
                <div style="font-size: 0.78rem; color: {OPTA_MUTED}; margin-bottom: 12px;">Voronoi calcolato su velocità per ruolo e tempo di reazione (fissato in 0.5s).</div>
        """, unsafe_allow_html=True)
        if not nodes_df.empty:
            fig_voro, df_voronoi_res = plot_team_voronoi_spacing(nodes_df, team_name, dominant_form)
            st.pyplot(fig_voro, width='stretch')
        else:
            st.warning("Dati insufficienti per generare il modello di Pitch Control.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # -------------------------------------------------------------
    # Riga 3: Tabella delle Metriche di Pitch Control
    # -------------------------------------------------------------
    if not df_voronoi_res.empty:
        st.markdown(f"""
            <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; margin-bottom: 20px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">REPORT PITCH CONTROL: METRICHE DI DOMINANZA TERRITORIALE</div>
                <div style="font-size: 0.8rem; color: {OPTA_MUTED}; margin-bottom: 12px;">Superficie metrica controllata da ciascun interprete su campo regolamentare FIFA (7.140 m² complessivi) integrando la velocità di sprint stimata.</div>
        """, unsafe_allow_html=True)
        st.dataframe(df_voronoi_res, width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Riga 4: Matrice di Progressione e Ranking Giocatori Chiave (Plotly)
    # -------------------------------------------------------------
    if not nodes_df.empty:
        col_mat, col_rank = st.columns([1.2, 1])

        with col_mat:
            st.markdown(f"""
                <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">MATRICE DI PROGRESSIONE</div>
                        <div style="font-size: 0.72rem; color: {OPTA_GREEN}; font-weight: 700;">VALUTAZIONE QUADRANTI</div>
                    </div>
                    <div style="font-size: 0.8rem; color: {OPTA_MUTED}; margin-bottom: 12px;">Distribuzione incrociata tra volume di trasmissione e verticalizzazioni utili (> 9.14m).</div>
            """, unsafe_allow_html=True)
            fig_matrix = plot_player_importance_matrix(nodes_df)
            st.plotly_chart(fig_matrix, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        with col_rank:
            st.markdown(f"""
                <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 20px; height: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">INDICE DI INFLUENZA TATTICA (0–100)</div>
                        <div style="font-size: 0.72rem; color: {OPTA_GREEN}; font-weight: 700;">SCORE SINTETICO</div>
                    </div>
                    <div style="font-size: 0.8rem; color: {OPTA_MUTED}; margin-bottom: 12px;">Centralità calcolata su passaggi inviati (40%), ricevuti (25%) e progressivi (35%).</div>
            """, unsafe_allow_html=True)
            fig_rank = plot_player_influence_ranking(nodes_df)
            st.plotly_chart(fig_rank, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")


        # -------------------------------------------------------------
        # Riga 5: VAEP FRAMEWORK (Valuing Actions by Estimating Probabilities)
        # Sostituisce il vecchio recap testuale statico
        # -------------------------------------------------------------
        ideal_players = [info["player"] for info in best_lineup.values()]
        df_vaep = compute_team_vaep(events_df, team_name, ideal_players)

        if not df_vaep.empty:
            df_vaep = df_vaep.merge(
                nodes_df[["player", "position"]], 
                left_on="player", 
                right_on="player", 
                how="left"
            )

            top_overall = df_vaep.sort_values("vaep_totale", ascending=False).iloc[0]
            top_offensive = df_vaep.sort_values("vaep_offensivo", ascending=False).iloc[0]
            top_defensive = df_vaep.sort_values("vaep_difensivo", ascending=False).iloc[0]


            val_tot = top_overall['vaep_totale']
            val_off = top_offensive['vaep_offensivo']
            val_def = top_defensive['vaep_difensivo']

            str_tot = f"+{val_tot:.3f}" if val_tot >= 0 else f"{val_tot:.3f}"
            str_off = f"+{val_off:.3f}" if val_off >= 0 else f"{val_off:.3f}"
            str_def = f"+{val_def:.3f}" if val_def >= 0 else f"{val_def:.3f}"

            vaep_html = textwrap.dedent(f"""
                <div style="background-color: {OPTA_CARD}; border: 1px solid {OPTA_BORDER}; border-left: 4px solid {OPTA_GREEN}; border-radius: 4px; padding: 20px; margin-bottom: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 0.72rem; font-weight: 800; color: {OPTA_MUTED}; text-transform: uppercase; letter-spacing: 0.08em;">VAEP FRAMEWORK (VALUING ACTIONS BY ESTIMATING PROBABILITIES)</div>
                            <div style="font-size: 1.05rem; font-weight: 800; color: {OPTA_TEXT};">Valutazione Globale di Impatto su Ogni Singola Giocata</div>
                        </div>
                        <div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 3px; padding: 5px 12px; font-size: 0.70rem; font-weight: 700; color: {OPTA_GREEN};">
                            ALL-ACTION METRIC
                        </div>
                    </div>
                    <div style="font-size: 0.80rem; color: {OPTA_MUTED}; margin-bottom: 16px; line-height: 1.5;">
                        Il modello <b>VAEP</b> valuta ogni evento (passaggi, contrasti, palle perse, tiri, intercettazioni) quantificando di quanto aumenta la probabilità di segnare (&Delta;P<sub>score</sub>) e di quanto riduce quella di subire (&minus;&Delta;P<sub>concede</sub>).
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                        <div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px;">
                            <div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase; font-weight: 700;">Leader Impatto Globale (VAEP Netto)</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: {OPTA_GREEN}; margin: 2px 0;">{top_overall['player']}</div>
                            <div style="font-size: 0.75rem; color: {OPTA_TEXT};">Valore Netto: <b>{str_tot}</b> ({top_overall['position']})</div>
                            <div style="font-size: 0.70rem; color: {OPTA_MUTED}; margin-top: 4px;">{int(top_overall['azioni_valutate'])} eventi analizzati</div>
                        </div>
                        <div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px;">
                            <div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase; font-weight: 700;">Miglior Creatore (&Delta;P Score)</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: {OPTA_BLUE}; margin: 2px 0;">{top_offensive['player']}</div>
                            <div style="font-size: 0.75rem; color: {OPTA_TEXT};">Valore Offensivo: <b>{str_off}</b> ({top_offensive['position']})</div>
                            <div style="font-size: 0.70rem; color: {OPTA_MUTED}; margin-top: 4px;">Massima pericolosità generata</div>
                        </div>
                        <div style="background: {OPTA_BG}; border: 1px solid {OPTA_BORDER}; border-radius: 4px; padding: 14px;">
                            <div style="font-size: 0.68rem; color: {OPTA_MUTED}; text-transform: uppercase; font-weight: 700;">Pilastro Difensivo (&minus;&Delta;P Concede)</div>
                            <div style="font-size: 1.15rem; font-weight: 800; color: {OPTA_ORANGE}; margin: 2px 0;">{top_defensive['player']}</div>
                            <div style="font-size: 0.75rem; color: {OPTA_TEXT};">Valore Difensivo: <b>{str_def}</b> ({top_defensive['position']})</div>
                            <div style="font-size: 0.70rem; color: {OPTA_MUTED}; margin-top: 4px;">Massima riduzione del rischio avversario</div>
                        </div>
                    </div>
                </div>
            """).strip()

            st.markdown(vaep_html, unsafe_allow_html=True)

            # Tabella Dettagliata VAEP per l'Undici Titolare
            table_display = df_vaep.copy()
            table_display["VAEP Offensivo"] = table_display["vaep_offensivo"].apply(lambda v: f"+{v:.3f}" if v >= 0 else f"{v:.3f}")
            table_display["VAEP Difensivo"] = table_display["vaep_difensivo"].apply(lambda v: f"+{v:.3f}" if v >= 0 else f"{v:.3f}")
            table_display["VAEP Netto"] = table_display["vaep_totale"].apply(lambda v: f"+{v:.3f}" if v >= 0 else f"{v:.3f}")
            table_display["VAEP/Azione"] = table_display["vaep_per_azione"].apply(lambda v: f"{v:+.4f}")
            
            table_clean = table_display[[
                "player", "position", "azioni_valutate", "VAEP Offensivo", "VAEP Difensivo", "VAEP Netto", "VAEP/Azione"
            ]].rename(columns={
                "player": "Calciatore",
                "position": "Ruolo",
                "azioni_valutate": "Eventi Totali"
            })

            st.dataframe(table_clean, width='stretch', hide_index=True)