import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional

# Costanti metriche FIFA 105m x 68m su coordinate StatsBomb (120yd x 80yd)
CONV_X = 105.0 / 120.0
CONV_Y = 68.0 / 80.0

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

ROLE_FALLBACK_GROUPS = {
    "GK": ["GK"],
    "RB": ["RB", "RWB", "RCB", "RM"],
    "RCB": ["RCB", "CB", "RB"],
    "CB": ["CB", "RCB", "LCB"],
    "LCB": ["LCB", "CB", "LB"],
    "LB": ["LB", "LWB", "LCB", "LM"],
    "RWB": ["RWB", "RB", "RM", "RW"],
    "LWB": ["LWB", "LB", "LM", "LW"],
    "DM": ["DM", "RDM", "LDM", "CM"],
    "RDM": ["RDM", "DM", "RCM", "CM"],
    "LDM": ["LDM", "DM", "LCM", "CM"],
    "CM": ["CM", "RCM", "LCM", "DM", "AM"],
    "RCM": ["RCM", "CM", "RDM", "RAM", "RM"],
    "LCM": ["LCM", "CM", "LDM", "LAM", "LM"],
    "RM": ["RM", "RWB", "RW", "RAM", "RCM"],
    "LM": ["LM", "LWB", "LW", "LAM", "LCM"],
    "RAM": ["RAM", "AM", "RW", "RCM", "SS"],
    "LAM": ["LAM", "AM", "LW", "LCM", "SS"],
    "AM": ["AM", "RAM", "LAM", "SS", "CM"],
    "RW": ["RW", "RAM", "RM", "RST", "ST"],
    "LW": ["LW", "LAM", "LM", "LST", "ST"],
    "ST": ["ST", "CF", "RST", "LST", "SS"],
    "CF": ["CF", "ST", "SS", "RST", "LST"],
    "RST": ["RST", "ST", "CF", "SS", "RW"],
    "LST": ["LST", "ST", "CF", "SS", "LW"],
    "SS": ["SS", "CF", "AM", "ST"]
}

SYSTEM_DEFINITIONS = {
    "4-3-3": [
        ("GK", "C", 6.8, 34.0),
        ("RB", "R", 30.0, 58.0), ("RCB", "CR", 25.0, 43.0), ("LCB", "CL", 25.0, 25.0), ("LB", "L", 30.0, 10.0),
        ("DM", "C", 44.0, 34.0), ("RCM", "CR", 54.0, 46.0), ("LCM", "CL", 54.0, 22.0),
        ("RW", "R", 74.0, 56.0), ("ST", "C", 78.0, 34.0), ("LW", "L", 74.0, 12.0)
    ],
    "4-2-3-1": [
        ("GK", "C", 6.8, 34.0),
        ("RB", "R", 30.0, 58.0), ("RCB", "CR", 25.0, 43.0), ("LCB", "CL", 25.0, 25.0), ("LB", "L", 30.0, 10.0),
        ("RDM", "CR", 42.0, 42.0), ("LDM", "CL", 42.0, 26.0),
        ("RAM", "R", 66.0, 55.0), ("AM", "C", 64.0, 34.0), ("LAM", "L", 66.0, 13.0),
        ("ST", "C", 78.0, 34.0)
    ],
    "3-5-2": [
        ("GK", "C", 6.8, 34.0),
        ("RCB", "CR", 25.0, 48.0), ("CB", "C", 23.0, 34.0), ("LCB", "CL", 25.0, 20.0),
        ("RWB", "R", 50.0, 60.0), ("RCM", "CR", 52.0, 44.0), ("DM", "C", 42.0, 34.0), ("LCM", "CL", 52.0, 24.0), ("LWB", "L", 50.0, 8.0),
        ("RST", "CR", 76.0, 42.0), ("LST", "CL", 76.0, 26.0)
    ],
    "3-4-2-1": [
        ("GK", "C", 6.8, 34.0),
        ("RCB", "CR", 25.0, 48.0), ("CB", "C", 23.0, 34.0), ("LCB", "CL", 25.0, 20.0),
        ("RWB", "R", 50.0, 60.0), ("RCM", "CR", 46.0, 40.0), ("LCM", "CL", 46.0, 28.0), ("LWB", "L", 50.0, 8.0),
        ("RAM", "CR", 66.0, 44.0), ("LAM", "CL", 66.0, 24.0),
        ("ST", "C", 78.0, 34.0)
    ],
    "4-4-2": [
        ("GK", "C", 6.8, 34.0),
        ("RB", "R", 30.0, 58.0), ("RCB", "CR", 25.0, 43.0), ("LCB", "CL", 25.0, 25.0), ("LB", "L", 30.0, 10.0),
        ("RM", "R", 52.0, 58.0), ("RCM", "CR", 48.0, 42.0), ("LCM", "CL", 48.0, 26.0), ("LM", "L", 52.0, 10.0),
        ("RST", "CR", 76.0, 42.0), ("LST", "CL", 76.0, 26.0)
    ]
}

def is_valid_11_men_formation(form_str: str) -> bool:
    if not form_str or not isinstance(form_str, str):
        return False
    parts = [int(p) for p in form_str.split("-") if p.isdigit()]
    return len(parts) in [3, 4] and sum(parts) == 10

def normalize_formation_code(val: Any) -> str:
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip().replace(".0", "")
    if "-" in s:
        return s if is_valid_11_men_formation(s) else ""
    digits = [c for c in s if c.isdigit()]
    if len(digits) in [3, 4]:
        cand = "-".join(digits)
        return cand if is_valid_11_men_formation(cand) else ""
    return ""

def get_available_team_formations(events_df: pd.DataFrame, team_name: str) -> List[str]:
    t_ev = events_df[events_df["team"] == team_name].copy()
    if t_ev.empty:
        return ["4-3-3", "4-2-3-1", "3-5-2"]

    formation_counts = {}
    if "tactics" in t_ev.columns:
        for t in t_ev["tactics"].dropna():
            if isinstance(t, dict):
                f_norm = normalize_formation_code(t.get("formation"))
                if f_norm and is_valid_11_men_formation(f_norm):
                    formation_counts[f_norm] = formation_counts.get(f_norm, 0) + 10

    for col in ["tactics_formation", "formation", "team_formation"]:
        if col in t_ev.columns:
            for val in t_ev[col].dropna().unique():
                f_norm = normalize_formation_code(val)
                if f_norm and is_valid_11_men_formation(f_norm):
                    formation_counts[f_norm] = formation_counts.get(f_norm, 0) + 5

    valid_found = [f[0] for f in sorted(formation_counts.items(), key=lambda x: x[1], reverse=True)]
    defaults = ["4-3-3", "4-2-3-1", "3-5-2", "3-4-2-1", "4-4-2"]
    for d in defaults:
        if d not in valid_found:
            valid_found.append(d)

    return valid_found[:5]

def get_team_players_list(events_df: pd.DataFrame, team_name: str) -> List[str]:
    t_ev = events_df[events_df["team"] == team_name]
    if "player" not in t_ev.columns:
        return []
    counts = t_ev["player"].dropna().value_counts()
    return counts.index.tolist()

def get_formation_line_groups(formation: str) -> Dict[str, List[Tuple[str, str]]]:
    sys_template = SYSTEM_DEFINITIONS.get(formation, SYSTEM_DEFINITIONS["4-3-3"])
    groups = {
        "Portiere": [],
        "Difesa": [],
        "Centrocampo": [],
        "Attacco": []
    }
    for role, ch, bx, by in sys_template:
        if role == "GK":
            groups["Portiere"].append((role, "Portiere (GK)"))
        elif any(k in role for k in ["B", "CB", "WB"]):
            groups["Difesa"].append((role, f"Difensore ({role})"))
        elif any(k in role for k in ["M", "DM", "CM", "AM"]):
            groups["Centrocampo"].append((role, f"Centrocampista ({role})"))
        else:
            groups["Attacco"].append((role, f"Attaccante ({role})"))
    return {k: v for k, v in groups.items() if len(v) > 0}

def get_role_candidates(events_df: pd.DataFrame, team_name: str, role: str, max_candidates: int = 6) -> List[str]:
    if events_df is None or events_df.empty:
        return []

    team_col = "team" if "team" in events_df.columns else ("team_name" if "team_name" in events_df.columns else None)
    if not team_col:
        return []

    target_clean = str(team_name).strip().lower()
    t_ev = events_df[events_df[team_col].astype(str).str.strip().str.lower() == target_clean].copy()
    if t_ev.empty:
        t_ev = events_df[
            events_df[team_col].astype(str).str.strip().str.lower().str.contains(target_clean.split()[0], na=False)
        ].copy()

    if t_ev.empty:
        return []

    if "player" not in t_ev.columns and "player_name" in t_ev.columns:
        t_ev["player"] = t_ev["player_name"]
    elif "player" in t_ev.columns:
        t_ev["player"] = t_ev["player"].apply(
            lambda p: p.get("name") if isinstance(p, dict) else (str(p) if pd.notna(p) and str(p) != "nan" else np.nan)
        )

    t_ev = t_ev[t_ev["player"].notna()].copy()
    records = []

    if "tactics" in t_ev.columns:
        for t_dict in t_ev["tactics"].dropna():
            if isinstance(t_dict, dict) and "lineup" in t_dict:
                for item in t_dict["lineup"]:
                    p_name = item.get("player", {}).get("name") if isinstance(item.get("player"), dict) else item.get("player")
                    pos_val = item.get("position", {}).get("name") if isinstance(item.get("position"), dict) else item.get("position")
                    if p_name and pos_val:
                        sigla = POSITION_ACRONYMS.get(str(pos_val).strip(), str(pos_val).strip())
                        records.append({"player": p_name, "sigla": sigla, "vol": 50})

    if "position" in t_ev.columns:
        for _, row in t_ev[t_ev["position"].notna()].iterrows():
            pos_val = row["position"]
            p_name = row["player"]
            pos_str = pos_val.get("name", "") if isinstance(pos_val, dict) else str(pos_val)
            if pos_str and pos_str not in ["nan", "None", "Substitute"]:
                sigla = POSITION_ACRONYMS.get(pos_str.strip(), pos_str.strip())
                records.append({"player": p_name, "sigla": sigla, "vol": 1})

    if not records:
        return []

    df_pos = pd.DataFrame(records)
    affines = ROLE_FALLBACK_GROUPS.get(role, [role])
    matched = df_pos[df_pos["sigla"].isin(affines)]

    if matched.empty:
        if role == "GK":
            macro_filter = df_pos["sigla"] == "GK"
        elif any(k in role for k in ["B", "CB", "WB"]):
            macro_filter = df_pos["sigla"].str.contains("B|CB|WB", na=False)
        elif any(k in role for k in ["M", "DM", "CM", "AM"]):
            macro_filter = df_pos["sigla"].str.contains("M|DM|CM|AM", na=False)
        else:
            macro_filter = df_pos["sigla"].str.contains("ST|CF|RW|LW|SS", na=False)
        matched = df_pos[macro_filter]

    if matched.empty:
        return []

    player_vol = matched.groupby("player")["vol"].sum().sort_values(ascending=False)
    return player_vol.index.tolist()[:max_candidates]

def apply_match_context_filters(
    events_df: pd.DataFrame,
    team_name: str,
    minute_range: Tuple[int, int] = (0, 100),
    game_state: str = "ALL"
) -> pd.DataFrame:
    """
    Filtra gli eventi per finestra temporale (minute_range) e Game State (TIED, LEADING, TRAILING).
    Calcola il punteggio cumulativo per match_id per classificare l'inerzia di gara[cite: 4].
    """
    if events_df.empty:
        return events_df

    df_ev = events_df.copy()

    # Filtro temporale (minuti)[cite: 4]
    if "minute" in df_ev.columns:
        min_start, min_end = minute_range
        df_ev = df_ev[(df_ev["minute"] >= min_start) & (df_ev["minute"] <= min_end)]

    if game_state.upper() == "ALL" or "match_id" not in df_ev.columns or "type" not in df_ev.columns:
        return df_ev

    target_clean = str(team_name).strip().lower()
    df_ev["is_goal"] = (df_ev["type"] == "Shot") & (df_ev.get("shot_outcome", "").astype(str).str.lower() == "goal")
    
    match_states = []
    for match_id, m_group in df_ev.groupby("match_id"):
        m_group = m_group.sort_values(["minute", "second"] if "second" in m_group.columns else "minute")
        
        our_goals = 0
        opp_goals = 0
        states = []

        for _, ev in m_group.iterrows():
            ev_team = str(ev.get("team", "")).strip().lower()
            if ev["is_goal"]:
                if ev_team == target_clean:
                    our_goals += 1
                else:
                    opp_goals += 1
            
            diff = our_goals - opp_goals
            if diff == 0:
                s = "TIED"
            elif diff > 0:
                s = "LEADING"
            else:
                s = "TRAILING"
            states.append(s)

        m_group["game_state"] = states
        match_states.append(m_group)

    if match_states:
        df_scored = pd.concat(match_states, ignore_index=True)
        filtered = df_scored[df_scored["game_state"] == game_state.upper()]
        return filtered if not filtered.empty else df_ev
    return df_ev

def extract_real_formation_nodes(
    events_df: pd.DataFrame, 
    team_name: str, 
    formation: str, 
    is_opponent: bool = False,
    custom_lineup: Optional[Dict[str, str]] = None,
    minute_range: Tuple[int, int] = (0, 100),
    game_state: str = "ALL"
) -> pd.DataFrame:
    """
    Calcola le posizioni reali empiriche dei singoli interpreti, supportando:
    - Personalizzazione della lineup[cite: 4]
    - Finestra temporale di gioco (minute_range)[cite: 4]
    - Condizione di punteggio (game_state: TIED, LEADING, TRAILING)[cite: 4]
    """
    t_ev = events_df[
        (events_df["team"] == team_name) & 
        (events_df["x"].notna()) & 
        (events_df["y"].notna())
    ].copy()

    t_ev = apply_match_context_filters(t_ev, team_name, minute_range=minute_range, game_state=game_state)

    if "pass_type" in t_ev.columns:
        t_ev = t_ev[~t_ev["pass_type"].isin(["Corner", "Throw-in", "Goal Kick"])]
    if "type" in t_ev.columns:
        t_ev = t_ev[t_ev["type"].isin([
            "Pass", "Ball Receipt", "Carry", "Duel", "Interception", "Ball Recovery", "Clearance", "Block"
        ])]

    t_ev["x_m"] = t_ev["x"] * CONV_X
    t_ev["y_m"] = t_ev["y"] * CONV_Y

    outfield = t_ev[t_ev["x_m"] > 16.5]
    real_cx = float(outfield["x_m"].mean()) if not outfield.empty else 48.0
    real_cy_spread = float(outfield["y_m"].std()) if not outfield.empty else 16.0

    right_x = float(outfield[outfield["y_m"] > 45.4]["x_m"].mean()) if not outfield[outfield["y_m"] > 45.4].empty else real_cx
    left_x = float(outfield[outfield["y_m"] < 22.6]["x_m"].mean()) if not outfield[outfield["y_m"] < 22.6].empty else real_cx
    delta_flank_r = right_x - real_cx
    delta_flank_l = left_x - real_cx

    player_role_stats = []
    if "player" in t_ev.columns and "position" in t_ev.columns:
        valid_p = t_ev[t_ev["position"].notna() & (~t_ev["position"].isin(["Substitute", "nan"]))].copy()
        valid_p["sigla"] = valid_p["position"].map(POSITION_ACRONYMS).fillna(valid_p["position"])

        for (p_name, sigla), grp in valid_p.groupby(["player", "sigla"]):
            player_role_stats.append({
                "player": p_name,
                "role": sigla,
                "count": len(grp),
                "med_x": float(grp["x_m"].median()),
                "med_y": float(grp["y_m"].median())
            })
    df_prs = pd.DataFrame(player_role_stats)

    sys_template = SYSTEM_DEFINITIONS.get(formation, SYSTEM_DEFINITIONS["4-3-3"])
    assigned_players = set()
    nodes = []

    for idx, (role, ch, base_x, base_y) in enumerate(sys_template):
        chosen_player = None
        player_x = None
        player_y = None

        if custom_lineup and role in custom_lineup and custom_lineup[role]:
            chosen_player = custom_lineup[role]
            assigned_players.add(chosen_player)
            p_events = t_ev[t_ev["player"] == chosen_player]
            if not p_events.empty:
                player_x = float(p_events["x_m"].median())
                player_y = float(p_events["y_m"].median())

        if chosen_player is None and not df_prs.empty:
            exact_cand = df_prs[(df_prs["role"] == role) & (~df_prs["player"].isin(assigned_players))].sort_values("count", ascending=False)
            if exact_cand.empty:
                affines = ROLE_FALLBACK_GROUPS.get(role, [role])
                exact_cand = df_prs[(df_prs["role"].isin(affines)) & (~df_prs["player"].isin(assigned_players))].sort_values("count", ascending=False)
            if exact_cand.empty:
                exact_cand = df_prs[~df_prs["player"].isin(assigned_players)].sort_values("count", ascending=False)

            if not exact_cand.empty:
                best_row = exact_cand.iloc[0]
                chosen_player = best_row["player"]
                player_x = best_row["med_x"]
                player_y = best_row["med_y"]
                assigned_players.add(chosen_player)

        if chosen_player is None:
            chosen_player = role

        if role == "GK":
            node_x = 6.8
            node_y = 34.0
        else:
            dx = (real_cx - 48.0) * (base_x / 70.0)
            flank_shift = delta_flank_r * 0.4 if ch == "R" else (delta_flank_l * 0.4 if ch == "L" else 0.0)
            nominal_x = np.clip(base_x + dx + flank_shift, 16.0, 96.0)
            nominal_y = np.clip(34.0 + (base_y - 34.0) * (real_cy_spread / 16.0), 5.0, 63.0)

            if player_x is not None and not np.isnan(player_x):
                node_x = round(float(0.70 * player_x + 0.30 * nominal_x), 2)
                node_y = round(float(0.70 * player_y + 0.30 * nominal_y), 2)
            else:
                node_x = round(float(nominal_x), 2)
                node_y = round(float(nominal_y), 2)

        sur = str(chosen_player).split()[-1] if len(str(chosen_player).split()) > 0 else chosen_player

        nodes.append({
            "role": role,
            "player": sur,
            "x_mean": node_x,
            "y_mean": node_y
        })

    df_res = pd.DataFrame(nodes).head(11).copy()

    if is_opponent:
        df_res["x_mean"] = (105.0 - df_res["x_mean"]).round(2)
        df_res["y_mean"] = (68.0 - df_res["y_mean"]).round(2)

    return df_res

def calculate_tactical_zones_matrix(nodes_our: pd.DataFrame, nodes_opp: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_bins = [0.0, 13.6, 27.2, 40.8, 54.4, 68.0]
    channel_names = ["Fascia Sx", "Half-Space Sx", "Centro", "Half-Space Dx", "Fascia Dx"]
    x_bins = [0.0, 35.0, 70.0, 105.0]
    third_names = ["Costruzione Nostra", "Mediana / Sviluppo", "Rifinitura Nostra"]

    zone_matrix = np.zeros((3, 5))

    pts_our = nodes_our[nodes_our["role"] != "GK"][["x_mean", "y_mean"]].values
    pts_opp = nodes_opp[nodes_opp["role"] != "GK"][["x_mean", "y_mean"]].values

    for i in range(3):
        x_c = (x_bins[i] + x_bins[i+1]) / 2.0
        for j in range(5):
            y_c = (y_bins[j] + y_bins[j+1]) / 2.0
            
            d_our = np.sqrt((pts_our[:, 0] - x_c)**2 + (pts_our[:, 1] - y_c)**2)
            w_our = np.sum(np.exp(-(d_our**2) / (2 * 12.0**2)))

            d_opp = np.sqrt((pts_opp[:, 0] - x_c)**2 + (pts_opp[:, 1] - y_c)**2)
            w_opp = np.sum(np.exp(-(d_opp**2) / (2 * 12.0**2)))

            tot = w_our + w_opp + 1e-5
            zone_matrix[i, j] = (w_our - w_opp) / tot

    mismatches = []
    for j in range(5):
        val = zone_matrix[1, j]
        ch = channel_names[j]
        if val > 0.20:
            mismatches.append(f"<b>{ch}</b>: Superiorità numerica nostra (+1 uomo / overload). Ottima corsia di progressione.")
        elif val < -0.20:
            mismatches.append(f"<b>{ch}</b>: Densità avversaria superiore. Rischio di imbuto e perdita del possesso.")
        else:
            mismatches.append(f"<b>{ch}</b>: Parità e duelli individuali diretti (1v1).")

    return zone_matrix, {
        "channel_names": channel_names,
        "third_names": third_names,
        "mismatches": mismatches
    }

def simulate_tactical_clash(
    events_df: pd.DataFrame,
    our_team: str,
    opp_team: str,
    our_formation: str,
    opp_formation: str,
    p_our: Dict[str, Any],
    p_opp: Dict[str, Any],
    n_simulations: int = 1000,
    custom_our_lineup: Optional[Dict[str, str]] = None,
    custom_opp_lineup: Optional[Dict[str, str]] = None,
    minute_range: Tuple[int, int] = (0, 100),
    game_state: str = "ALL"
) -> Dict[str, Any]:
    """
    Esegue la simulazione predittiva integrando forzature dei titolari,
    finestra temporale (minute_range) e Game State (TIED, LEADING, TRAILING)[cite: 4].
    """
    df_nodes_our = extract_real_formation_nodes(
        events_df, our_team, our_formation, is_opponent=False, 
        custom_lineup=custom_our_lineup, minute_range=minute_range, game_state=game_state
    )
    df_nodes_opp = extract_real_formation_nodes(
        events_df, opp_team, opp_formation, is_opponent=True, 
        custom_lineup=custom_opp_lineup, minute_range=minute_range, game_state=game_state
    )

    zone_mat, mismatch_info = calculate_tactical_zones_matrix(df_nodes_our, df_nodes_opp)

    mid_roles = ["DM", "RDM", "LDM", "CM", "RCM", "LCM", "AM", "RAM", "LAM", "RM", "LM", "MID"]
    our_mids = len(df_nodes_our[df_nodes_our["role"].isin(mid_roles)])
    opp_mids = len(df_nodes_opp[df_nodes_opp["role"].isin(mid_roles)])
    midfield_delta = our_mids - opp_mids

    h_our = float(df_nodes_our[df_nodes_our["role"] != "GK"]["x_mean"].mean())
    h_opp_real = float(105.0 - df_nodes_opp[df_nodes_opp["role"] != "GK"]["x_mean"].mean())

    base_our_xg = max(float(p_our.get("xg_p90", 1.4)), 0.4)
    base_opp_xg = max(float(p_opp.get("xg_p90", 1.2)), 0.4)

    control_boost = float(np.mean(zone_mat[1:, :])) * 0.15
    mod_our = 1.0 + (midfield_delta * 0.04) + control_boost
    mod_opp = 1.0 - (midfield_delta * 0.04) - control_boost

    np.random.seed(42)
    sim_our = np.random.gamma(shape=base_our_xg * mod_our * 3, scale=1/3, size=n_simulations)
    sim_opp = np.random.gamma(shape=base_opp_xg * mod_opp * 3, scale=1/3, size=n_simulations)

    prob_win = float(np.mean(sim_our > sim_opp) * 100)
    prob_draw = float(np.mean(np.abs(sim_our - sim_opp) < 0.25) * 100)
    prob_loss = max(100.0 - (prob_win + prob_draw), 0.0)

    base_tilt = float(p_our.get("field_tilt", 50.0))
    pred_tilt = np.clip(base_tilt + (midfield_delta * 3.0) + (control_boost * 40.0), 25.0, 75.0)

    return {
        "our_formation": our_formation,
        "opp_formation": opp_formation,
        "nodes_our": df_nodes_our,
        "nodes_opp": df_nodes_opp,
        "zone_matrix": zone_mat,
        "mismatch_info": mismatch_info,
        "expected_our_xg": round(float(np.mean(sim_our)), 2),
        "expected_opp_xg": round(float(np.mean(sim_opp)), 2),
        "prob_win": round(prob_win, 1),
        "prob_draw": round(prob_draw, 1),
        "prob_loss": round(prob_loss, 1),
        "predicted_field_tilt": round(pred_tilt, 1),
        "midfield_delta": midfield_delta,
        "baricentro_our": round(h_our, 1),
        "baricentro_opp": round(h_opp_real, 1)
    }

# =====================================================================
# WRAPPER CON CACHE DETERMINISTICA STREAMLIT (@st.cache_data)
# =====================================================================

@st.cache_data(show_spinner=False)
def simulate_tactical_clash_cached(
    events_df: pd.DataFrame,
    our_team: str,
    opp_team: str,
    our_formation: str,
    opp_formation: str,
    p_our_tuple: tuple,
    p_opp_tuple: tuple,
    custom_our_items: tuple,
    custom_opp_items: tuple,
    minute_range: Tuple[int, int] = (0, 100),
    game_state: str = "ALL"
) -> Dict[str, Any]:
    """
    Wrapper deterministico con argomenti immutabili (tuple) compatibili con @st.cache_data.
    Elimina i ricalcoli di Monte Carlo e della matrice zonale al cambio tab.
    """
    p_our = dict(p_our_tuple)
    p_opp = dict(p_opp_tuple)
    custom_our = dict(custom_our_items) if custom_our_items else None
    custom_opp = dict(custom_opp_items) if custom_opp_items else None

    return simulate_tactical_clash(
        events_df=events_df,
        our_team=our_team,
        opp_team=opp_team,
        our_formation=our_formation,
        opp_formation=opp_formation,
        p_our=p_our,
        p_opp=p_opp,
        custom_our_lineup=custom_our,
        custom_opp_lineup=custom_opp,
        minute_range=minute_range,
        game_state=game_state
    )

# =====================================================================
# GESTIONE PRESET TATTICI DELLO STAFF (SERIALIZZAZIONE JSON)
# =====================================================================

def export_tactical_preset(
    preset_name: str,
    formation: str,
    lineup_dict: Dict[str, Optional[str]],
    time_phase: str,
    game_state: str
) -> str:
    """Esporta un assetto tattico dello staff in formato JSON standard."""
    payload = {
        "preset_name": preset_name,
        "formation": formation,
        "lineup": {k: v for k, v in lineup_dict.items() if v is not None},
        "time_phase": time_phase,
        "game_state": game_state
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)

def parse_tactical_preset(json_str: str) -> Optional[Dict[str, Any]]:
    """Carica e valida un preset tattico da stringa JSON."""
    try:
        data = json.loads(json_str)
        if "formation" in data and "lineup" in data:
            return data
    except Exception:
        return None
    return None

FORMATION_TEMPLATES = {}