import os
import json
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

STATSBOMB_DIR = os.path.join("statsbomb", "data")
CACHE_DIR = "data"

def log_telemetry(msg: str):
    print(f"[TELEMETRIA DATA_LOADER] {msg}", flush=True)

def ensure_cache_directory() -> str:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR

def get_data_source_status() -> Dict[str, Any]:
    """
    Rileva lo stato di funzionamento e la provenienza effettiva dei dati.
    """
    ensure_cache_directory()
    has_statsbomb = os.path.exists(STATSBOMB_DIR)
    
    parquet_matches = [f for f in os.listdir(CACHE_DIR) if f.startswith("matches_") and f.endswith(".parquet")]
    parquet_events = [f for f in os.listdir(CACHE_DIR) if f.startswith("events_") and f.endswith(".parquet")]
    has_comps_parquet = os.path.exists(os.path.join(CACHE_DIR, "competitions.parquet"))
    
    if has_statsbomb:
        mode = "FULL ARCHIVE (LOCALE)"
        active_source = STATSBOMB_DIR
        desc = "Lettura da archivio completo JSON con caching attivo su Parquet"
    else:
        mode = "SHOWCASE / CLOUD"
        active_source = CACHE_DIR
        desc = "Lettura ottimizzata esclusivamente da file Parquet pre-calcolati"
        
    return {
        "mode": mode,
        "active_source": active_source,
        "has_statsbomb": has_statsbomb,
        "has_comps_parquet": has_comps_parquet,
        "matches_parquet_count": len(parquet_matches),
        "events_parquet_count": len(parquet_events),
        "description": desc
    }

def render_data_source_telemetry(sidebar: bool = True):
    """
    Mostra un badge visivo di telemetria per lo staff nella webapp.
    """
    status = get_data_source_status()
    target_container = st.sidebar if sidebar else st
    
    badge_color = "#00ff85" if status["has_statsbomb"] else "#00b0ff"
    
    html_telemetry = f"""
    <div style="background-color: #161b22; border: 1px solid #2a2a2a; border-left: 3px solid {badge_color}; border-radius: 4px; padding: 10px 12px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.65rem; font-weight: 800; color: #8e8e93; text-transform: uppercase; letter-spacing: 0.05em;">DATA ENGINE SOURCE</span>
            <span style="font-size: 0.65rem; font-weight: 700; color: {badge_color};">{status['mode']}</span>
        </div>
        <div style="font-size: 0.78rem; font-weight: 700; color: #f8f9fa; margin: 4px 0 2px 0; word-break: break-all;">
            📂 <code>{status['active_source']}</code>
        </div>
        <div style="font-size: 0.68rem; color: #8e8e93; line-height: 1.3;">
            {status['description']}<br/>
            Indice Parquet: <b>{status['events_parquet_count']}</b> gare pronte
        </div>
    </div>
    """
    target_container.markdown(html_telemetry, unsafe_allow_html=True)

def _sanitize_df_for_parquet(df: pd.DataFrame, context_name: str = "") -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(
                lambda val: str(val) if isinstance(val, (dict, list, tuple, set)) else val
            )
            df[col] = df[col].astype(str).replace({
                "nan": None, 
                "None": None, 
                "<NA>": None,
                "": None
            })
    return df

# =====================================================================
# 1. CARICAMENTO COMPETIZIONI
# =====================================================================

def get_competitions() -> pd.DataFrame:
    ensure_cache_directory()
    cache_path = os.path.join(CACHE_DIR, "competitions.parquet")
    
    # 1. Lettura da cache Parquet (Directory: data/)
    if os.path.exists(cache_path):
        log_telemetry(f"Caricamento competizioni da CACHE PARQUET: {cache_path}")
        comps = pd.read_parquet(cache_path)
        
        # In assenza della cartella statsbomb, filtra solo i tornei presenti in data/
        if not os.path.exists(STATSBOMB_DIR):
            available_matches_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("matches_") and f.endswith(".parquet")]
            valid_pairs = []
            for f in available_matches_files:
                parts = f.replace("matches_", "").replace(".parquet", "").split("_")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    valid_pairs.append((int(parts[0]), int(parts[1])))
            
            if valid_pairs:
                mask = comps.apply(lambda r: (int(r["competition_id"]), int(r["season_id"])) in valid_pairs, axis=1)
                filtered_comps = comps[mask].copy()
                if not filtered_comps.empty:
                    log_telemetry(f"Filtro Showcase applicato: {len(filtered_comps)} competizioni identificate da Parquet.")
                    return filtered_comps

        return comps
    
    # 2. Lettura da archivio grezzo (Directory: statsbomb/data/)
    comp_file = os.path.join(STATSBOMB_DIR, "competitions.json")
    if os.path.exists(comp_file):
        log_telemetry(f"Caricamento competizioni da RAW JSON STATSBOMB: {comp_file}")
        with open(comp_file, "r", encoding="utf-8") as f:
            comps_data = json.load(f)

        comps = pd.DataFrame(comps_data)
        comps = _sanitize_df_for_parquet(comps, "competitions")
        comps.to_parquet(cache_path, index=False)
        log_telemetry(f"Generato file Parquet di cache in: {cache_path}")
        return comps

    raise FileNotFoundError(
        f"Impossibile inizializzare le competizioni: né '{cache_path}' né '{comp_file}' esistono."
    )

# =====================================================================
# 2. CARICAMENTO GARE (MATCHES)
# =====================================================================

def get_matches_for_competition(competition_id: int, season_id: int) -> pd.DataFrame:
    ensure_cache_directory()
    filename = f"matches_{competition_id}_{season_id}.parquet"
    cache_path = os.path.join(CACHE_DIR, filename)
    
    # 1. Lettura da cache Parquet (Directory: data/)
    if os.path.exists(cache_path):
        log_telemetry(f"Caricamento gare da CACHE PARQUET: {cache_path}")
        matches_df = pd.read_parquet(cache_path)
        
        # Filtro sulle sole partite che dispongono del parquet eventi in assenza di statsbomb/
        if not os.path.exists(STATSBOMB_DIR):
            available_events = set()
            for f in os.listdir(CACHE_DIR):
                if f.startswith("events_") and f.endswith(".parquet"):
                    mid = f.replace("events_", "").replace(".parquet", "")
                    if mid.isdigit():
                        available_events.add(int(mid))
            
            if available_events:
                filtered = matches_df[matches_df["match_id"].isin(available_events)].copy()
                if not filtered.empty:
                    log_telemetry(f"Gare filtrate per eventi pre-caricati: {len(filtered)}/{len(matches_df)}")
                    return filtered

        return matches_df
        
    # 2. Lettura da archivio grezzo (Directory: statsbomb/data/matches/{comp_id}/{season_id}.json)
    matches_file = os.path.join(STATSBOMB_DIR, "matches", str(competition_id), f"{season_id}.json")
    if not os.path.exists(matches_file):
        raise FileNotFoundError(f"File gare non trovato in: {matches_file} né in {cache_path}")

    log_telemetry(f"Caricamento gare da RAW JSON STATSBOMB: {matches_file}")
    with open(matches_file, "r", encoding="utf-8") as f:
        matches_data = json.load(f)

    flattened = []
    for m in matches_data:
        row = {
            "match_id": m.get("match_id"),
            "match_date": m.get("match_date"),
            "kick_off": m.get("kick_off"),
            "home_score": m.get("home_score"),
            "away_score": m.get("away_score"),
            "home_team": m.get("home_team", {}).get("home_team_name") if isinstance(m.get("home_team"), dict) else m.get("home_team"),
            "away_team": m.get("away_team", {}).get("away_team_name") if isinstance(m.get("away_team"), dict) else m.get("away_team"),
            "competition_id": competition_id,
            "season_id": season_id
        }
        flattened.append(row)

    matches = pd.DataFrame(flattened)
    matches = _sanitize_df_for_parquet(matches, f"matches_{competition_id}_{season_id}")
    matches.to_parquet(cache_path, index=False)
    log_telemetry(f"Generato file Parquet gare in: {cache_path}")
    return matches

def get_team_matches(matches_df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    return matches_df[(matches_df["home_team"] == team_name) | (matches_df["away_team"] == team_name)]

def get_available_teams(matches_df: pd.DataFrame) -> List[str]:
    teams = set(matches_df["home_team"].dropna().unique()).union(set(matches_df["away_team"].dropna().unique()))
    return sorted(list(teams))

# =====================================================================
# 3. CARICAMENTO EVENTI
# =====================================================================

def _clean_and_flatten_raw_events(raw_events_data: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []
    
    for ev in raw_events_data:
        row = {
            "id": ev.get("id"),
            "index": ev.get("index"),
            "period": ev.get("period"),
            "timestamp": ev.get("timestamp"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "type": ev.get("type", {}).get("name") if isinstance(ev.get("type"), dict) else ev.get("type"),
            "possession": ev.get("possession"),
            "possession_team": ev.get("possession_team", {}).get("name") if isinstance(ev.get("possession_team"), dict) else ev.get("possession_team"),
            "play_pattern": ev.get("play_pattern", {}).get("name") if isinstance(ev.get("play_pattern"), dict) else ev.get("play_pattern"),
            "team": ev.get("team", {}).get("name") if isinstance(ev.get("team"), dict) else ev.get("team"),
            "player": ev.get("player", {}).get("name") if isinstance(ev.get("player"), dict) else ev.get("player"),
            "position": ev.get("position", {}).get("name") if isinstance(ev.get("position"), dict) else ev.get("position"),
            "duration": ev.get("duration"),
            "under_pressure": ev.get("under_pressure")
        }

        loc = ev.get("location")
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            row["x"] = float(loc[0])
            row["y"] = float(loc[1])
            row["location"] = loc
        else:
            row["x"] = None
            row["y"] = None
            row["location"] = None

        p_info = ev.get("pass")
        if isinstance(p_info, dict):
            end_loc = p_info.get("end_location")
            if isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2:
                row["pass_end_x"] = float(end_loc[0])
                row["pass_end_y"] = float(end_loc[1])
                row["pass_end_location"] = end_loc
            else:
                row["pass_end_x"] = None
                row["pass_end_y"] = None
                row["pass_end_location"] = None

            row["pass_length"] = p_info.get("length")
            row["pass_angle"] = p_info.get("angle")
            row["pass_recipient"] = p_info.get("recipient", {}).get("name") if isinstance(p_info.get("recipient"), dict) else p_info.get("recipient")
            row["pass_outcome"] = p_info.get("outcome", {}).get("name") if isinstance(p_info.get("outcome"), dict) else p_info.get("outcome")
            row["pass_type"] = p_info.get("type", {}).get("name") if isinstance(p_info.get("type"), dict) else p_info.get("type")
            row["pass_technique"] = p_info.get("technique", {}).get("name") if isinstance(p_info.get("technique"), dict) else p_info.get("technique")
            row["pass_cross"] = p_info.get("cross")
            row["pass_shot_assist"] = p_info.get("shot_assist")
            row["pass_goal_assist"] = p_info.get("goal_assist")

        s_info = ev.get("shot")
        if isinstance(s_info, dict):
            row["shot_statsbomb_xg"] = s_info.get("statsbomb_xg")
            row["shot_outcome"] = s_info.get("outcome", {}).get("name") if isinstance(s_info.get("outcome"), dict) else s_info.get("outcome")

        d_info = ev.get("dribble")
        if isinstance(d_info, dict):
            row["dribble_outcome"] = d_info.get("outcome", {}).get("name") if isinstance(d_info.get("outcome"), dict) else d_info.get("outcome")

        tactics_info = ev.get("tactics")
        if isinstance(tactics_info, dict):
            row["tactics"] = tactics_info

        records.append(row)

    events_df = pd.DataFrame(records)
    return _sanitize_df_for_parquet(events_df, "events")

def fetch_single_match_events(match_id: int) -> pd.DataFrame:
    ensure_cache_directory()
    cache_path = os.path.join(CACHE_DIR, f"events_{match_id}.parquet")
    
    # 1. Lettura da cache Parquet (Directory: data/)
    if os.path.exists(cache_path):
        log_telemetry(f"Match {match_id}: caricamento da CACHE PARQUET ({cache_path})")
        return pd.read_parquet(cache_path)
    
    # 2. Lettura da archivio grezzo (Directory: statsbomb/data/events/{match_id}.json)
    event_file = os.path.join(STATSBOMB_DIR, "events", f"{match_id}.json")
    if not os.path.exists(event_file):
        raise FileNotFoundError(f"File eventi non trovato per match {match_id}: {event_file} né in {cache_path}")

    log_telemetry(f"Match {match_id}: estrazione da RAW JSON ({event_file})")
    with open(event_file, "r", encoding="utf-8") as f:
        raw_events_data = json.load(f)

    processed_events = _clean_and_flatten_raw_events(raw_events_data)
    processed_events["match_id"] = match_id
    processed_events.to_parquet(cache_path, index=False)
    log_telemetry(f"Match {match_id}: salvato in cache parquet ({cache_path})")
    return processed_events

def fetch_multi_match_events(match_ids: List[int], progress_callback=None) -> pd.DataFrame:
    all_events = []
    total = len(match_ids)
    
    for i, mid in enumerate(match_ids):
        try:
            ev = fetch_single_match_events(mid)
            all_events.append(ev)
        except Exception as e:
            log_telemetry(f"Attenzione: impossibile caricare il match {mid}: {e}")
            
        if progress_callback:
            progress_callback((i + 1) / total, f"Caricato match {mid} ({i+1}/{total})")
            
    if not all_events:
        return pd.DataFrame()
        
    return pd.concat(all_events, ignore_index=True)