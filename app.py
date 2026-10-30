import streamlit as st
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from services.data_loader import (
    get_competitions, 
    get_matches_for_competition, 
    get_available_teams,
    get_team_matches,
    fetch_multi_match_events
)
from services.team_tactics import (
    render_team_tactics_view, 
    get_dominant_formation_and_best_lineup,
    compute_lineup_passing_network,
    plot_team_voronoi_spacing
)

from services.style_profiler import (
    render_style_profiler_view, 
    compute_team_style_metrics, 
    compute_tactical_scores
)

from services.pdf_exporter import build_pdf_match_dossier

from services.opponent_analysis import (
    extract_team_profile,
    plot_tactical_zones_clash,
    plot_dual_team_spatial_clash,
    plot_mplsoccer_opponent_turnovers,
    plot_mplsoccer_opponent_shots_conceded,
    get_squad_tactical_roster,
    render_opponent_analysis_view
)

from services.tactical_agent import generate_tactical_training_guidelines

from services.data_loader import render_data_source_telemetry

st.set_page_config(
    page_title="Tactical Intelligence Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design System Opta Dark
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f8f9fa;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #121212 !important;
        border-right: 1px solid #2a2a2a;
    }
    h1, h2, h3, h4 {
        color: #f8f9fa !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    .dashboard-header {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 18px 24px;
        margin-bottom: 20px;
    }
    .dashboard-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #f8f9fa;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .dashboard-subtitle {
        font-size: 0.82rem;
        color: #8e8e93;
        margin-top: 4px;
    }
    
    /* Stile Bottoni Standard per Command Deck */
    div.stButton > button {
        background-color: #1a1a1a;
        color: #f8f9fa;
        border: 1px solid #383838;
        border-radius: 3px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        border-color: #00ff85;
        color: #00ff85;
    }
    div.stButton > button:focus {
        border-color: #00ff85;
        box-shadow: 0 0 8px rgba(0, 255, 133, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Inizializzazione Session State
if "events_data" not in st.session_state:
    st.session_state.events_data = None
if "our_team" not in st.session_state:
    st.session_state.our_team = None
if "opponent_team" not in st.session_state:
    st.session_state.opponent_team = None
if "active_module" not in st.session_state:
    st.session_state.active_module = 1

# =========================================================
# SIDEBAR: PANNELLO DI CONTROLLO
# =========================================================
st.sidebar.markdown("""
    <div style="padding-bottom: 15px; border-bottom: 1px solid #2a2a2a; margin-bottom: 20px;">
        <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; color: #00ff85; text-transform: uppercase;">MATCH ANALYSIS ENGINE</div>
        <div style="font-size: 1.1rem; font-weight: 800; color: #f8f9fa;">TACTICAL HUB</div>
    </div>
""", unsafe_allow_html=True)

try:
    comps = get_competitions()
    comps["label"] = comps["competition_name"] + " — " + comps["season_name"]
    
    render_data_source_telemetry(sidebar=True)
    st.sidebar.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #8e8e93; margin-bottom: 4px; text-transform: uppercase;'>1. Competizione & Torneo</p>", unsafe_allow_html=True)
    comp_selected = st.sidebar.selectbox("Competizione", comps["label"].unique(), label_visibility="collapsed")
    
    row_comp = comps[comps["label"] == comp_selected].iloc[0]
    comp_id = int(row_comp["competition_id"])
    season_id = int(row_comp["season_id"])
    
    matches = get_matches_for_competition(comp_id, season_id)
    teams_list = get_available_teams(matches)
    
    st.sidebar.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #8e8e93; margin-top: 15px; margin-bottom: 4px; text-transform: uppercase;'>2. La Nostra Squadra (Target)</p>", unsafe_allow_html=True)
    target_team = st.sidebar.selectbox("Nostra Squadra", teams_list, label_visibility="collapsed", key="select_our_team")
    
    opponents_available = [t for t in teams_list if t != target_team]
    st.sidebar.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #8e8e93; margin-top: 15px; margin-bottom: 4px; text-transform: uppercase;'>3. Squadra Avversaria</p>", unsafe_allow_html=True)
    opp_team = st.sidebar.selectbox("Avversaria", opponents_available, label_visibility="collapsed", key="select_opp_team")
    
    our_matches = get_team_matches(matches, target_team)
    opp_matches = get_team_matches(matches, opp_team)

    st.sidebar.markdown(f"""
        <div style="background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px; padding: 10px; margin: 15px 0; font-size: 0.75rem; color: #8e8e93;">
            Archivio Gare: <b>{len(our_matches)}</b> ({target_team}) | <b>{len(opp_matches)}</b> ({opp_team})
        </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("ACQUISIZIONE DATI (ENTRAMBE LE SQUADRE)", width='stretch'):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        def update_progress(ratio, msg):
            progress_bar.progress(ratio)
            status_text.text(msg)
            
        with st.spinner("Elaborazione dati in corso..."):
            combined_match_ids = list(set(our_matches["match_id"].tolist() + opp_matches["match_id"].tolist()))
            events = fetch_multi_match_events(combined_match_ids, progress_callback=update_progress)
            
            st.session_state.events_data = events
            st.session_state.our_team = target_team
            st.session_state.opponent_team = opp_team
            status_text.empty()
            progress_bar.empty()
            st.sidebar.success(f"{len(events)} eventi indicizzati con successo.")

except Exception as err:
    import traceback
    st.sidebar.error(f"Errore caricamento parametri: {err}")
    with st.sidebar.expander("Dettagli Tecnici Traceback"):
        st.code(traceback.format_exc(), language="python")

# =========================================================
# BODY PRINCIPALE: COMMAND DECK
# =========================================================
if st.session_state.events_data is None:
    st.markdown("""
        <div class="dashboard-header">
            <div class="dashboard-title">Tactical Intelligence & Match Analysis Engine</div>
            <div class="dashboard-subtitle">Selezionare competizione e squadre dal pannello laterale ed avviare l'acquisizione per inizializzare il modello di gioco.</div>
        </div>
    """, unsafe_allow_html=True)
else:
    our_t = st.session_state.our_team
    opp_t = st.session_state.opponent_team
    events = st.session_state.events_data

    # Estrazione micro-metriche sintetiche per le card
    form_info = get_dominant_formation_and_best_lineup(events, our_t)
    dom_formation = form_info.get("dominant_formation", "N/D")
    dom_minutes = form_info.get("total_minutes", 0)

    perf_metrics = compute_team_style_metrics(events, our_t, sample_size=5)
    perf_scores = compute_tactical_scores(perf_metrics)
    overall_rating = perf_scores.get("overall_rating", 0.0)

    # Header Superiore
    st.markdown(f"""
        <div style="background-color: #1a1a1a; border: 1px solid #2a2a2a; border-left: 4px solid #00ff85; border-radius: 4px; padding: 16px 22px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.70rem; font-weight: 700; color: #8e8e93; text-transform: uppercase; letter-spacing: 0.08em;">MATCH COMMAND CENTER</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin: 2px 0;">
                    <span style="color: #00ff85;">{our_t}</span> 
                    <span style="color: #8e8e93; font-size: 0.95rem; font-weight: 500; margin: 0 6px;">VS</span> 
                    <span style="color: #ff5722;">{opp_t}</span>
                </div>
                <div style="font-size: 0.78rem; color: #8e8e93;">Dimensioni campo: 105m &times; 68m | Provider: StatsBomb Open Data Engine</div>
            </div>
            <div style="text-align: right;">
                <div style="display: inline-block; background: #121212; border: 1px solid #2a2a2a; border-radius: 3px; padding: 6px 14px;">
                    <span style="font-size: 0.70rem; font-weight: 700; color: #00ff85; text-transform: uppercase; letter-spacing: 0.06em;">DATASET SYNCHRONIZED</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # COMMAND DECK: GRIGLIA A 4 CARD INTERATTIVE CON MICRO-KPI
    # -------------------------------------------------------------
    c_deck0, c_deck1, c_deck2, c_deck3 = st.columns([1.2, 1, 1, 1])

    # CARD 0: ONE-PAGE EXECUTIVE (DIGITALE + PDF)
    is_active_0 = (st.session_state.active_module == 0)
    border_0 = "#00ff85" if is_active_0 else "#2a2a2a"
    bg_0 = "#1f2937" if is_active_0 else "#161b22"
    btn_label_0 = "● VISTA ATTIVA" if is_active_0 else "ONE-PAGE EXECUTIVE"

    with c_deck0:
        st.markdown(f"""
            <div style="background-color: {bg_0}; border: 1.5px solid {border_0}; border-radius: 4px; padding: 14px 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.70rem; font-weight: 800; color: {'#00ff85' if is_active_0 else '#8e8e93'}; letter-spacing: 0.08em;">HUB-00 | EXECUTIVE</span>
                    <span style="font-size: 0.70rem; color: #00ff85;">DIGITALE + PDF</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 6px 0 2px 0;">Sintesi Staff</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 8px; border-top: 1px solid #2a2a2a; padding-top: 8px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Formato</div>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #00ff85;">Tablet / A4</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Export</div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">Pronto</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(btn_label_0, key="btn_mod_0", width='stretch'):
            st.session_state.active_module = 0
            st.rerun()

    # CARD 1: ASSETTO TATTICO & VORONOI
    is_active_1 = (st.session_state.active_module == 1)
    border_1 = "#00ff85" if is_active_1 else "#2a2a2a"
    bg_1 = "#1f2937" if is_active_1 else "#161b22"
    btn_label_1 = "● MODULO ATTIVO" if is_active_1 else "APRI MODULO 01"

    with c_deck1:
        st.markdown(f"""
            <div style="background-color: {bg_1}; border: 1.5px solid {border_1}; border-radius: 4px; padding: 14px 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.70rem; font-weight: 800; color: {'#00ff85' if is_active_1 else '#8e8e93'}; letter-spacing: 0.08em;">MOD-01 | STRUTTURA</span>
                    <span style="font-size: 0.70rem; color: #8e8e93;">SELF ANALYSIS</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 6px 0 2px 0;">Assetto Tattico</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 8px; border-top: 1px solid #2a2a2a; padding-top: 8px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Modulo Base</div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #00ff85;">{dom_formation}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Utilizzo</div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">{dom_minutes}'</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(btn_label_1, key="btn_mod_1", width='stretch'):
            st.session_state.active_module = 1
            st.rerun()

    # CARD 2: PROFILO DI STILE & SCOREBOARD
    is_active_2 = (st.session_state.active_module == 2)
    border_2 = "#00ff85" if is_active_2 else "#2a2a2a"
    bg_2 = "#1f2937" if is_active_2 else "#161b22"
    btn_label_2 = "● MODULO ATTIVO" if is_active_2 else "APRI MODULO 02"

    with c_deck2:
        st.markdown(f"""
            <div style="background-color: {bg_2}; border: 1.5px solid {border_2}; border-radius: 4px; padding: 14px 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.70rem; font-weight: 800; color: {'#00ff85' if is_active_2 else '#8e8e93'}; letter-spacing: 0.08em;">MOD-02 | DNA SQUADRA</span>
                    <span style="font-size: 0.70rem; color: #8e8e93;">7 PARAMETRI</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 6px 0 2px 0;">Stile di gioco e Score</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 8px; border-top: 1px solid #2a2a2a; padding-top: 8px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Rating Efficienza</div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #00ff85;">{overall_rating}<span style="font-size: 0.75rem; color: #8e8e93;">/100</span></div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Campione</div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">5 Gare</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(btn_label_2, key="btn_mod_2", width='stretch'):
            st.session_state.active_module = 2
            st.rerun()

    # CARD 3: MATCH PLAN VS AVVERSARIO
    is_active_3 = (st.session_state.active_module == 3)
    border_3 = "#ff5722" if is_active_3 else "#2a2a2a"
    bg_3 = "#2a1b18" if is_active_3 else "#161b22"
    btn_label_3 = "● MODULO ATTIVO" if is_active_3 else "APRI MODULO 03"

    with c_deck3:
        st.markdown(f"""
            <div style="background-color: {bg_3}; border: 1.5px solid {border_3}; border-radius: 4px; padding: 14px 16px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.70rem; font-weight: 800; color: {'#ff5722' if is_active_3 else '#8e8e93'}; letter-spacing: 0.08em;">MOD-03 | STRATEGIA GARA</span>
                    <span style="font-size: 0.70rem; color: #ff5722;">TESTA-A-TESTA</span>
                </div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 6px 0 2px 0;">{opp_t}</div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 8px; border-top: 1px solid #2a2a2a; padding-top: 8px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Duelli Tattici</div>
                        <div style="font-size: 1.2rem; font-weight: 800; color: #ff5722;">4 Fasi</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.68rem; color: #8e8e93; text-transform: uppercase;">Piazzati</div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: #f8fafc;">Inclusi</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button(btn_label_3, key="btn_mod_3", width='stretch'):
            st.session_state.active_module = 3
            st.rerun()

    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # RENDERING DEL MODULO SELEZIONATO DAL DECK
    # -------------------------------------------------------------
    if st.session_state.active_module == 0:
        p_our = extract_team_profile(events, our_t, sample_size=5)
        p_opp = extract_team_profile(events, opp_t, sample_size=5)
        
        form_our = get_dominant_formation_and_best_lineup(events, our_t)
        form_opp = get_dominant_formation_and_best_lineup(events, opp_t)
        
        net_our = compute_lineup_passing_network(events, our_t, form_our["best_lineup"])
        net_opp = compute_lineup_passing_network(events, opp_t, form_opp["best_lineup"])
        
        # Generazione figure per la vista e il PDF (con chiusura figure)
        fig_clash, clash_kpi = plot_dual_team_spatial_clash(net_our["nodes"], net_opp["nodes"], our_t, opp_t)
        fig_voro_our, _ = plot_team_voronoi_spacing(net_our["nodes"], our_t, form_our["dominant_formation"])
        fig_to = plot_mplsoccer_opponent_turnovers(p_opp["turnovers_own_half"], opp_t)
        fig_sc = plot_mplsoccer_opponent_shots_conceded(p_opp["shots_conceded"], opp_t)

        # Dati aggiuntivi per il Match Dossier PDF
        roster_opp_df = get_squad_tactical_roster(events, opp_t)
        ai_advice_data = generate_tactical_training_guidelines(p_our, p_opp, {})

        # Barra di Azione Superiore: Download PDF
        col_head_txt, col_dl = st.columns([3, 1])
        with col_head_txt:
            st.markdown(f"""
                <div style="font-size: 1.2rem; font-weight: 800; color: #f8f9fa;">
                    ONE-PAGE EXECUTIVE SUMMARY: <span style="color:#00ff85;">{our_t}</span> vs <span style="color:#ff5722;">{opp_t}</span>
                </div>
                <div style="font-size: 0.8rem; color: #8e8e93;">Consultazione rapida pre-seduta e generazione del report cartaceo per lo staff.</div>
            """, unsafe_allow_html=True)
        
        with col_dl:
            pdf_bytes = build_pdf_match_dossier(
                our_t, opp_t, fig_clash, fig_voro_our, fig_to, fig_sc, 
                p_our, p_opp, roster_df=roster_opp_df, ai_advice=ai_advice_data
            )
            st.download_button(
                label="📥 SCARICA MATCH DOSSIER PDF",
                data=pdf_bytes,
                file_name=f"Match_Dossier_{our_t}_vs_{opp_t}.pdf",
                mime="application/pdf",
                width='stretch'
            )

        st.write("")

        # Visualizzazione Schermata Digitale: Dual Clash + Prescrizioni Immediate
        c_left, c_right = st.columns([1.3, 1])
        with c_left:
            st.markdown("""
                <div style="background-color: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 4px; padding: 16px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #00ff85; text-transform: uppercase; margin-bottom: 8px;">CONTRAPPOSIZIONE TERRITORIALE (MATRICE 15 ZONE)</div>
            """, unsafe_allow_html=True)
            st.pyplot(fig_clash, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            focus_html = textwrap.dedent(f"""
                <div style="background-color: #1a1a1a; border: 1px solid #2a2a2a; border-left: 4px solid #00ff85; border-radius: 4px; padding: 18px; height: 100%;">
                    <div style="font-size: 0.72rem; font-weight: 800; color: #8e8e93; text-transform: uppercase; letter-spacing: 0.08em;">PRIORITÀ PER IL MICROCICLO</div>
                    <div style="font-size: 1.05rem; font-weight: 800; color: #f8fafc; margin: 6px 0 14px 0;">Focus Sedute di Allenamento</div>
                    <div style="margin-bottom: 12px; background: #121212; padding: 12px; border-radius: 3px; border: 1px solid #2a2a2a;">
                        <div style="color: #00ff85; font-size: 0.72rem; font-weight: 800; text-transform: uppercase;">1. COSTRUZIONE VS LORO PRESSING</div>
                        <div style="font-size: 0.82rem; color: #f8f9fa; margin-top: 3px; line-height: 1.4;">Loro PPDA: <b>{p_opp['ppda']}</b>. Ricerca rapida del terzo uomo ed evitare passaggi orizzontali lenti.</div>
                    </div>
                    <div style="margin-bottom: 12px; background: #121212; padding: 12px; border-radius: 3px; border: 1px solid #2a2a2a;">
                        <div style="color: #ff5722; font-size: 0.72rem; font-weight: 800; text-transform: uppercase;">2. SCHERMATURA CATENA CENTRALE</div>
                        <div style="font-size: 0.82rem; color: #f8f9fa; margin-top: 3px; line-height: 1.4;">Metronomo: <b>{p_opp['top_passer']}</b>. Schermare le linee interne e contrastare sul primo controllo.</div>
                    </div>
                    <div style="background: #121212; padding: 12px; border-radius: 3px; border: 1px solid #2a2a2a;">
                        <div style="color: #00b0ff; font-size: 0.72rem; font-weight: 800; text-transform: uppercase;">3. DIFESA CALCI D'ANGOLO</div>
                        <div style="font-size: 0.82rem; color: #f8f9fa; margin-top: 3px; line-height: 1.4;"><b>{p_opp['corner_gittata']['In Area (18-35m)']}%</b> in area diretta. Castello compatto sul primo palo e 2 uomini pronti per la transizione rapida.</div>
                    </div>
                </div>
            """).strip()
            st.markdown(focus_html, unsafe_allow_html=True)

        # Chiusura figure Matplotlib dopo il rendering per liberare memoria
        plt.close(fig_clash)
        plt.close(fig_voro_our)
        plt.close(fig_to)
        plt.close(fig_sc)

    elif st.session_state.active_module == 1:
        render_team_tactics_view(events, our_t)
    elif st.session_state.active_module == 2:
        render_style_profiler_view(events, our_t)
    elif st.session_state.active_module == 3:
        render_opponent_analysis_view(events, our_t, opp_t)