import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import pandas as pd
from typing import Dict, Any, Optional

# Palette Opta Dark per PDF
PDF_BG = "#121212"
PDF_CARD = "#1a1a1a"
PDF_BORDER = "#2a2a2a"
PDF_GREEN = "#00ff85"
PDF_ORANGE = "#ff5722"
PDF_BLUE = "#00b0ff"
PDF_TEXT = "#f8f9fa"
PDF_MUTED = "#8e8e93"

def _draw_page_header(fig, title: str, subtitle: str, badge_text: str = "CONFIDENTIAL - MATCH STAFF"):
    """Disegna l'header scuro comune a tutte le pagine del dossier."""
    fig.patch.set_facecolor(PDF_BG)

    # Linea decorativa superiore fluo
    fig.patches.append(patches.Rectangle((0, 0.985), 1.0, 0.015, transform=fig.transFigure,
                                         facecolor=PDF_GREEN, edgecolor='none', zorder=10))

    # Badge in alto a destra
    fig.text(0.94, 0.94, badge_text, color=PDF_GREEN, fontsize=7.5, weight="bold",
             ha="right", va="center",
             bbox=dict(boxstyle="square,pad=0.4", facecolor=PDF_CARD, edgecolor=PDF_BORDER, linewidth=0.8))

    # Titoli
    fig.text(0.06, 0.945, title.upper(), color=PDF_TEXT, fontsize=15, weight="bold", va="center")
    fig.text(0.06, 0.915, subtitle, color=PDF_MUTED, fontsize=8.5, va="center")

    # Separatore orizzontale
    fig.patches.append(patches.Rectangle((0.06, 0.895), 0.88, 0.0015, transform=fig.transFigure,
                                         facecolor=PDF_BORDER, edgecolor='none'))

def _draw_page_footer(fig, current_page: int, total_pages: int = 3):
    """Footer istituzionale con numerazione pagine."""
    fig.patches.append(patches.Rectangle((0.06, 0.06), 0.88, 0.001, transform=fig.transFigure,
                                         facecolor=PDF_BORDER, edgecolor='none'))
    fig.text(0.06, 0.04, "OPTA ANALYST PRO ENGINE | FIFA STANDARD PITCH 105m x 68m",
             color=PDF_MUTED, fontsize=7, va="center")
    fig.text(0.94, 0.04, f"PAGINA {current_page} DI {total_pages}",
             color=PDF_MUTED, fontsize=7, weight="bold", ha="right", va="center")


def build_pdf_match_dossier(
    our_team: str,
    opp_team: str,
    fig_clash: plt.Figure,
    fig_voro_our: plt.Figure,
    fig_to: plt.Figure,
    fig_sc: plt.Figure,
    p_our: Dict[str, Any],
    p_opp: Dict[str, Any],
    roster_df: Optional[pd.DataFrame] = None,
    ai_advice: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Genera il Match Dossier PDF in formato A4 Landscape (11.69 x 8.27 pollici)
    perfettamente impaginato per lo staff tecnico.
    """
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        # =============================================================
        # PAGINA 1: EXECUTIVE COVER, KPI & ROSTER POLIVALENZA AVVERSARIO
        # =============================================================
        fig1 = plt.figure(figsize=(11.69, 8.27), facecolor=PDF_BG)
        _draw_page_header(
            fig1,
            title=f"MATCH DOSSIER: {our_team} vs {opp_team}",
            subtitle="Contrapposizione Territoriale Diretta — Modello Pitch Control a 22 Giocatori"
        )

        # Box KPI Comparativi Rapidi
        kpis = [
            (f"COSTRUZIONE {our_team.upper()}", f"{p_our.get('buildup_accuracy', 0)}%", "Precisione Uscita", PDF_GREEN),
            (f"PRESSING {opp_team.upper()}", f"{p_opp.get('ppda', 0)}", "Indice PPDA", PDF_ORANGE),
            ("BARICENTRO AVVERSARIO", f"{p_opp.get('defensive_height_m', 0)}m", "Altezza Media", PDF_TEXT),
            ("REGISTA DA SCHERMARE", f"{p_opp.get('top_passer', 'N/D')}", "Metronomo", PDF_ORANGE),
        ]

        card_w, card_h = 0.205, 0.08
        start_x, start_y = 0.06, 0.795
        gap = 0.02

        for i, (title, val, sub, val_c) in enumerate(kpis):
            bx = start_x + i * (card_w + gap)
            fig1.patches.append(patches.FancyBboxPatch(
                (bx, start_y), card_w, card_h, transform=fig1.transFigure,
                boxstyle="square,pad=0", facecolor=PDF_CARD, edgecolor=PDF_BORDER, linewidth=1.0
            ))
            fig1.text(bx + 0.012, start_y + 0.060, title, color=PDF_MUTED, fontsize=6.5, weight="bold")
            display_val = (str(val)[:16] + "..") if len(str(val)) > 18 else str(val)
            fig1.text(bx + 0.012, start_y + 0.030, display_val, color=val_c, fontsize=11, weight="bold")
            fig1.text(bx + 0.012, start_y + 0.012, sub, color=PDF_MUTED, fontsize=6.2)

        # Tabella Roster & Polivalenza Avversario (Integrata nella Cover)
        fig1.patches.append(patches.FancyBboxPatch(
            (0.06, 0.09), 0.88, 0.67, transform=fig1.transFigure,
            boxstyle="square,pad=0", facecolor=PDF_CARD, edgecolor=PDF_BORDER, linewidth=1.0
        ))
        fig1.text(0.08, 0.725, f"ORGANICO & POLIVALENZA TATTICA RILEVATA: {opp_team.upper()}", color=PDF_ORANGE, fontsize=8.5, weight="bold")
        fig1.text(0.08, 0.705, "Distribuzione ruoli primari, ruoli secondari riscontrati a referto e baricentro medio in metri FIFA", color=PDF_MUTED, fontsize=6.8)

        # Header Tabella
        t_header_y = 0.665
        fig1.patches.append(patches.Rectangle((0.08, t_header_y), 0.84, 0.030, transform=fig1.transFigure, facecolor="#18222c", edgecolor="none"))
        fig1.text(0.095, t_header_y + 0.009, "RUOLO", color=PDF_MUTED, fontsize=6.5, weight="bold")
        fig1.text(0.180, t_header_y + 0.009, "CALCIATORE", color=PDF_MUTED, fontsize=6.5, weight="bold")
        fig1.text(0.440, t_header_y + 0.009, "RUOLI SECONDARI / POLIVALENZA", color=PDF_MUTED, fontsize=6.5, weight="bold")
        fig1.text(0.720, t_header_y + 0.009, "EVENTI TOTALI", color=PDF_MUTED, fontsize=6.5, weight="bold")
        fig1.text(0.830, t_header_y + 0.009, "BARICENTRO X", color=PDF_MUTED, fontsize=6.5, weight="bold")

        if roster_df is not None and not roster_df.empty:
            for idx, r in enumerate(roster_df.head(14).iterrows()):
                row_data = r[1]
                ry = t_header_y - 0.038 - (idx * 0.036)
                
                # Riga alternata soft
                if idx % 2 == 1:
                    fig1.patches.append(patches.Rectangle((0.08, ry - 0.006), 0.84, 0.034, transform=fig1.transFigure, facecolor="#14181d", edgecolor="none"))

                role_code = str(row_data.get("primary_role", "MID"))
                badge_col = PDF_BLUE if any(k in role_code for k in ["B", "CB", "WB"]) else (PDF_GREEN if any(k in role_code for k in ["M", "DM", "CM", "AM"]) else PDF_ORANGE)
                if role_code == "GK":
                    badge_col = PDF_MUTED

                fig1.text(0.095, ry + 0.008, role_code, color=badge_col, fontsize=7.0, weight="bold")
                fig1.text(0.180, ry + 0.008, str(row_data.get("player", "—"))[:28], color=PDF_TEXT, fontsize=7.0, weight="bold")
                fig1.text(0.440, ry + 0.008, str(row_data.get("secondary_roles", "—"))[:36], color=PDF_MUTED, fontsize=6.6)
                fig1.text(0.740, ry + 0.008, str(row_data.get("volume", 0)), color=PDF_TEXT, fontsize=7.0, weight="bold")
                fig1.text(0.845, ry + 0.008, f"{row_data.get('height_x', 50.0)} m", color=PDF_GREEN, fontsize=7.0, weight="bold")
        else:
            fig1.text(0.095, 0.600, "Nessun dato di organico dettagliato disponibile per la generazione del prospetto.", color=PDF_MUTED, fontsize=7.5)

        _draw_page_footer(fig1, 1, 3)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Salvataggio delle figure del campo (Contrapposizione Territoriale)
        pdf.savefig(fig_clash, bbox_inches="tight", facecolor=PDF_BG)

        # =============================================================
        # PAGINA 2: DOVE COLPIRLI (VULNERABILITÀ DELL'AVVERSARIO)
        # =============================================================
        fig2 = plt.figure(figsize=(11.69, 8.27), facecolor=PDF_BG)
        _draw_page_header(
            fig2,
            title=f"VULNERABILITÀ DIFENSIVE STRUTTURALI: {opp_team.upper()}",
            subtitle="Analisi Spaziale su Campione di 5 Gare — Palle Perse in Uscita e Zone di Concessione Tiri"
        )

        fig2.patches.append(patches.FancyBboxPatch(
            (0.06, 0.78), 0.42, 0.09, transform=fig2.transFigure,
            boxstyle="square,pad=0", facecolor=PDF_CARD, edgecolor=PDF_ORANGE, linewidth=1.0
        ))
        fig2.text(0.075, 0.845, "TARGET ZONE RICONQUISTA (HIGH TURNOVERS)", color=PDF_ORANGE, fontsize=7.5, weight="bold")
        fig2.text(0.075, 0.800,
                  "Mappa delle palle perse dall'avversario entro i propri 52.5 metri.\n"
                  "Attivare il pressing ultra-offensivo indirizzando le corse nelle aree calde.",
                  color=PDF_TEXT, fontsize=7.2, linespacing=1.3)

        fig2.patches.append(patches.FancyBboxPatch(
            (0.52, 0.78), 0.42, 0.09, transform=fig2.transFigure,
            boxstyle="square,pad=0", facecolor=PDF_CARD, edgecolor=PDF_GREEN, linewidth=1.0
        ))
        fig2.text(0.535, 0.845, "SETTORE DI CONCESSIONE TIRI & xG", color=PDF_GREEN, fontsize=7.5, weight="bold")
        fig2.text(0.535, 0.800,
                  "Mappa delle conclusioni subite dall'avversario nelle ultime 5 gare.\n"
                  "Cerchi proporzionali a xG; individuare il corridoio debole per gli inserimenti.",
                  color=PDF_TEXT, fontsize=7.2, linespacing=1.3)

        _draw_page_footer(fig2, 2, 3)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Salvataggio figure Vulnerabilità
        pdf.savefig(fig_to, bbox_inches="tight", facecolor=PDF_BG)
        pdf.savefig(fig_sc, bbox_inches="tight", facecolor=PDF_BG)

        # =============================================================
        # PAGINA 3: TRAINING SESSION BLUEPRINT & SET PIECE INTEL
        # =============================================================
        fig3 = plt.figure(figsize=(11.69, 8.27), facecolor=PDF_BG)
        _draw_page_header(
            fig3,
            title="TRAINING SESSION BLUEPRINT & SET PIECE INTEL",
            subtitle=f"Prescrizioni Operative Settimanali per il Microciclo Pre-Gara contro {opp_team}"
        )

        # --- SEZIONE CORNER (Top Pagina 3) ---
        fig3.patches.append(patches.FancyBboxPatch(
            (0.06, 0.63), 0.88, 0.24, transform=fig3.transFigure,
            boxstyle="square,pad=0", facecolor=PDF_CARD, edgecolor=PDF_BORDER, linewidth=1.0
        ))
        fig3.text(0.08, 0.835, "RIPARTIZIONE GITTATA CORNER AVVERSARI (ANALISI STATISTICA)", color=PDF_GREEN, fontsize=8, weight="bold")

        corner_boxes = [
            ("CORTI / RIGIOCATI (<18m)", f"{p_opp.get('corner_gittata', {}).get('Corti (<18m)', 0)}%", "Uscita a 2 uomini", PDF_TEXT),
            ("IN AREA DI RIGORE (18-35m)", f"{p_opp.get('corner_gittata', {}).get('In Area (18-35m)', 0)}%", f"Traiettoria: {max(p_opp.get('corner_trajectories', {'Diretta': 1}), key=p_opp.get('corner_trajectories', {'Diretta': 1}).get)}", PDF_GREEN),
            ("OLTRE L'AREA (>35m)", f"{p_opp.get('corner_gittata', {}).get('Oltre Area (>35m)', 0)}%", "Presidio secondo palo", PDF_ORANGE),
            ("PERICOLOSITÀ xG SUBITA", f"{p_opp.get('sp_xg_against_p90', 0)}", "xG concessi/90m da corner", PDF_TEXT)
        ]
        c_w = 0.195
        for i, (title, val, sub, col) in enumerate(corner_boxes):
            c_x = 0.08 + i * (c_w + 0.02)
            fig3.patches.append(patches.FancyBboxPatch(
                (c_x, 0.655), c_w, 0.155, transform=fig3.transFigure,
                boxstyle="square,pad=0", facecolor=PDF_BG, edgecolor=PDF_BORDER, linewidth=0.8
            ))
            fig3.text(c_x + 0.01, 0.775, title, color=PDF_MUTED, fontsize=6.2, weight="bold")
            fig3.text(c_x + 0.01, 0.710, val, color=col, fontsize=13, weight="bold")
            fig3.text(c_x + 0.01, 0.675, sub, color=PDF_MUTED, fontsize=6.0)

        # --- SEZIONE BLUEPRINT MICROCICLO (AI COPILOT) ---
        fig3.patches.append(patches.FancyBboxPatch(
            (0.06, 0.09), 0.88, 0.51, transform=fig3.transFigure,
            boxstyle="square,pad=0", facecolor=PDF_CARD, edgecolor=PDF_BORDER, linewidth=1.0
        ))
        
        # Recupero direttive (da parametro o fallback su generator)
        if ai_advice is None or not isinstance(ai_advice, dict) or "guidelines" not in ai_advice:
            from services.tactical_agent import generate_tactical_training_guidelines
            ai_advice = generate_tactical_training_guidelines(p_our, p_opp, {})

        theme = ai_advice.get("core_theme", "Preparazione Tattica")
        fig3.text(0.08, 0.565, f"PROGRAMMAZIONE DEL MICROCICLO (AI COPILOT): {theme.upper()}", color=PDF_TEXT, fontsize=8.5, weight="bold")

        guidelines_list = ai_advice.get("guidelines", [])[:4]
        p_h = 0.095

        for i, g in enumerate(guidelines_list):
            py = 0.450 - i * (p_h + 0.015)
            color_bar = PDF_GREEN if i % 2 == 0 else PDF_BLUE

            fig3.patches.append(patches.FancyBboxPatch(
                (0.08, py), 0.84, p_h, transform=fig3.transFigure,
                boxstyle="square,pad=0", facecolor=PDF_BG, edgecolor=PDF_BORDER, linewidth=0.8
            ))
            fig3.patches.append(patches.Rectangle(
                (0.08, py), 0.006, p_h, transform=fig3.transFigure,
                facecolor=color_bar, edgecolor='none'
            ))

            giorn_fase = f"{g.get('giorno', 'MD').upper()} — {g.get('fase', 'TATTICA').upper()}"
            fig3.text(0.095, py + 0.075, giorn_fase, color=color_bar, fontsize=6.8, weight="bold")

            principio = g.get("principio_guida", "")
            principio_clean = (principio[:135] + "..") if len(principio) > 135 else principio
            fig3.text(0.095, py + 0.052, f"Principio: {principio_clean}", color=PDF_TEXT, fontsize=6.6, weight="bold")

            spazi = g.get("sviluppo_spaziale", "")
            spazi_clean = (spazi[:130] + "..") if len(spazi) > 130 else spazi
            fig3.text(0.095, py + 0.030, f"Spazi & Innesco: {spazi_clean}", color=PDF_MUTED, fontsize=6.2)

            compiti = g.get("obiettivo_reparto", "")
            compiti_clean = (compiti[:130] + "..") if len(compiti) > 130 else compiti
            fig3.text(0.095, py + 0.010, f"Compiti Reparti: {compiti_clean}", color=PDF_TEXT, fontsize=6.2)

        _draw_page_footer(fig3, 3, 3)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    buffer.seek(0)
    return buffer.getvalue()