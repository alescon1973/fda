from typing import Dict, Any, List

def generate_tactical_training_guidelines(
    p_our: Dict[str, Any], 
    p_opp: Dict[str, Any], 
    clash_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Agente di Consulenza Metodologica Dinamica:
    Sintetizza i dati di possesso, non-possesso, territorialità e calci piazzati
    per generare un piano operativo per il microciclo pre-gara (G-4, G-3, G-2, G-1).
    """
    opp_ppda = float(p_opp.get("ppda", 12.0))
    opp_height = float(p_opp.get("defensive_height_m", 45.0))
    opp_top_passer = str(p_opp.get("top_passer", "Regista Avversario"))
    opp_top_prog = str(p_opp.get("top_progressor", "Verticalizzatore"))
    
    our_speed = float(p_our.get("direct_speed_ms", 1.55))
    our_buildup_acc = float(p_our.get("buildup_accuracy", 80.0))
    our_tilt = float(clash_metrics.get("pct_our", 50.0))

    corner_gittata = p_opp.get("corner_gittata", {})
    box_corner_pct = float(corner_gittata.get("In Area (18-35m)", 60.0))
    short_corner_pct = float(corner_gittata.get("Corti (<18m)", 15.0))
    opp_corner_traj = max(p_opp.get("corner_trajectories", {"Diretta": 1}), key=p_opp.get("corner_trajectories", {}).get)

    # -------------------------------------------------------------
    # 1. DIAGNOSI STRATEGICA SUL MATCH
    # -------------------------------------------------------------
    if opp_ppda < 10.0 and opp_height >= 46.0:
        scenario = f"Pressione ultra-offensiva e linea difensiva alta ({opp_height}m). PPDA aggressivo ({opp_ppda})."
        core_theme = "Manipolazione della prima linea con appoggio sul terzo uomo e ricerca immediata dello spazio profondo alle spalle della difesa."
        archetype = "HIGH_PRESS"
    elif opp_ppda > 13.5 and opp_height < 41.0:
        scenario = f"Blocco basso a protezione del limite ({opp_height}m) con PPDA attendista ({opp_ppda}). Densità centrale asfissiante."
        core_theme = "Pazienza nel consolidamento del possesso, scaglionamento nei mezzi-spazi e isolamento sul lato debole."
        archetype = "LOW_BLOCK"
    elif our_speed > 1.75:
        scenario = "Gara ad alta intensità di ribaltamento: avversario con transizioni disordinate e campo lungo concesso."
        core_theme = "Attacco diretto in verticale a ritmo sostenuto con penetrazione nei canali centrali entro 3 trasmissioni."
        archetype = "TRANSITION_WAR"
    else:
        scenario = f"Equilibrio posizionale con baricentro medio ({opp_height}m). Pressing avversario a innesco zonale."
        core_theme = "Circolazione rapida da terzino a mezzala per attrarre i mediani e imbucare sui rimorchi centrali."
        archetype = "BALANCED"

    # -------------------------------------------------------------
    # 2. STRUTTURAZIONE DELLE SESSIONI (G-4, G-3, G-2, G-1)
    # -------------------------------------------------------------
    guidelines: List[Dict[str, str]] = []

    # --- G-4: COSTRUZIONE & DOMINANZA SPAZIALE ---
    if archetype == "HIGH_PRESS":
        g4_principio = "Superamento della prima linea di pressione tramite rotazione continua del vertice basso e scarico diagonale."
        g4_spazio = "Campo a dimensioni regolamentari suddiviso in 3 macro-settori longitudinali, con superiorità numerica 7v6 nella propria metà campo."
        g4_innesco = f"Avversario che porta 3 o 4 uomini oltre la nostra trequarti difensiva (PPDA {opp_ppda})."
        g4_reparto = "Difensori centrali aperti a piede invertito; terzini pronti ad attaccare il corridoio interno se il mediano viene schermato."
    elif archetype == "LOW_BLOCK":
        g4_principio = "Sovraccarico del lato forte per disordinare la linea a cinque avversaria e cambio gioco radente sul lato cieco."
        g4_spazio = "Tre quarti di campo offensivo delimitato con corridoi laterali dedicati al 2v1 continuo tra terzino e ala."
        g4_innesco = f"Avversario schiacciato negli ultimi 35 metri (baricentro medio {opp_height}m)."
        g4_reparto = "Centrocampisti a scaglione sfalsato per impedire la ripartenza preventiva; difensori centrali avanzati stabilmente sul cerchio di metà campo."
    else:
        g4_principio = "Risalita del pallone scambiando la posizione tra mezzala e trequartista nei mezzi-spazi offensivi."
        g4_spazio = "Settori intermedi (zona di rifinitura) con porte piccole sui lati per stimolare l'invasione rapida."
        g4_innesco = "Pressione avversaria indirizzata lateralmente."
        g4_reparto = "Mediani costanti nel dare linee di passaggio d'uscita pulite a un tocco per eludere il raddoppio."

    guidelines.append({
        "giorno": "G-4 | Carico e Dimensione Spaziale",
        "fase": "Fase Offensiva & Uscita Bassa",
        "principio_guida": g4_principio,
        "sviluppo_spaziale": g4_spazio,
        "innesco_situazionale": g4_innesco,
        "obiettivo_reparto": g4_reparto
    })

    # --- G-3: INTENSITÀ & NON-POSSESSO ---
    if opp_ppda < 10.5:
        g3_principio = f"Gegenpressing e schermatura immediata del regista avversario ({opp_top_passer}) impedendo la prima verticalizzazione."
        g3_spazio = "Spazi ridotti (30x30m o 40x40m) ad alta intensità neuromuscolare con parità numerica e sponde esterne rapide."
        g3_innesco = f"Ricezione spalle alla porta del loro verticalizzatore ({opp_top_prog}) entro i primi 35 metri difensivi."
        g3_reparto = "Attaccanti dediti all'oscuramento della linea di passaggio centrale; linea difensiva che accorcia in avanti di 10 metri a palla coperta."
    else:
        g3_principio = f"Compattezza del blocco medio e raddoppio sistematico sulle ricezioni laterali, limitando il raggio d'azione di {opp_top_passer}."
        g3_spazio = "Metà campo difensiva con vincolo di densità centrale massima (mai più di 15 metri di distanza tra le due linee di 4)."
        g3_innesco = "Passaggio orizzontale lento tra i loro difensori centrali."
        g3_reparto = "Centrocampisti coordinati nello scivolamento di reparto; terzino sul lato forte aggressivo sull'esterno senza farsi superare."

    guidelines.append({
        "giorno": "G-3 | Densità e Pressione",
        "fase": "Fase Difensiva & Recupero Palla",
        "principio_guida": g3_principio,
        "sviluppo_spaziale": g3_spazio,
        "innesco_situazionale": g3_innesco,
        "obiettivo_reparto": g3_reparto
    })

    # --- G-2: RIFINITURA & VELOCITÀ DI SCELTA ---
    if our_tilt > 55.0 or archetype == "LOW_BLOCK":
        g2_principio = "Attacco coordinato dell'area di rigore con inserimenti da dietro a rimorchio, evitando traversoni da fermo dalla trequarti."
        g2_spazio = "Ultimi 30 metri di campo con sagome difensive a simulare l'intasamento centrale e porte difese dal portiere titolare."
        g2_innesco = "Palla condotta sul fondo all'altezza del vertice dell'area di rigore."
        g2_reparto = "Attaccanti che tagliano sul primo e secondo palo per liberare lo spazio a rimorchio per le mezze ali a fari spenti."
    else:
        g2_principio = "Transizione offensiva rapida in campo aperto sfruttando la disorganizzazione dei loro centrali in ripiegamento."
        g2_spazio = "Situazioni di superiorità numerica (3v2 e 4v3) partendo dal cerchio di centrocampo a tutto campo."
        g2_innesco = "Riconquista della palla oltre la linea mediana."
        g2_reparto = "Esterni a trazione diretta che allargano il campo; centravanti che attacca la profondità centrale con taglio diagonale."

    guidelines.append({
        "giorno": "G-2 | Rifinitura e Scelta",
        "fase": "Attacco Ultimi 35m & Transizioni",
        "principio_guida": g2_principio,
        "sviluppo_spaziale": g2_spazio,
        "innesco_situazionale": g2_innesco,
        "obiettivo_reparto": g2_reparto
    })

    # --- G-1: REATTIVITÀ & PALLE INATTIVE ---
    if box_corner_pct > 55.0:
        g1_principio = (
            f"Presidio ferreo dell'area piccola contro traiettorie aeree ({box_corner_pct}% in area con traiettoria {opp_corner_traj}) "
            "e contrattacco a campo aperto a palla liberata."
        )
        g1_spazio = "Fermo campo sull'area di rigore con 11 contro 11 e 2 riferimenti offensivi posizionati preventivamente sul cerchio di centrocampo."
        g1_innesco = "Fischio dell'arbitro e battuta diretta verso il primo palo o dischetto del rigore."
        g1_reparto = "Castello difensivo a 3 uomini in area piccola; marcatori a uomo aggressivi sui saltatori; portiere dominante sulle uscite alte."
    elif short_corner_pct > 25.0:
        g1_principio = f"Uscita coordinata a due uomini per neutralizzare lo schema corto ({short_corner_pct}%) negando il cross ravvicinato."
        g1_spazio = "Settore bandierina e linea di fondo dell'area difensiva."
        g1_innesco = "Secondo uomo avversario che si avvicina alla bandierina prima della battuta."
        g1_reparto = "Primo uomo a 9.15m che esce in pressione sul ricevitore; secondo uomo che copre la traiettoria di scarico al limite dell'area."
    else:
        g1_principio = f"Copertura della seconda palla e della mezzaluna fuori area contro respinte corte da corner ({opp_corner_traj})."
        g1_spazio = "Area di rigore e zona 14 (limite dei 16 metri)."
        g1_innesco = "Palla respinta di testa dalla prima linea del castello difensivo."
        g1_reparto = "Mediani obbligati a salire sulla respinta per impedire conclusioni di controbalzo e attivare la trappola del fuorigioco."

    guidelines.append({
        "giorno": "G-1 | Reattività e Palle Inattive",
        "fase": "Palle Inattive",
        "principio_guida": g1_principio,
        "sviluppo_spaziale": g1_spazio,
        "innesco_situazionale": g1_innesco,
        "obiettivo_reparto": g1_reparto
    })

    return {
        "scenario": scenario,
        "core_theme": core_theme,
        "guidelines": guidelines
    }