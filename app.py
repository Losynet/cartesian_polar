import streamlit as st
import pandas as pd
import numpy as np
import math
import itertools
from collections import defaultdict
import plotly.graph_objects as go
import base64
from datetime import datetime
from io import BytesIO

# Import per PDF professionale (ReportLab - Standard industriale)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Import per conversione grafico Plotly in immagine
try:
    import plotly.io as pio
    PLOTLY_IO_AVAILABLE = True
except ImportError:
    PLOTLY_IO_AVAILABLE = False

# Import matplotlib per rendering LaTeX nel PDF
try:
    import matplotlib
    matplotlib.use('Agg') # Backend non interattivo per server
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GeoSolver v69 - Prof. Losenno")

# --- CSS ---
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
    section[data-testid="stSidebar"] { width: 280px !important; min-width: 280px !important; }
    .prof-title { color: #2c3e50; font-family: 'Helvetica', sans-serif; font-size: 1.5rem; border-bottom: 2px solid #2980b9; margin-bottom: 0px; }
    .subtitle { color: #7f8c8d; font-size: 0.9rem; margin-bottom: 10px; }
    .stRadio label { font-size: 16px !important; }
    .stInfo { background-color: #e8f4f8; border-left: 5px solid #2980b9; }
    .step-box { background-color: #fff8e1; border-left: 5px solid #f1c40f; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
    
    /* CSS per PDF nativo browser - Compatibilità universale */
    @media print {
        [data-testid="stSidebar"], [data-testid="stToolbar"], header, footer, 
        .stDeployButton, .stButton, [data-testid="stNumberInput"], 
        [data-testid="stSelectbox"], [data-testid="stMultiSelect"],
        [data-testid="stRadio"], hr, button { display: none !important; }
        @page { margin: 1.5cm; size: A4 portrait; }
        body { font-family: 'Helvetica', 'Arial', sans-serif; font-size: 11pt; line-height: 1.4; }
        .block-container { max-width: 100% !important; padding: 0 !important; }
        * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
        .step-box, .stInfo, h1, h2, h3 { page-break-inside: avoid; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'raw_data' not in st.session_state: st.session_state.raw_data = [] 
if 'points' not in st.session_state: st.session_state.points = {} 
if 'log' not in st.session_state: st.session_state.log = []
if 'projections_visible' not in st.session_state: st.session_state.projections_visible = False 
if 'solved_items' not in st.session_state: st.session_state.solved_items = set() 
if 'solved_values' not in st.session_state: st.session_state.solved_values = {} 
if 'partial_angle_metadata' not in st.session_state: st.session_state.partial_angle_metadata = {}
if 'current_options' not in st.session_state: st.session_state.current_options = None
if 'current_mission' not in st.session_state: st.session_state.current_mission = None
if 'last_calc_msg' not in st.session_state: st.session_state.last_calc_msg = None
if 'input_interpretation' not in st.session_state: st.session_state.input_interpretation = 'GON'
if 'ang_method_choice' not in st.session_state: st.session_state.ang_method_choice = None
if 'selected_partial_suffix' not in st.session_state: st.session_state.selected_partial_suffix = None
if 'az_segments_confirmed' not in st.session_state: st.session_state.az_segments_confirmed = False
if 'angle_workflow_target' not in st.session_state: st.session_state.angle_workflow_target = None
if 'specific_method_choice' not in st.session_state: st.session_state.specific_method_choice = None
if 'student_name' not in st.session_state: st.session_state.student_name = ""
if 'student_surname' not in st.session_state: st.session_state.student_surname = ""
if 'student_class' not in st.session_state: st.session_state.student_class = ""

if 'az_workflow' not in st.session_state:
    st.session_state.az_workflow = {
        'active': False,
        'vertex': None,
        'step': 0,
        'side1': None,
        'side2': None,
        'az1_val': None,
        'az2_val': None,
        'pending_target': None,
        'quiz_options': None,
        'partial_suffix': None,
        'valid_neighbors': None
    }

def get_strategies_for_mission(mission_code, method_filter=None):
    parts = mission_code.split("_"); act = parts[0] + "_" + parts[1]
    q = {}
    if act == "calc_X": 
        pt = parts[2]
        q['correct'] = rf"$X_{{{pt}}} = \overline{{O{pt}}} \cdot \sin(O{pt})$"
        q['wrongs'] = [
            rf"$X_{{{pt}}} = \overline{{O{pt}}} \cdot \cos(O{pt})$",  # Confusione sin/cos
            rf"$X_{{{pt}}} = \overline{{O{pt}}} \cdot \tan(O{pt})$"   # Formula simile ma sbagliata
        ]
        q['latex'] = rf"X_{{{pt}}} = \overline{{O{pt}}} \cdot \sin(O{pt})"
        q['desc'] = rf"Calcolo X di {pt}"
    elif act == "calc_Y": 
        pt = parts[2]
        q['correct'] = rf"$Y_{{{pt}}} = \overline{{O{pt}}} \cdot \cos(O{pt})$"
        q['wrongs'] = [
            rf"$Y_{{{pt}}} = \overline{{O{pt}}} \cdot \sin(O{pt})$",  # Confusione sin/cos
            rf"$Y_{{{pt}}} = \overline{{O{pt}}} / \cos(O{pt})$"       # Formula inversa sbagliata
        ]
        q['latex'] = rf"Y_{{{pt}}} = \overline{{O{pt}}} \cdot \cos(O{pt})"
        q['desc'] = rf"Calcolo Y di {pt}"
    elif act == "calc_dist":
        pt = parts[2]
        q['correct'] = rf"$\overline{{O{pt}}} = \sqrt{{X_{{{pt}}}^2 + Y_{{{pt}}}^2}}$"
        q['wrongs'] = [
            rf"$\overline{{O{pt}}} = X_{{{pt}}} + Y_{{{pt}}}$",                    # Somma invece di Pitagora
            rf"$\overline{{O{pt}}} = \sqrt{{X_{{{pt}}} \cdot Y_{{{pt}}}}}$"        # Prodotto invece di somma quadrati
        ]
        q['latex'] = rf"\overline{{O{pt}}} = \sqrt{{X_{{{pt}}}^2 + Y_{{{pt}}}^2}}"
        q['desc'] = rf"Calcolo Distanza O-{pt}"
    elif act == "calc_az":
        pt = parts[2]
        q['correct'] = rf"$(O{pt}) = \arctan(X_{{{pt}}} / Y_{{{pt}}})$"
        q['wrongs'] = [
            rf"$(O{pt}) = \arctan(Y_{{{pt}}} / X_{{{pt}}})$",          # Rapporto invertito
            rf"$(O{pt}) = \arcsin(X_{{{pt}}} / \overline{{O{pt}}})$"    # Funzione diversa (arcsin)
        ]
        q['latex'] = rf"(O{pt}) = \arctan\left(\frac{{X_{{{pt}}}}}{{Y_{{{pt}}}}}\right)"
        q['desc'] = rf"Calcolo Azimut O-{pt}"
    elif act == "seg_dist":
        p1, p2 = parts[2], parts[3]
        methods = {}
        
        # 1. Metodo Indiretto (Cartesiano)
        cart_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items and
                   f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
        if cart_ok:
            methods["Metodo Indiretto (Cartesiano)"] = {
                "correct": rf"Pitagora: $\overline{{{p1}{p2}}} = \sqrt{{\Delta X^2 + \Delta Y^2}}$",
                "wrongs": [r"Pitagora: $\overline{{{p1}{p2}}} = \Delta X + \Delta Y$", r"Pitagora: $\overline{{{p1}{p2}}} = \sqrt{\Delta X \cdot \Delta Y}$"]
            }
            
        # 2. Metodo Diretto (Carnot)
        pol_ok = (f"Dist_{p1}" in st.session_state.solved_items and f"Az_{p1}" in st.session_state.solved_items and
                  f"Dist_{p2}" in st.session_state.solved_items and f"Az_{p2}" in st.session_state.solved_items)
        if pol_ok:
            methods["Metodo Diretto (Carnot)"] = {
                "correct": rf"Carnot: $\overline{{{p1}{p2}}} = \sqrt{{ \overline{{O{p1}}}^2 + \overline{{O{p2}}}^2 - 2 \cdot \overline{{O{p1}}} \cdot \overline{{O{p2}}} \cdot \cos(\Delta \theta) }}$",
                "wrongs": [
                    rf"Carnot: $\overline{{{p1}{p2}}} = \sqrt{{ \overline{{O{p1}}}^2 + \overline{{O{p2}}}^2 + 2 \cdot \overline{{O{p1}}} \cdot \overline{{O{p2}}} \cdot \cos(\Delta \theta) }}$",  # Segno + invece di -
                    rf"Seni: $\frac{{\overline{{{p1}{p2}}}}}{{\sin(\Delta \theta)}} = \frac{{\overline{{O{p1}}}}}{{\sin(\theta_2)}}$"  # Teorema dei Seni (altro teorema)
                ]
            }

        q['available_methods'] = list(methods.keys())
        selected_data = methods.get(method_filter, list(methods.values())[0]) if methods else None
        if selected_data: q.update(selected_data); q['correct_list'] = [selected_data['correct']]
        else: q.update({'correct': "N/A", 'correct_list': [], 'wrongs': []})
        
        q['latex'] = rf"\overline{{{p1}{p2}}} = ..."
        q['desc'] = rf"Calcolo lunghezza {p1}-{p2}"
    elif act == "seg_az":
        p1, p2 = parts[2], parts[3]
        q['correct'] = rf"$({p1}{p2}) = \arctan(\Delta X / \Delta Y)$"
        q['wrongs'] = [
            rf"$({p1}{p2}) = \arctan(\Delta Y / \Delta X)$",              # Rapporto invertito
            rf"$({p1}{p2}) = \arcsin(\Delta X / \overline{{{p1}{p2}}})$"  # Funzione diversa
        ]
        q['latex'] = rf"({p1}{p2}) = \arctan\left(\frac{{X_{{{p2}}}-X_{{{p1}}}}}{{Y_{{{p2}}}-Y_{{{p1}}}}}\right)"
        q['desc'] = rf"Calcolo Azimut {p1}-{p2}"
    elif act == "calc_poly":
        if parts[2] == "area": q={'correct':r"$S = \frac{1}{2} \cdot L_1 \cdot L_2 \cdot \sin(\alpha)$", 'wrongs':[r"Base x Altezza"], 'latex':r"S...", 'desc':"Calcolo Area"}
        elif parts[2] == "perim": q={'correct':r"$\sum L_i$", 'wrongs':[r"Prod"], 'latex':r"2p...", 'desc':"Calcolo Perimetro"}
    elif act == "calc_ang":
        pt = parts[2]
        pts_lbls = sorted(st.session_state.points.keys())
        idx = pts_lbls.index(pt)
        prev = pts_lbls[idx-1]; nex = pts_lbls[(idx+1)%len(pts_lbls)]
        
        greeks_latex = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta"]
        greek_pt = greeks_latex[idx] if idx < len(greeks_latex) else rf"\alpha_{{{pt}}}"
        
        # Raccogli metodi disponibili
        methods = {}
        
        # 1. Differenza Azimut
        has_az_to_prev = f"SegAz_{pt}_{prev}" in st.session_state.solved_items
        has_az_to_nex = f"SegAz_{pt}_{nex}" in st.session_state.solved_items
        if has_az_to_prev and has_az_to_nex:
            methods["Azimut"] = {
                "correct": rf"Azimut: ${greek_pt} = ({pt}{nex}) - ({pt}{prev})$",
                "wrongs": [rf"Azimut: ${greek_pt} = ({pt}{nex}) + ({pt}{prev})$", rf"Azimut: ${greek_pt} = ({pt}{prev}) - ({pt}{nex})$"]
            }

        # --- ANALISI TRIANGOLI (Full & Partial) ---
        others = [x for x in pts_lbls if x != pt]
        triangles_found = []
        
        for pA, pB in itertools.combinations(others, 2):
            # Verifica esistenza lati triangolo
            s1_k = f"SegDist_{pt}_{pA}" if f"SegDist_{pt}_{pA}" in st.session_state.solved_values else f"SegDist_{pA}_{pt}"
            s2_k = f"SegDist_{pt}_{pB}" if f"SegDist_{pt}_{pB}" in st.session_state.solved_values else f"SegDist_{pB}_{pt}"
            s3_k = f"SegDist_{pA}_{pB}" if f"SegDist_{pA}_{pB}" in st.session_state.solved_values else f"SegDist_{pB}_{pA}"
            
            sides_ok = (s1_k in st.session_state.solved_values and s2_k in st.session_state.solved_values and s3_k in st.session_state.solved_values)
            
            if sides_ok:
                is_full = ({pA, pB} == {prev, nex}) and len(pts_lbls) == 3
                
                # Determina indice parziale (1 o 2) basato sull'ordine CORRETTO
                # Convenzione: in senso orario (A→B→C→D), se diagonale AC divide l'angolo in A:
                # - α₁ è nel primo triangolo (ABC, che contiene il vertice successivo B)
                # - α₂ è nel secondo triangolo (ACD, che contiene il vertice precedente D)
                # Quindi: _1 per triangolo con nex, _2 per triangolo con prev
                suffix = ""
                ordered_pA, ordered_pB = pA, pB
                
                if not is_full:
                    # NUOVA LOGICA: usa la stessa identificazione delle etichette "1" e "2"
                    # Trova quale diagonale esiste nel quadrilatero
                    suffix = ""
                    # Mantieni l'ordine originale di pA e pB per ora
                    ordered_pA, ordered_pB = pA, pB
                    
                    # Il triangolo dell'angolo contiene: pt, pA, pB
                    triangle_vertices = {pt, pA, pB}
                    
                    # Cerca la diagonale
                    diagonal_found = False
                    for p1_lbl, p3_lbl in itertools.combinations(pts_lbls, 2):
                        idx1 = pts_lbls.index(p1_lbl)
                        idx3 = pts_lbls.index(p3_lbl)
                        
                        # È una diagonale? (non adiacenti)
                        if abs(idx1 - idx3) % (len(pts_lbls)-1) > 1:
                            d_key1 = f"SegDist_{p1_lbl}_{p3_lbl}"
                            d_key2 = f"SegDist_{p3_lbl}_{p1_lbl}"
                            
                            if d_key1 in st.session_state.solved_items or d_key2 in st.session_state.solved_items:
                                # Trovata la diagonale!
                                other_pts = [p for p in pts_lbls if p not in [p1_lbl, p3_lbl]]
                                p2_lbl, p4_lbl = other_pts[0], other_pts[1]
                                
                                # Triangolo "1": (p1, p2, p3)
                                triangle_1_set = {p1_lbl, p2_lbl, p3_lbl}
                                # Triangolo "2": (p1, p4, p3)
                                triangle_2_set = {p1_lbl, p4_lbl, p3_lbl}
                                
                                # Confronta con il triangolo dell'angolo
                                if triangle_vertices == triangle_1_set:
                                    # Assegna suffisso solo se il vertice dell'angolo (pt) è un estremo della diagonale
                                    if pt == p1_lbl or pt == p3_lbl:
                                        suffix = "_1"
                                    diagonal_found = True
                                    # Imposta ordered_pA e ordered_pB come gli altri due vertici del triangolo
                                    # IMPORTANTE: Ordina alfabeticamente per consistenza con le chiavi salvate
                                    other_verts = sorted([v for v in triangle_vertices if v != pt])
                                    ordered_pA, ordered_pB = other_verts[0], other_verts[1]
                                elif triangle_vertices == triangle_2_set:
                                    # Assegna suffisso solo se il vertice dell'angolo (pt) è un estremo della diagonale
                                    if pt == p1_lbl or pt == p3_lbl:
                                        suffix = "_2"
                                    diagonal_found = True
                                    # Imposta ordered_pA e ordered_pB come gli altri due vertici del triangolo
                                    # IMPORTANTE: Ordina alfabeticamente per consistenza con le chiavi salvate
                                    other_verts = sorted([v for v in triangle_vertices if v != pt])
                                    ordered_pA, ordered_pB = other_verts[0], other_verts[1]
                                
                                if diagonal_found:
                                    break
                    

                
                # Etichetta triangolo nell'ordine corretto: pt, primo vertice nell'ordine, secondo vertice
                tri_label = f"{pt}{ordered_pA}{ordered_pB}"
                greek_sub = greek_pt + suffix
                
                # 1. Carnot
                m_name = f"Carnot {greek_sub} (Tri. {tri_label})"
                # Converti greek_sub in formato display Unicode
                greek_display = latex_to_unicode_greek(greek_sub)
                m_name_display = f"Carnot {greek_display} (Tri. {tri_label})"
                methods[m_name_display] = {
                    "correct": rf"Carnot: ${greek_sub} = \arccos\left(\frac{{\overline{{{pt}{ordered_pA}}}^2 + \overline{{{pt}{ordered_pB}}}^2 - \overline{{{ordered_pA}{ordered_pB}}}^2}}{{2 \cdot \overline{{{pt}{ordered_pA}}} \cdot \overline{{{pt}{ordered_pB}}}}}\right)$",
                    "wrongs": [
                        rf"Carnot: ${greek_sub} = \arccos\left(\frac{{\overline{{{pt}{ordered_pA}}}^2 + \overline{{{pt}{ordered_pB}}}^2 + \overline{{{ordered_pA}{ordered_pB}}}^2}}{{2 \cdot \overline{{{pt}{ordered_pA}}} \cdot \overline{{{pt}{ordered_pB}}}}}\right)$",  # Segno + invece di -
                        rf"Seni: ${greek_sub} = \arcsin\left(\frac{{\overline{{{ordered_pA}{ordered_pB}}} \cdot \sin(\beta)}}{{\overline{{{pt}{ordered_pA}}}}}\right)$"  # Teorema dei Seni (altro teorema)
                    ],
                    "meta": {"type": "carnot_tri", "pA": ordered_pA, "pB": ordered_pB, "sub": suffix}
                }
                
                # 2. Seni (se un altro angolo nel triangolo è noto)
                # Cerchiamo se PartAng o Ang esiste per pA o pB in questo triangolo
                # Il triangolo è formato da: pt, ordered_pA, ordered_pB
                other_ang_val = None
                other_pt = None
                other_opp_side = None
                
                # Cerca tra i vertici del triangolo
                for cand in [ordered_pA, ordered_pB]:
                    # Check Full (solo se il poligono è un triangolo)
                    full_key = f"Ang_{cand}"
                    if full_key in st.session_state.solved_values:
                        if len(pts_lbls) == 3: 
                            other_ang_val = st.session_state.solved_values[full_key]
                            other_pt = cand
                            break
                    
                    # Check Partial nel triangolo corrente
                    v1, v2 = [x for x in [pt, ordered_pA, ordered_pB] if x != cand]
                    k_part = f"PartAng_{cand}_{min(v1, v2)}_{max(v1, v2)}"
                    
                    if k_part in st.session_state.solved_values:
                        other_ang_val = st.session_state.solved_values[k_part]
                        other_pt = cand
                        break
                    else:
                        # Prova anche a cercare con ordine diverso (fallback)
                        k_part_alt = f"PartAng_{cand}_{max(v1, v2)}_{min(v1, v2)}"
                        if k_part_alt in st.session_state.solved_values:
                            other_ang_val = st.session_state.solved_values[k_part_alt]
                            other_pt = cand
                            break
                
                # Se non trovato, cerca QUALSIASI angolo parziale nel triangolo corrente
                if not other_ang_val:
                    triangle_verts = {pt, ordered_pA, ordered_pB}
                    for k, v in st.session_state.solved_values.items():
                        if k.startswith("PartAng_"):
                            parts = k.split("_")
                            if len(parts) >= 4:
                                ang_vertex = parts[1]
                                ang_v1 = parts[2]
                                ang_v2 = parts[3]
                                # Verifica che questo angolo sia nel triangolo corrente
                                if ang_vertex in triangle_verts and ang_v1 in triangle_verts and ang_v2 in triangle_verts:
                                    if ang_vertex != pt:  # Non vogliamo l'angolo che stiamo cercando di calcolare
                                        other_ang_val = v
                                        other_pt = ang_vertex
                                        break
                
                if other_ang_val:
                    # side opposto a pt è ordered_pA-ordered_pB
                    # side opposto a other_pt è tra pt e il terzo vertice
                    rem = ordered_pB if other_pt == ordered_pA else ordered_pA
                    s_opp_other = f"SegDist_{pt}_{rem}" if f"SegDist_{pt}_{rem}" in st.session_state.solved_values else f"SegDist_{rem}_{pt}"
                    
                    # Determina il simbolo greco per other_pt con suffisso corretto
                    idx_other = pts_lbls.index(other_pt) if other_pt in pts_lbls else -1
                    if idx_other >= 0 and idx_other < len(greeks_latex):
                        other_greek_base = greeks_latex[idx_other]
                    else:
                        other_greek_base = r"\alpha"
                    
                    # CORREZIONE: Cerca il suffisso dell'angolo nel TRIANGOLO CORRENTE
                    # Il triangolo è: pt, ordered_pA, ordered_pB
                    # L'angolo in other_pt è tra i lati verso gli altri due vertici
                    triangle_vertices = {pt, ordered_pA, ordered_pB}
                    other_vertices = [v for v in triangle_vertices if v != other_pt]
                    
                    k_other_part = f"PartAng_{other_pt}_{min(other_vertices)}_{max(other_vertices)}"
                    
                    other_suffix = ""
                    if k_other_part in st.session_state.solved_values:
                        # È un angolo parziale - recupera il suffisso dai metadata
                        other_metadata = st.session_state.partial_angle_metadata.get(k_other_part, {})
                        other_suffix = other_metadata.get('suffix', '')
                    
                    other_greek_symbol = other_greek_base + other_suffix
                    
                    m_name_sin = f"Seni {greek_sub} (Tri. {tri_label})"
                    greek_display = latex_to_unicode_greek(greek_sub)
                    m_name_sin_display = f"Seni {greek_display} (Tri. {tri_label})"
                    methods[m_name_sin_display] = {
                        "correct": rf"Seni: $\sin({greek_sub}) = \frac{{\overline{{{ordered_pA}{ordered_pB}}} \cdot \sin({other_greek_symbol})}}{{\overline{{{pt}{rem}}}}}$",
                        "wrongs": [
                            rf"Seni: $\sin({greek_sub}) = \frac{{\overline{{{pt}{rem}}} \cdot \sin({other_greek_symbol})}}{{\overline{{{ordered_pA}{ordered_pB}}}}}$",
                            rf"Seni: $\cos({greek_sub}) = \frac{{\overline{{{ordered_pA}{ordered_pB}}} \cdot \sin({other_greek_symbol})}}{{\overline{{{pt}{rem}}}}}$"
                        ],
                        "meta": {"type": "sines_tri", "pA": ordered_pA, "pB": ordered_pB, "sub": suffix, "other": other_pt}
                    }

                # 3. Sottrazione (180 - altri due)
                # Cerca se sono noti gli altri DUE angoli del triangolo CORRENTE
                triangle_verts_set = {pt, ordered_pA, ordered_pB}
                val_A, val_B = None, None
                k_A, k_B = None, None # Chiavi degli angoli trovati

                # NUOVA LOGICA: Cerca TUTTI gli angoli (parziali o completi) dei vertici del triangolo
                # escludendo quello che stiamo calcolando
                
                angles_found = {}  # {vertex: (key, value)}
                
                # Per ogni vertice del triangolo (escluso pt che stiamo calcolando)
                for vertex in [ordered_pA, ordered_pB]:
                    # 1. Prova a cercare un angolo completo (per triangoli semplici)
                    if len(pts_lbls) == 3 and f"Ang_{vertex}" in st.session_state.solved_values:
                        angles_found[vertex] = (f"Ang_{vertex}", st.session_state.solved_values[f"Ang_{vertex}"])
                    else:
                        # 2. Cerca angoli parziali che coinvolgono questo vertice nel triangolo corrente
                        # Cerca tutte le chiavi PartAng_{vertex}_...
                        for k, v in st.session_state.solved_values.items():
                            if k.startswith(f"PartAng_{vertex}_"):
                                # Estrai i vertici dalla chiave: PartAng_{vertex}_{v1}_{v2}
                                parts = k.split("_")
                                if len(parts) >= 4:
                                    v1, v2 = parts[2], parts[3]
                                    angle_vertices = {vertex, v1, v2}
                                    
                                    # Verifica che questo angolo sia nel triangolo corrente
                                    if angle_vertices == triangle_verts_set:
                                        angles_found[vertex] = (k, v)
                                        break
                
                # Assegna val_A e val_B dai risultati trovati
                if ordered_pA in angles_found:
                    k_A, val_A = angles_found[ordered_pA]
                
                if ordered_pB in angles_found:
                    k_B, val_B = angles_found[ordered_pB]

                if val_A and val_B:
                    m_name_sub = f"Sottrazione {greek_sub} (Tri. {tri_label})"
                    greek_display = latex_to_unicode_greek(greek_sub)
                    m_name_sub_display = f"Sottrazione {greek_display} (Tri. {tri_label})"
                    tot_str = r"180^{\circ}" if st.session_state.input_interpretation == AngleUnit.DEG else r"200^g"
                    
                    # GENERA SIMBOLI GRECI CORRETTI per gli angoli trovati (k_A, k_B)
                    def get_greek_symbol_from_key(key):
                        parts = key.split('_')
                        vertex = parts[1]
                        idx = pts_lbls.index(vertex)
                        base_symbol = greeks_latex[idx] if idx < len(greeks_latex) else r"\alpha"
                        
                        # Aggiungi suffisso solo se la chiave è di un angolo parziale
                        if key.startswith("PartAng_"):
                           # Recupera il suffisso corretto dai metadati
                           suffix = st.session_state.partial_angle_metadata.get(key, {}).get('suffix', '')
                           return base_symbol + suffix
                        return base_symbol # È un angolo completo (Ang_...)

                    greek_A = get_greek_symbol_from_key(k_A)
                    greek_B = get_greek_symbol_from_key(k_B)
                    
                    methods[m_name_sub_display] = {
                        "correct": rf"Sottrazione: ${greek_sub} = {tot_str} - ({greek_A} + {greek_B})$",
                        "wrongs": [
                            rf"Sottrazione: ${greek_sub} = {tot_str} + ({greek_A} + {greek_B})$",
                            rf"Sottrazione: ${greek_sub} = {tot_str} - ({greek_A} - {greek_B})$"
                        ],
                        "meta": {"type": "sub_tri", "pA": ordered_pA, "pB": ordered_pB, "sub": suffix, "k_A": k_A, "k_B": k_B}
                    }

        # 4. Somma Parziali (Se esistono)
        # Cerca se esistono coppie di PartAng adiacenti che formano l'angolo completo
        partials_at_pt = {} # Store solved partials by their vertices: {('B','C'): value, ('C','D'): value}
        for k, v in st.session_state.solved_values.items():
            if k.startswith(f"PartAng_{pt}_"):
                pA, pB = k.split('_')[2], k.split('_')[3]
                partials_at_pt[tuple(sorted((pA, pB)))] = v

        # L'angolo completo è tra 'prev' e 'nex'.
        # Cerchiamo un 'diag_pt' tale per cui esistono i parziali (prev, diag) e (diag, nex)
        all_diagonals = [p for p in pts_lbls if p not in [pt, prev, nex]]
        
        for diag_pt in all_diagonals:
            key1 = tuple(sorted((prev, diag_pt)))
            key2 = tuple(sorted((diag_pt, nex)))
            
            if key1 in partials_at_pt and key2 in partials_at_pt:
                # Trovata la coppia che forma l'angolo completo!
                methods["Somma Parziali"] = {
                    "correct": rf"Somma: ${greek_pt} = {greek_pt}_1 + {greek_pt}_2$",
                    "wrongs": [
                        rf"Somma: ${greek_pt} = {greek_pt}_1 - {greek_pt}_2$",                    # Sottrazione invece di somma
                        rf"Sottrazione: ${greek_pt} = 180° - {greek_pt}_1 - {greek_pt}_2$"  # Confusione con sottrazione angolare
                    ],
                    "meta": {"type": "sum_partials"}
                }
                break # Trovata una coppia valida, basta.

        q['available_methods'] = list(methods.keys())
        
        # Se l'utente ha selezionato un angolo parziale specifico, filtra solo i metodi per quel parziale
        if hasattr(st.session_state, 'selected_partial_suffix') and st.session_state.selected_partial_suffix:
            suffix_to_find = st.session_state.selected_partial_suffix  # "_1" o "_2"
            filtered_methods = {}
            for method_name, method_data in methods.items():
                # Controlla se il metodo è per il parziale selezionato
                meta = method_data.get('meta', {})
                method_suffix = meta.get('sub', '')
                
                # Includi il metodo se:
                # 1. Ha lo stesso suffisso del parziale selezionato
                # 2. È un metodo di somma parziali
                # 3. È un metodo per angolo completo (suffix vuoto) - permette di usare angoli parziali per calcolare angoli completi
                if (method_suffix == suffix_to_find or 
                    meta.get('type') == 'sum_partials' or 
                    method_suffix == ''):
                    # Include questo metodo
                    filtered_methods[method_name] = method_data
            
            # Se abbiamo trovato metodi filtrati, usa quelli. Altrimenti usa tutti.
            if filtered_methods:
                methods = filtered_methods
                q['available_methods'] = list(methods.keys())
        
        # Selezione Metodo
        selected_data = None
        if method_filter and method_filter in methods:
            selected_data = methods[method_filter]
        elif method_filter:
            # Prova una ricerca fuzzy se la chiave esatta non esiste
            # Cerca per nome parziale (es. "Seni" in "Seni α₁ (Tri. ABC)")
            for method_name, method_data in methods.items():
                if method_filter in method_name or method_name in method_filter:
                    selected_data = method_data
                    break
        
        # Fallback se ancora non trovato
        if not selected_data and len(methods) > 0:
            # Prendi il primo disponibile
            selected_data = list(methods.values())[0]
            
        if selected_data:
            q['correct'] = selected_data['correct']
            q['correct_list'] = [selected_data['correct']]
            q['wrongs'] = selected_data['wrongs']
            q['meta'] = selected_data.get('meta')
            
            # DEBUG CRITICO
            
        else:
            q['correct'] = "N/A"; q['correct_list'] = []; q['wrongs'] = []

        q['latex'] = rf"{greek_pt} = ({pt}{nex}) - ({pt}{prev})"
        q['desc'] = f"Calcolo Angolo in {pt}"
    return q

# --- HEADER ---
st.markdown("<div class='prof-title'>📐 GeoSolver Ultimate</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Progetto didattico dei Prof. G. Losenno - E. D'Aranno</div>", unsafe_allow_html=True)

# --- UTILS ---
class AngleUnit:
    GON = "Gon"
    DEG = "Deg"
    RAD = "Rad"

def parse_angle(val, mode):
    try:
        # Convert string mode to AngleUnit if necessary
        if isinstance(mode, str):
            mode_upper = mode.upper()
            if 'GON' in mode_upper or 'CENTESIMAL' in mode_upper:
                mode = AngleUnit.GON
            elif 'DEG' in mode_upper or 'SESSAGESIMAL' in mode_upper:
                mode = AngleUnit.DEG
            elif 'RAD' in mode_upper:
                mode = AngleUnit.RAD
            else:
                mode = AngleUnit.GON  # Default
        
        # If val is a string, try to infer unit from suffix and strip it
        if isinstance(val, str):
            val_clean = val.strip().lower()
            if val_clean.endswith('g'):
                mode = AngleUnit.GON
                val = val_clean.rstrip('g')
            elif val_clean.endswith('°'):
                mode = AngleUnit.DEG
                val = val_clean.rstrip('°')
            elif "rad" in val_clean:
                mode = AngleUnit.RAD
                val = val_clean.replace("rad", "").strip()
        
        val = float(val)
        if mode == AngleUnit.GON: return val * (math.pi / 200.0)
        elif mode == AngleUnit.DEG: return math.radians(val)
        elif mode == AngleUnit.RAD: return val
        else: return val * (math.pi / 200.0)  # Default to GON
    except: return 0.0

def format_angle_output(rad_val, unit_mode, latex=False):
    deg_sym = r"^{\circ}" if latex else "°"
    if unit_mode == AngleUnit.RAD: return f"{rad_val:.4f} rad"
    elif unit_mode == AngleUnit.DEG: return f"{math.degrees(rad_val):.4f}{deg_sym}"
    elif unit_mode == AngleUnit.GON: return f"{(rad_val * 200.0 / math.pi):.4f}g"
    return str(rad_val)

def format_coord(val): return f"{val:.2f}"

def latex_to_unicode_greek(latex_str):
    """Converte simboli LaTeX greci in caratteri Unicode.
    Es: '\\alpha_1' -> 'α₁', '\\beta_2' -> 'β₂'
    """
    # Mappatura simboli greci
    greek_map = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ'
    }
    
    # Mappatura numeri in pedice
    subscript_map = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                     '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    
    result = latex_str
    
    # Sostituisci lettere greche
    for latex, unicode_char in greek_map.items():
        result = result.replace(latex, unicode_char)
    
    # Gestisci i pedici: _1 -> ₁, _2 -> ₂
    import re
    result = re.sub(r'_(\d)', lambda m: subscript_map.get(m.group(1), m.group(1)), result)
    
    return result

def get_shuffled_options(correct, wrongs):
    import random
    opts = [correct] + wrongs
    random.shuffle(opts)
    return opts

# --- REPORT HTML ---
def clean_latex_for_text(text):
    """Rimuove LaTeX e converte in testo leggibile"""
    if not isinstance(text, str):
        return str(text)
    
    import re
    
    # Rimuovi delimitatori LaTeX
    text = text.replace('$$', '').replace('$', '')
    
    # Rimuovi ambienti LaTeX complessi
    text = re.sub(r'\\begin\{[^}]+\}', '', text)
    text = re.sub(r'\\end\{[^}]+\}', '', text)
    text = re.sub(r'\\aligned', '', text)
    
    # Sostituzioni simboli matematici
    replacements = {
        r'\overline{': '', r'\overline': '',
        r'\cdot': '×', r'\times': '×', r'\div': '÷',
        r'\pm': '±', r'\le': '≤', r'\ge': '≥', r'\ne': '≠', r'\approx': '≈',
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ', r'\Delta': 'Δ',
        r'\theta': 'θ', r'\pi': 'π', r'\epsilon': 'ε',
        r'\sqrt': '√', r'\sum': 'Σ', r'\int': '∫',
        r'\quad': ' ', r'\qquad': '  ',
        '\\\\': ' ', '&': '', '{': '', '}': '',
        '^2': '²', '^3': '³', '_': '', '~': ' '
    }
    
    for latex, replacement in replacements.items():
        text = text.replace(latex, replacement)
    
    # Pulisci comandi LaTeX generici rimasti
    text = re.sub(r'\\[a-zA-Z]+\{', '', text)  # \comando{
    text = re.sub(r'\\[a-zA-Z]+', '', text)    # \comando
    
    # Pulisci spazi multipli e newline
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def latex_to_rl_image(latex_str, fontsize=12):
    """Converte stringa LaTeX in immagine ReportLab usando Matplotlib"""
    if not MATPLOTLIB_AVAILABLE or not latex_str:
        return None
    
    try:
        # Pulisci la stringa
        s = latex_str.strip()
        # Rimuovi delimitatori se presenti (matplotlib li vuole o no a seconda del contesto, ma meglio gestire)
        if s.startswith('$$') and s.endswith('$$'): s = s[2:-2]
        elif s.startswith('$') and s.endswith('$'): s = s[1:-1]
        
        # Matplotlib mathtext richiede $...$ per il math mode
        render_str = f"${s}$"
        
        # Configura plot
        buf = BytesIO()
        fig = plt.figure(figsize=(0.1, 0.1)) # Dimensione dummy
        # Renderizza testo
        fig.text(0, 0, render_str, fontsize=fontsize)
        
        # Salva con bounding box stretto
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.02)
        plt.close(fig)
        buf.seek(0)
        
        # Calcola dimensioni per ReportLab
        from PIL import Image as PILImage
        with PILImage.open(buf) as pil_img:
            w_px, h_px = pil_img.size
        
        # Conversione px -> punti (300dpi -> 72dpi)
        scale_factor = 72 / 300 * 0.8 # 0.8 correzione visiva
        buf.seek(0)
        return RLImage(buf, width=w_px*scale_factor, height=h_px*scale_factor)
        
    except Exception as e:
        # In caso di errore (es. sintassi non supportata da mathtext), ritorna None per fallback testuale
        return None

def generate_pdf_with_reportlab(log_entries, solved_values, student_surname, student_name, fig):
    """
    Genera PDF usando ReportLab - Standard industriale
    Funziona su TUTTI i dispositivi (Mac, Windows, Linux, iOS, Android)
    """
    if not REPORTLAB_AVAILABLE:
        print("❌ ReportLab non disponibile")
        return None
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
        
        # Stili
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#2c3e50'), spaceAfter=8, alignment=TA_CENTER)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2980b9'), spaceAfter=6, spaceBefore=10)
        normal_style = styles['Normal']
        
        story = []
        
        # Intestazione
        story.append(Paragraph('<b>GeoSolver - Report Topografia</b>', title_style))
        story.append(Paragraph('Prof. G. Losenno - Prof. E. D\'Aranno', styles['Normal']))
        story.append(Spacer(1, 10*mm))
        
        # Info studente
        now = datetime.now()
        info_data = [
            ['Studente:', f'{student_surname} {student_name}'],
            ['Data:', now.strftime('%d/%m/%Y %H:%M')]
        ]
        info_table = Table(info_data, colWidths=[40*mm, 120*mm])
        info_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2980b9')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#2980b9'))
        ]))
        story.append(info_table)
        story.append(Spacer(1, 8*mm))
        
        # Grafico (con multipli metodi fallback per massima compatibilità)
        tmp_path = None  # Track temp file
        if fig:
            graph_included = False
            
            # METODO 1: Prova con write_image (Richiede 'kaleido' installato)
            try:
                # Questo metodo richiede che la libreria 'kaleido' sia installata (versione 0.2.1 consigliata)
                img_bytes_io = BytesIO()
                
                # Salva figura come immagine usando il metodo write_image se disponibile
                try:
                    # Nota: Se kaleido non è installato, questo comando fallirà
                    fig.write_image(img_bytes_io, format='png', width=1200, height=800)
                    img_bytes_io.seek(0)
                    
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        tmp.write(img_bytes_io.read())
                        tmp_path = tmp.name
                    
                    story.append(Paragraph('<b>Grafico del Poligono</b>', heading_style))
                    img = RLImage(tmp_path, width=160*mm, height=106*mm)
                    story.append(img)
                    story.append(Spacer(1, 6*mm))
                    graph_included = True
                    print("✅ Grafico incluso (metodo write_image)")
                except:
                    raise Exception("write_image fallito (probabilmente kaleido mancante)")
                    
            except Exception as e1:
                # METODO 2: Prova con kaleido (fallback)
                try:
                    if not PLOTLY_IO_AVAILABLE:
                        raise ImportError("plotly.io non disponibile")
                    
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        img_bytes = pio.to_image(fig, format='png', width=1200, height=800, engine='kaleido')
                        tmp.write(img_bytes)
                        tmp_path = tmp.name
                    
                    story.append(Paragraph('<b>Grafico del Poligono</b>', heading_style))
                    img = RLImage(tmp_path, width=160*mm, height=106*mm)
                    story.append(img)
                    story.append(Spacer(1, 6*mm))
                    graph_included = True
                    print("✅ Grafico incluso (metodo kaleido)")
                    
                except Exception as e2:
                    # METODO 3: Prova con orca (vecchio metodo)
                    try:
                        import tempfile, os
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                            img_bytes = pio.to_image(fig, format='png', width=1200, height=800, engine='orca')
                            tmp.write(img_bytes)
                            tmp_path = tmp.name
                        
                        story.append(Paragraph('<b>Grafico del Poligono</b>', heading_style))
                        img = RLImage(tmp_path, width=160*mm, height=106*mm)
                        story.append(img)
                        story.append(Spacer(1, 6*mm))
                        graph_included = True
                        print("✅ Grafico incluso (metodo orca)")
                    except Exception as e3:
                        # Tutti i metodi falliti
                        print(f"⚠️ Impossibile includere grafico:")
                        print(f"  - Metodo 1 (write_image/kaleido): {e1}")
                        print(f"  - Metodo 2 (kaleido): {e2}")
                        print(f"  - Metodo 3 (orca): {e3}")
                        
            if not graph_included:
                story.append(Paragraph('<b>Grafico del Poligono</b>', heading_style))
                story.append(Paragraph('<i>[Grafico non disponibile - Il grafico è visibile nell\'applicazione web]</i>', normal_style))
                story.append(Spacer(1, 3*mm))
        
        # Passaggi
        story.append(Paragraph('<b>Passaggi di Risoluzione</b>', heading_style))
        story.append(Spacer(1, 3*mm))
        
        if not log_entries:
            story.append(Paragraph('<i>Nessun passaggio registrato</i>', normal_style))
        
        for i, entry in enumerate(log_entries, 1):
            is_error = entry.get('is_error', False)
            action = clean_latex_for_text(entry.get('action', 'N/A'))
            
            # Stile per i passaggi con background colorato
            bg_color = colors.HexColor('#ffe6e6') if is_error else colors.HexColor('#e6ffe6')
            # Usa simboli Unicode supportati invece di emoji
            icon = '✗' if is_error else '✓'  # ✓ checkmark, ✗ ballot X
            
            step_style = ParagraphStyle(
                'StepStyle',
                parent=normal_style,
                fontSize=9,
                leading=11,
                leftIndent=8,
                rightIndent=8,
                spaceBefore=3,
                spaceAfter=3
            )
            
            step_title_style = ParagraphStyle(
                'StepTitleStyle',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold',
                leftIndent=8,
                spaceBefore=3
            )
            
            # Tenta di generare immagine LaTeX per il Metodo (Formula)
            method_raw = entry.get('method', 'N/A')
            method_img = latex_to_rl_image(method_raw, fontsize=11)
            if method_img:
                method_content = [Paragraph('<i>Metodo:</i>', step_style), method_img]
            else:
                method_content = [Paragraph(f'<i>Metodo:</i> {clean_latex_for_text(method_raw)}', step_style)]
            
            # Per il Risultato, usa testo pulito (spesso è multiline aligned, difficile per mathtext)
            # Se è semplice, si potrebbe provare a renderizzarlo, ma aligned fallisce in matplotlib standard
            result_raw = entry.get('result', 'N/A')
            # Fallback a testo per il risultato per garantire leggibilità dei passaggi numerici
            result_clean = clean_latex_for_text(result_raw)
            result_content = [Paragraph(f'<i>Risultato:</i> {result_clean}', step_style)]
            
            # NON limitare la lunghezza - usa word wrapping
            
            # Usa Paragraph per permettere word wrapping automatico
            step_content = [
                [Paragraph(f'{icon} <b>Step {i}: {action}</b>', step_title_style)],
                method_content,
                result_content
            ]
            
            step_table = Table(step_content, colWidths=[165*mm])
            step_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            
            story.append(step_table)
            story.append(Spacer(1, 3*mm))
        
        # Statistiche
        total = len(log_entries)
        errors = sum(1 for e in log_entries if e.get('is_error', False))
        success_rate = ((total - errors) / total * 100) if total > 0 else 0
        
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph('<b>Statistiche</b>', heading_style))
        
        # Usa simboli Unicode compatibili con Helvetica invece di emoji
        stats_data = [
            ['Totale passaggi:', str(total)],
            ['Corretti:', f'{total - errors} ●'],  # Pallino nero
            ['Errori:', f'{errors} ●'],            # Pallino nero
            ['Successo:', f'{success_rate:.1f}%']
        ]
        
        stats_table = Table(stats_data, colWidths=[50*mm, 40*mm])
        stats_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            # Colore verde per "Corretti"
            ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#27ae60')),
            # Colore rosso per "Errori"
            ('TEXTCOLOR', (1, 2), (1, 2), colors.HexColor('#e74c3c')),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#2980b9')),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#2980b9'))
        ]))
        
        story.append(stats_table)
        
        # Footer
        story.append(Spacer(1, 8*mm))
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        story.append(Paragraph('<i>Report generato da GeoSolver v70</i>', footer_style))
        
        # Genera PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        # Pulisci file temporaneo se esiste
        if tmp_path:
            try:
                import os
                os.unlink(tmp_path)
                print("🗑️ File temporaneo rimosso")
            except:
                pass
        
        return pdf_bytes
        
    except Exception as e:
        print(f"Errore generazione PDF: {e}")
        import traceback
        traceback.print_exc()
        
        # Pulisci file temporaneo anche in caso di errore
        if 'tmp_path' in locals() and tmp_path:
            try:
                import os
                os.unlink(tmp_path)
            except:
                pass
        
        return None


def generate_html_report(log_entries, solved_values, student_surname, student_name, fig):
    student_fullname = f"{student_surname} {student_name}".strip()
    if not student_fullname:
        student_fullname = "___________________"

    rows_html = ""
    error_count = 0
    
    if not log_entries:
        rows_html = "<div class='entry'><div class='entry-body'><p style='text-align:center; color:#999; font-style:italic;'>Nessuna operazione eseguita. L'esercizio è stato terminato senza svolgimento.</p></div></div>"
    else:
        for i, entry in enumerate(log_entries):
            clean_res = entry['result']
            clean_method = entry['method'].replace(r"$", "")
            desc_text = entry.get('desc_verbose', 'Calcolo eseguito.')
            is_error = entry.get('is_error', False)
            
            if is_error:
                error_count += 1
                # Stile per errori
                rows_html += f"""
                <div class="entry error-entry">
                    <div class="entry-header error-header">❌ Step {i+1}: {entry['action']} - ERRORE</div>
                    <div class="entry-body">
                        <div class="description error-description">
                            <span class="icon">⚠️</span> <b>Errore:</b> {desc_text}
                        </div>
                        <div class="formula-row">
                            <p><b>Formula scelta (ERRATA):</b> ${clean_method}$</p>
                        </div>
                        <div class="math-box error-box">$${clean_res}$$</div>
                    </div>
                </div>
                """
            else:
                # Stile per successi
                rows_html += f"""
                <div class="entry">
                    <div class="entry-header">✅ Step {i+1}: {entry['action']}</div>
                    <div class="entry-body">
                        <div class="description">
                            <span class="icon">💡</span> <b>Analisi:</b> {desc_text}
                        </div>
                        <div class="formula-row">
                            <p><b>Formula:</b> ${clean_method}$</p>
                        </div>
                        <div class="math-box">$${clean_res}$$</div>
                    </div>
                </div>
                """
    
    # Converti grafico Plotly in immagine statica per compatibilità PDF
    if fig:
        try:
            # Metodo 1: Prova con kaleido (migliore qualità)
            import plotly.io as pio
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            plot_html = f'<div style="text-align:center;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto; border:1px solid #ddd; border-radius:8px;"/></div>'
            print("✅ Grafico convertito in PNG con kaleido")
        except Exception as e:
            print(f"⚠️ Kaleido non disponibile: {e}")
            try:
                # Metodo 2: Salva come HTML statico semplificato (senza JavaScript)
                print("ℹ️ Usando metodo alternativo: HTML statico")
                # Estrai solo i dati del grafico, senza interattività
                import json
                # Crea SVG invece di HTML interattivo
                svg_str = pio.to_image(fig, format='svg').decode('utf-8')
                plot_html = f'<div style="text-align:center;">{svg_str}</div>'
                print("✅ Grafico convertito in SVG")
            except Exception as e2:
                print(f"⚠️ Conversione SVG fallita: {e2}")
                print("ℹ️ Nascondo grafico dal PDF (solo formule)")
                # Fallback: nessun grafico
                plot_html = """
                <div style="text-align:center; padding:40px; background:#f8f9fa; border:2px dashed #ddd; border-radius:8px; margin:20px 0;">
                    <p style="color:#666; font-size:14px; margin:0;">
                        📊 <b>Grafico non disponibile nel PDF</b><br>
                        <small>Visualizza il grafico nell'applicazione web oppure installa kaleido:<br>
                        <code>pip3 install kaleido</code></small>
                    </p>
                </div>
                """
    else:
        plot_html = "<p style='text-align:center; color:#999; font-style:italic;'>Grafico non disponibile.</p>"
    
    # Aggiungi statistiche errori
    total_steps = len(log_entries)
    success_count = total_steps - error_count
    success_rate = (success_count / total_steps * 100) if total_steps > 0 else 0
    
    if total_steps > 0:
        stats_html = f"""
        <div class="stats-box">
            <h3>📊 Statistiche Esercizio</h3>
            <p><b>Totale passaggi:</b> {total_steps}</p>
            <p><b>Passaggi corretti:</b> {success_count} ✅</p>
            <p><b>Errori commessi:</b> {error_count} ❌</p>
            <p><b>Tasso di successo:</b> {success_rate:.1f}%</p>
        </div>
        """
    else:
        stats_html = """
        <div class="stats-box" style="background: #fff3cd; border-color: #ffc107;">
            <h3>⚠️ Esercizio Non Completato</h3>
            <p>L'esercizio è stato terminato senza svolgere alcun passaggio.</p>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <script>MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']]}}}};</script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; padding: 40px; color: #333; background: #fdfdfd; }}
            h1 {{ color: #2c3e50; border-bottom: 4px solid #2980b9; padding-bottom: 10px; }}
            h2 {{ color: #2c3e50; border-bottom: 2px solid #2980b9; padding-bottom: 8px; margin-top: 40px;}}
            .plot-container {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            .entry {{ border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 25px; background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05); overflow: hidden; }}
            .entry-header {{ background: #2980b9; color: white; padding: 10px 15px; font-weight: bold; font-size: 1.1em; }}
            .entry-body {{ padding: 20px; }}
            .description {{ background: #fffbe6; border-left: 4px solid #f1c40f; padding: 15px; margin-bottom: 15px; font-style: italic; color: #555; }}
            .formula-row {{ margin-bottom: 10px; color: #7f8c8d; font-weight: bold; }}
            .math-box {{ font-family: 'Times New Roman', serif; font-size: 1.2em; background: #f8f9fa; padding: 15px; border: 1px dashed #ccc; border-radius: 5px; overflow-x: auto; text-align: center; }}
            
            /* Stili per gli errori */
            .error-entry {{ border: 2px solid #e74c3c; background: #ffebee; }}
            .error-header {{ background: #e74c3c !important; }}
            .error-description {{ background: #ffcdd2; border-left: 4px solid #c62828; }}
            .error-box {{ background: #ffebee; border: 2px solid #e74c3c; }}
            
            /* Stats box */
            .stats-box {{ background: #e8f4f8; border: 2px solid #2980b9; border-radius: 8px; padding: 20px; margin: 30px 0; }}
            .stats-box h3 {{ margin-top: 0; color: #2980b9; }}
            .stats-box p {{ margin: 8px 0; font-size: 1.1em; }}
            
            table {{ width: 100%; border-collapse: collapse; margin-top: 30px; border: 1px solid #ddd; }}
            td, th {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #f2f2f2; color: #2c3e50; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>📐 GeoSolver - Report Tecnico</h1>
        <p><b>Studente:</b> {student_fullname} &nbsp;|&nbsp; <b>Docenti:</b> Prof. G. Losenno - Prof. E. D'Aranno</p>
        <p><b>Data:</b> {datetime.now().strftime("%d/%m/%Y alle %H:%M")}</p>
        <hr>
        {stats_html}
        <h2>Grafico Esercizio</h2>
        <div class="plot-container">
            {plot_html}
        </div>
        <hr>
        <h2>Log Operazioni</h2>
        {rows_html}
        <h3>Riepilogo Risultati</h3>
        <table>
            <tr><th>Grandezza</th><th>Valore</th></tr>
            {''.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in solved_values.items()]) if solved_values else '<tr><td colspan="2" style="text-align:center; color:#999; font-style:italic;">Nessun risultato calcolato</td></tr>'}
        </table>
    </body>
    </html>
    """
    return html

def undo_last_action():
    if st.session_state.log:
        last = st.session_state.log.pop()
        for k in last.get('added_items', []): 
            st.session_state.solved_items.discard(k)
            # Rimuovi anche i metadata se è un angolo parziale
            if k.startswith('PartAng_'):
                st.session_state.partial_angle_metadata.pop(k, None)
        for k in last.get('added_values', []): st.session_state.solved_values.pop(k, None)
        st.session_state.last_calc_msg = None
        st.session_state.current_mission = None
        st.session_state.az_workflow['active'] = False
        st.rerun()

def recalculate_points():
    st.session_state.points = {}
    st.session_state.solved_items = set()
    st.session_state.solved_values = {}
    st.session_state.partial_angle_metadata = {}
    st.session_state.az_workflow = {'active': False, 'vertex': None, 'step': 0, 'side1': None, 'side2': None, 'az1_val': None, 'az2_val': None, 'pending_target': None, 'quiz_options': None}
    st.session_state.dist_method_preference = None
    st.session_state.specific_method_choice = None
    unit_in = st.session_state.input_interpretation
    for item in st.session_state.raw_data:
        lbl = item['lbl']
        p_unit = item.get('unit', unit_in) # Usa unità specifica se presente
        if item['type'] == 'pol':
            v1, v2 = item['v1'], item['v2']
            rads = parse_angle(v2, p_unit)
            st.session_state.points[lbl] = {'x': v1*math.sin(rads), 'y': v1*math.cos(rads), 'r': v1, 'alpha': rads, 'type': 'pol'}
            st.session_state.solved_items.add(f"Dist_{lbl}"); st.session_state.solved_items.add(f"Az_{lbl}")
            st.session_state.solved_values[f"Dist_{lbl}"] = format_coord(v1)
            st.session_state.solved_values[f"Az_{lbl}"] = format_angle_output(rads, p_unit)
        elif item['type'] == 'cart':
            v1, v2 = item['v1'], item['v2']
            st.session_state.points[lbl] = {'x': v1, 'y': v2, 'r': math.sqrt(v1**2+v2**2), 'alpha': math.atan2(v1, v2), 'type': 'cart'}
            st.session_state.solved_items.add(f"X_{lbl}"); st.session_state.solved_items.add(f"Y_{lbl}")
            st.session_state.solved_values[f"X_{lbl}"] = format_coord(v1)
            st.session_state.solved_values[f"Y_{lbl}"] = format_coord(v2)

def activate_projections():
    st.session_state.projections_visible = True
    st.session_state.log.append({'action': 'Grafico', 'method': 'Setup', 'result': 'Attivazione assi cartesiani', 'desc_verbose': 'Inizializzazione del sistema di riferimento cartesiano.'})

def reset_all(): st.session_state.clear()

# --- SIDEBAR ---
with st.sidebar:
    st.header("👤 Studente")
    st.session_state.student_surname = st.text_input("Cognome *", st.session_state.student_surname, placeholder="Inserisci il cognome")
    st.session_state.student_name = st.text_input("Nome *", st.session_state.student_name, placeholder="Inserisci il nome")
    
    # Aggiungi campo Classe (opzionale)
    if 'student_class' not in st.session_state:
        st.session_state.student_class = ""
    st.session_state.student_class = st.text_input("Classe", st.session_state.student_class, placeholder="es. 3A, 4B (opzionale)")
    
    # Verifica se cognome e nome sono stati inseriti
    student_info_complete = bool(st.session_state.student_surname.strip() and st.session_state.student_name.strip())
    
    if not student_info_complete:
        st.warning("⚠️ Inserisci Cognome e Nome per iniziare l'esercizio")
        st.stop()  # Ferma l'esecuzione della sidebar qui
    
    # Mostra info studente con classe se presente
    class_info = f" - {st.session_state.student_class}" if st.session_state.student_class.strip() else ""
    st.success(f"✅ Studente: {st.session_state.student_surname} {st.session_state.student_name}{class_info}")
    st.divider()
    
    st.header("1. Dati")
    input_mode = st.selectbox("Esercizio", ["Manuale", "Es. 34", "Es. 35", "Es. 36", "Es. 37", "Es. 38", "Es. 39", "Es. 40"])

    def load_raw(data):
        st.session_state.raw_data = []
        st.session_state.log = []
        st.session_state.projections_visible = False
        st.session_state.last_calc_msg = None
        st.session_state.dist_method_preference = None
        for d in data: st.session_state.raw_data.append(d)
        recalculate_points()

    if st.button("Carica"):
        if "Es. 34" in input_mode: load_raw([{'lbl':'A','type':'pol','v1':56.84,'v2':46.8845}, {'lbl':'B','type':'pol','v1':78.12,'v2':91.2488}])
        elif "Es. 35" in input_mode: load_raw([{'lbl':'S','type':'cart','v1':-18.46,'v2':67.52}, {'lbl':'R','type':'cart','v1':-85.62,'v2':-10.45}])
        elif "Es. 36" in input_mode: load_raw([{'lbl':'A','type':'pol','v1':106.15,'v2':43.1846}, {'lbl':'B','type':'pol','v1':137.95,'v2':112.2084}, {'lbl':'C','type':'pol','v1':198.26,'v2':78.1121}])
        elif "Es. 37" in input_mode: load_raw([{'lbl':'A','type':'cart','v1':-27.58,'v2':15.81},{'lbl':'B','type':'cart','v1':55.07,'v2':-27.86},{'lbl':'C','type':'cart','v1':25.87,'v2':61.75}])
        elif "Es. 38" in input_mode: load_raw([{'lbl':'A','type':'pol','v1':77.18,'v2':318.8560}, {'lbl':'B','type':'pol','v1':96.84,'v2':241.2450}, {'lbl':'C','type':'pol','v1':89.06,'v2':162.4672}, {'lbl':'D','type':'pol','v1':104.66,'v2':48.0050}])
        elif "Es. 39" in input_mode:
            xA, yA = -134.77, 88.06
            radB = 157.1380 * (math.pi/200.0); xB = xA + 601.24 * math.sin(radB); yB = yA + 601.24 * math.cos(radB)
            radC = 78.0657 * (math.pi/200.0); xC = xA + 660.34 * math.sin(radC); yC = yA + 660.34 * math.cos(radC)
            load_raw([{'lbl':'A','type':'cart','v1':xA,'v2':yA},{'lbl':'B','type':'cart','v1':xB,'v2':yB},{'lbl':'C','type':'cart','v1':xC,'v2':yC}])
        elif "Es. 40" in input_mode:
            ipo = 15.30; alpha_rad = 70.1867 * (math.pi/200.0); beta_rad = 29.8133 * (math.pi/200.0)
            a = ipo * math.sin(alpha_rad); b = ipo * math.sin(beta_rad)
            load_raw([{'lbl':'C','type':'cart','v1':0.0,'v2':0.0},{'lbl':'A','type':'cart','v1':0.0,'v2':a},{'lbl':'B','type':'cart','v1':b,'v2':0.0}])
        elif "Manuale" in input_mode: st.session_state.raw_data = []; st.rerun()

    if st.session_state.raw_data:
        st.divider()
        st.markdown("### 📋 Dati Noti")
        for item in st.session_state.raw_data:
            lbl = item['lbl']
            if item['type'] == 'pol':
                st.info(f"**Punto {lbl}** (Polare)\n\nDist: `{item['v1']}`\nAzimut: `{item['v2']}`")
            elif item['type'] == 'cart':
                st.info(f"**Punto {lbl}** (Cartesiano)\n\nX: `{item['v1']}`\nY: `{item['v2']}`")
        
        st.divider()
        st.caption("Unità:")
        new_interpretation = st.radio("", [AngleUnit.GON, AngleUnit.DEG], key="unit_selector")
        if new_interpretation != st.session_state.input_interpretation:
            st.session_state.input_interpretation = new_interpretation
            recalculate_points()
            st.rerun()

    if input_mode == "Manuale":
        st.divider()
        col1, col2 = st.columns(2)
        # Alfabeto completo per le etichette
        alfabeto = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        with col1: pt_lbl = st.selectbox("Etichetta", alfabeto, index=0)
        with col2: c_type = st.selectbox("Tipo", ["Polare", "Cartesiano"])
        
        v1 = st.number_input("Distanza (m)" if "Polare" in c_type else "Coord. X (m)", format="%.4f")
        v2 = st.number_input("Azimut" if "Polare" in c_type else "Coord. Y (m)", format="%.4f")
        
        man_unit = st.selectbox("Unità", [AngleUnit.GON, AngleUnit.DEG, AngleUnit.RAD]) if "Polare" in c_type else None

        if st.button("Aggiungi Punto"):
            dtype = 'pol' if "Polare" in c_type else 'cart'
            st.session_state.raw_data.append({'lbl':pt_lbl, 'type':dtype, 'v1':v1, 'v2':v2, 'unit': man_unit})
            recalculate_points()
            st.rerun()
    
    st.markdown("---")
    st.button("Reset", on_click=reset_all)

# --- HELPER GRAFICI ---
def get_arc_path(center_x, center_y, radius, start_angle_rad, end_angle_rad, points=50):
    thetas = np.linspace(start_angle_rad, end_angle_rad, points)
    xs = center_x + radius * np.sin(thetas)
    ys = center_y + radius * np.cos(thetas)
    return xs, ys

# [SEZIONE 2: NUOVA FUNZIONE ARCO AZIMUTALE TOPOGRAFICO]
def get_azimuth_arc_points_topo(center, radius, start_az, end_az, num_points=30):
    # Genera punti per un arco che parte da Nord (Y) e ruota in senso ORARIO
    # Gestione passaggio per lo zero (es. 350 -> 10)
    if end_az < start_az: 
        thetas = np.linspace(start_az, 360, num_points // 2).tolist() + np.linspace(0, end_az, num_points // 2).tolist()
    else:
        thetas = np.linspace(start_az, end_az, num_points)
    
    # Conversione topografica: X = Cx + R*sin(theta), Y = Cy + R*cos(theta)
    xs = [center[0] + radius * math.sin(math.radians(t)) for t in thetas]
    ys = [center[1] + radius * math.cos(math.radians(t)) for t in thetas]
    return xs, ys

def calculate_label_levels(coords):
    sorted_coords = sorted(coords, key=lambda x: x[0])
    levels = {}
    if not sorted_coords: return levels
    level_last_pos = defaultdict(lambda: -float('inf'))
    for val, label in sorted_coords:
        assigned_lvl = 0
        while True:
            if abs(val - level_last_pos[assigned_lvl]) > (max(sorted_coords[-1][0] - sorted_coords[0][0], 10) * 0.08):
                levels[label] = assigned_lvl
                level_last_pos[assigned_lvl] = val
                break
            assigned_lvl += 1
    return levels

def get_text_rotation(p1, p2):
    dx = p2['x'] - p1['x']; dy = p2['y'] - p1['y']
    deg = math.degrees(math.atan2(dy, dx))
    if deg > 90: deg -= 180
    elif deg < -90: deg += 180
    return -deg

def get_point_label_pos(x, y):
    if x >= 0 and y >= 0: return "top right"
    elif x < 0 and y >= 0: return "top left"
    elif x < 0 and y < 0: return "bottom left"
    else: return "bottom right"

def draw_angle_wedge(fig, pt_lbl, color, radius, text=None, p_from_lbl=None, p_to_lbl=None, triangle_suffix=None):
    """
    Disegna un arco di angolo.
    Se triangle_suffix è specificato (_1 o _2), usa la stessa logica delle etichette
    dei triangoli per determinare quale arco disegnare.
    """
    pts_lbls = sorted(st.session_state.points.keys())
    try:
        curr = st.session_state.points[pt_lbl]
        
        if p_from_lbl and p_to_lbl:
            prev = st.session_state.points[p_from_lbl]
            nex = st.session_state.points[p_to_lbl]
        else:
            idx = pts_lbls.index(pt_lbl)
            prev = st.session_state.points[pts_lbls[idx-1]]
            nex = st.session_state.points[pts_lbls[(idx+1)%len(pts_lbls)]]
        
        az_in = math.atan2(prev['x']-curr['x'], prev['y']-curr['y'])
        az_out = math.atan2(nex['x']-curr['x'], nex['y']-curr['y'])
        
        # Se è un angolo parziale con suffisso specificato, usa la logica delle etichette
        if triangle_suffix and len(pts_lbls) == 4 and p_from_lbl and p_to_lbl:
            # Cerca quale diagonale esiste
            for p1_lbl, p3_lbl in itertools.combinations(pts_lbls, 2):
                idx1 = pts_lbls.index(p1_lbl)
                idx3 = pts_lbls.index(p3_lbl)
                
                # È una diagonale?
                if abs(idx1 - idx3) % (len(pts_lbls)-1) > 1:
                    d_key1 = f"SegDist_{p1_lbl}_{p3_lbl}"
                    d_key2 = f"SegDist_{p3_lbl}_{p1_lbl}"
                    
                    if d_key1 in st.session_state.solved_items or d_key2 in st.session_state.solved_items:
                        # Trovata la diagonale! Ora determina i due triangoli
                        other_pts = [p for p in pts_lbls if p not in [p1_lbl, p3_lbl]]
                        p2_lbl, p4_lbl = other_pts[0], other_pts[1]
                        
                        # Triangolo "1": (p1, p2, p3) dove p2 = other_pts[0]
                        # Triangolo "2": (p1, p4, p3) dove p4 = other_pts[1]
                        
                        # Determina in quale triangolo si trova l'angolo parziale
                        # usando i vertici ESATTI dell'angolo salvati nei metadata
                        
                        # I tre vertici del triangolo dell'angolo sono: pt_lbl, p_from_lbl, p_to_lbl
                        # Confronta questo set con i due triangoli
                        
                        angle_triangle_set = {pt_lbl, p_from_lbl, p_to_lbl}
                        triangle_1_set = {p1_lbl, p2_lbl, p3_lbl}
                        triangle_2_set = {p1_lbl, p4_lbl, p3_lbl}
                        
                        target_centroid = None
                        
                        if angle_triangle_set == triangle_1_set:
                            # L'angolo è nel triangolo "1"
                            target_centroid = (
                                (st.session_state.points[p1_lbl]['x'] + st.session_state.points[p2_lbl]['x'] + st.session_state.points[p3_lbl]['x']) / 3,
                                (st.session_state.points[p1_lbl]['y'] + st.session_state.points[p2_lbl]['y'] + st.session_state.points[p3_lbl]['y']) / 3
                            )
                        elif angle_triangle_set == triangle_2_set:
                            # L'angolo è nel triangolo "2"
                            target_centroid = (
                                (st.session_state.points[p1_lbl]['x'] + st.session_state.points[p4_lbl]['x'] + st.session_state.points[p3_lbl]['x']) / 3,
                                (st.session_state.points[p1_lbl]['y'] + st.session_state.points[p4_lbl]['y'] + st.session_state.points[p3_lbl]['y']) / 3
                            )
                        
                        if target_centroid:
                            # Calcola l'azimut verso il baricentro target
                            az_centroid = math.atan2(target_centroid[0] - curr['x'], target_centroid[1] - curr['y'])
                            
                            # Normalizza gli angoli
                            az_in = (az_in + 2*math.pi) % (2*math.pi)
                            az_out = (az_out + 2*math.pi) % (2*math.pi)
                            az_centroid = (az_centroid + 2*math.pi) % (2*math.pi)
                            
                            # Determina quale arco contiene il baricentro
                            def angle_between(start, end, target):
                                diff = (end - start) % (2*math.pi)
                                target_diff = (target - start) % (2*math.pi)
                                return target_diff <= diff
                            
                            if angle_between(az_in, az_out, az_centroid):
                                start, end = az_in, az_out
                            else:
                                start, end = az_out, az_in
                            
                            if end < start:
                                end += 2*math.pi
                        else:
                            # Fallback: usa arco minore
                            diff = (az_out - az_in) % (2*math.pi)
                            if diff <= math.pi:
                                start, end = az_in, az_out
                            else:
                                start, end = az_out, az_in
                            if end < start:
                                end += 2*math.pi
                        
                        break  # Trovata la diagonale, esci dal loop
            else:
                # Nessuna diagonale trovata - fallback
                diff = (az_out - az_in) % (2*math.pi)
                if diff <= math.pi:
                    start, end = az_in, az_out
                else:
                    start, end = az_out, az_in
                if end < start:
                    end += 2*math.pi
        else:
            # Angolo completo - usa la logica originale
            diff = (az_out - az_in) % (2*math.pi)
            if diff > math.pi:
                start, end = az_out, az_in
                if end < start: end += 2*math.pi
            else:
                start, end = az_in, az_out
                if end < start: end += 2*math.pi

        wedge_x, wedge_y = get_arc_path(curr['x'], curr['y'], radius*0.5, start, end, 30)
        wedge_x = np.append(np.insert(wedge_x, 0, curr['x']), curr['x'])
        wedge_y = np.append(np.insert(wedge_y, 0, curr['y']), curr['y'])
        
        hover_txt = f"Angolo in {pt_lbl}"
        if f"Ang_{pt_lbl}" in st.session_state.solved_values:
            val = st.session_state.solved_values[f"Ang_{pt_lbl}"]
            hover_txt += f"<br>Valore: {val}"
        
        fig.add_trace(go.Scatter(x=wedge_x, y=wedge_y, fill="toself", fillcolor=color, line=dict(width=0), showlegend=False, hoverinfo='text', text=hover_txt))
        
        if text:
            mid_angle = (start + end) / 2
            # Coordinate topografiche: X=sin, Y=cos
            txt_x = curr['x'] + (radius * 0.6) * math.sin(mid_angle)
            txt_y = curr['y'] + (radius * 0.6) * math.cos(mid_angle)
            fig.add_annotation(x=txt_x, y=txt_y, text=text, showarrow=False, font=dict(color="black", size=14, weight="bold"), bgcolor="rgba(255,255,255,0.4)")
            
        return curr, az_in, az_out
    except: return None, 0, 0

# --- NUOVO: AZIMUT CON OFFSET INTELLIGENTE ---
def draw_azimuth_visuals(fig, p1_lbl, p2_lbl, base_radius, radius_offset_step):
    p1 = st.session_state.points[p1_lbl]
    p2 = st.session_state.points[p2_lbl]
    
    # Raggio effettivo (Base + Offset se ci sono più azimut dallo stesso punto)
    eff_radius = base_radius + radius_offset_step
    
    # 1. Freccia Nord
    fig.add_trace(go.Scatter(x=[p1['x'], p1['x']], y=[p1['y'], p1['y'] + eff_radius], mode='lines', line=dict(color='gray', width=1, dash='dash'), showlegend=False))
    
    # 2. Arco Azimut
    dx = p2['x'] - p1['x']; dy = p2['y'] - p1['y']
    az_rad = (math.atan2(dx, dy) + 2*math.pi) % (2*math.pi)
    
    hover_txt = f"Azimut ({p1_lbl}-{p2_lbl})"
    if f"SegAz_{p1_lbl}_{p2_lbl}" in st.session_state.solved_values:
        val = st.session_state.solved_values[f"SegAz_{p1_lbl}_{p2_lbl}"]
        hover_txt += f"<br>Valore: {val}"
    
    arc_x, arc_y = get_arc_path(p1['x'], p1['y'], eff_radius*0.7, 0, az_rad, 40)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color='#d35400', width=2), showlegend=False, hoverinfo='text', text=hover_txt))
    
    # Freccia
    if len(arc_x) > 0:
        last_x, last_y = arc_x[-1], arc_y[-1]
        fig.add_trace(go.Scatter(x=[last_x], y=[last_y], mode='markers', marker=dict(symbol='arrow-bar-up', size=8, color='#d35400', angle=180), showlegend=False))
    
    # Etichetta
    if len(arc_x) > 0:
        mid_idx = len(arc_x)//2
        fig.add_annotation(x=arc_x[mid_idx], y=arc_y[mid_idx], text=f"({p1_lbl}{p2_lbl})", font=dict(color="#d35400", size=10), showarrow=False, xshift=5, yshift=5, bgcolor="rgba(255,255,255,0.6)")

# --- GRAPH ---
col_graph, col_tutor = st.columns([3, 1])

with col_graph:
    if st.session_state.points:
        fig = go.Figure()
        fig.update_layout(uirevision='static_layout', margin=dict(l=20, r=20, t=20, b=20), height=750, plot_bgcolor='white')
        # Salva il grafico in session_state per usarlo nel report
        st.session_state.current_fig = fig
        xs = [p['x'] for p in st.session_state.points.values()] + [0]; ys = [p['y'] for p in st.session_state.points.values()] + [0]
        max_dim = max(max(map(abs, xs)), max(map(abs, ys))) * 1.5 or 10
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black', gridcolor='#e0e0e0', gridwidth=0.5)
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='black', gridcolor='#e0e0e0', gridwidth=0.5, scaleanchor="x", scaleratio=1)
        fig.add_annotation(x=0, y=max_dim*0.95, text="N (Y)", showarrow=False, font=dict(size=14, color="black", weight="bold"))
        fig.add_annotation(x=max_dim*0.95, y=0, text="E (X)", showarrow=False, font=dict(size=14, color="black", weight="bold"))

        radius = max_dim * 0.2
        
        # 1. Angoli Interni
        pts_lbls = sorted(st.session_state.points.keys())
        greeks = ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]
        for item in st.session_state.solved_items:
            if item.startswith("Ang_"):
                pt_lbl = item.split("_")[1]
                
                ang_lbl = None
                if len(pts_lbls) >= 3:
                    try:
                        idx = pts_lbls.index(pt_lbl)
                        if idx < len(greeks):
                            greek_sym = greeks[idx]
                            # Recupera il valore dell'angolo
                            angle_value = st.session_state.solved_values.get(item, "")
                            if angle_value:
                                try:
                                    angle_display = angle_value.split()[0][:7]
                                except:
                                    angle_display = angle_value[:7]
                                ang_lbl = f"{greek_sym}={angle_display}"
                            else:
                                ang_lbl = greek_sym
                    except: pass
                draw_angle_wedge(fig, pt_lbl, "rgba(0,128,0,0.15)", radius, text=ang_lbl)

        # 1b. Angoli Parziali (Visualizzazione specifica richiesta)
        for item in st.session_state.solved_items:
            if item.startswith("PartAng_"):
                parts = item.split("_")
                pt_lbl = parts[1]

                # Se l'angolo completo per questo vertice è già stato calcolato (es. da una somma),
                # non disegnare i suoi componenti parziali per evitare etichette duplicate.
                if f"Ang_{pt_lbl}" in st.session_state.solved_items:
                    continue

                pA = parts[2]
                pB = parts[3]
                
                try:
                    idx = pts_lbls.index(pt_lbl)
                    
                    # Usa i metadata salvati per determinare suffisso e vertici corretti
                    metadata = st.session_state.partial_angle_metadata.get(item, {})
                    suffix = metadata.get('suffix', '')
                    ordered_pA = metadata.get('pA', pA)
                    ordered_pB = metadata.get('pB', pB)
                    
                    # Estrai il numero dal suffisso (_1 -> 1, _2 -> 2)
                    # Se suffix è vuoto, prova a dedurlo dalla chiave dell'item o dal contesto
                    if suffix:
                        suffix_num = suffix.replace('_', '')
                    else:
                        # Prova a dedurre il suffisso dalla posizione del triangolo
                        # Cerca la diagonale per determinare se è triangolo 1 o 2
                        try:
                            triangle_vertices = {pt_lbl, pA, pB}
                            for p1_lbl_d, p3_lbl_d in itertools.combinations(pts_lbls, 2):
                                idx1_d = pts_lbls.index(p1_lbl_d)
                                idx3_d = pts_lbls.index(p3_lbl_d)
                                # È una diagonale? (non adiacenti)
                                if abs(idx1_d - idx3_d) % (len(pts_lbls)-1) > 1:
                                    d_key1_d = f"SegDist_{p1_lbl_d}_{p3_lbl_d}"
                                    d_key2_d = f"SegDist_{p3_lbl_d}_{p1_lbl_d}"
                                    if d_key1_d in st.session_state.solved_items or d_key2_d in st.session_state.solved_items:
                                        other_pts_d = [p for p in pts_lbls if p not in [p1_lbl_d, p3_lbl_d]]
                                        p2_lbl_d, p4_lbl_d = other_pts_d[0], other_pts_d[1]
                                        triangle_1_set = {p1_lbl_d, p2_lbl_d, p3_lbl_d}
                                        triangle_2_set = {p1_lbl_d, p4_lbl_d, p3_lbl_d}
                                        if triangle_vertices == triangle_1_set:
                                            suffix_num = '1'
                                            # Aggiorna anche il metadata
                                            st.session_state.partial_angle_metadata[item] = {
                                                'suffix': '_1', 'pA': pA, 'pB': pB
                                            }
                                            break
                                        elif triangle_vertices == triangle_2_set:
                                            suffix_num = '2'
                                            # Aggiorna anche il metadata
                                            st.session_state.partial_angle_metadata[item] = {
                                                'suffix': '_2', 'pA': pA, 'pB': pB
                                            }
                                            break
                            else:
                                # Nessuna diagonale trovata, è probabilmente un triangolo completo
                                suffix_num = ''
                        except:
                            suffix_num = ''
                    
                    greek_base = greeks[idx] if idx < len(greeks) else "α"
                    
                    # Recupera il valore dell'angolo
                    angle_value = st.session_state.solved_values.get(item, "")
                    # Estrai solo il valore numerico (es. "45.1234g" -> "45.12g")
                    if angle_value:
                        try:
                            val_parts = angle_value.split()
                            angle_display = val_parts[0][:7]  # Tronca a 7 caratteri
                        except:
                            angle_display = angle_value[:7]
                    else:
                        angle_display = ""
                    
                    if suffix_num:
                        label = f"{greek_base}<sub>{suffix_num}</sub>={angle_display}"
                    else:
                        label = f"{greek_base}={angle_display}"
                    
                    # Disegna l'arco tra pt_lbl, ordered_pA e ordered_pB
                    # Passa il suffisso per determinare quale arco disegnare
                    draw_angle_wedge(fig, pt_lbl, "rgba(0,128,128,0.2)", radius*0.7, text=label, p_from_lbl=ordered_pA, p_to_lbl=ordered_pB, triangle_suffix=suffix)
                except: pass

        # 2. Azimut (Smart Overlap)
        # Contatore per sapere quanti azimut partono da ogni punto
        azimuth_counts = defaultdict(int)
        
        for item in st.session_state.solved_items:
            if item.startswith("SegAz_"):
                parts = item.split("_")
                p1, p2 = parts[1], parts[2]
                
                # Nascondi arco marrone se è gestito dal workflow colorato
                if st.session_state.az_workflow['active'] and st.session_state.az_workflow['vertex'] == p1:
                    continue

                # Calcola offset basato su quanti azimut sono già partiti da p1
                current_offset = azimuth_counts[p1] * (radius * 0.15) 
                draw_azimuth_visuals(fig, p1, p2, radius, current_offset)
                
                # Incrementa il contatore per quel punto
                azimuth_counts[p1] += 1

        # 3. Missione & Deltas (Ghosting per Distanza E Azimut)
        if st.session_state.current_mission:
            mission = st.session_state.current_mission
            
            # NUOVA LOGICA PER PROIEZIONI CARTESIANE
            if "calc_X" in mission or "calc_Y" in mission:
                pt_lbl = mission.split("_")[-1]
                if pt_lbl in st.session_state.points:
                    p = st.session_state.points[pt_lbl]
                    
                    # Evidenzia punto target
                    fig.add_trace(go.Scatter(
                        x=[p['x']], y=[p['y']],
                        mode='markers',
                        marker=dict(size=20, color='rgba(255, 165, 0, 0.5)', symbol='circle-open', line=dict(width=3)),
                        showlegend=False, hoverinfo='skip'
                    ))
                    
                    # Mostra proiezione "ghost" per X
                    if "calc_X" in mission and f"X_{pt_lbl}" not in st.session_state.solved_values:
                        fig.add_trace(go.Scatter(x=[p['x'], p['x']], y=[0, p['y']], mode='lines', line=dict(color='orange', width=2, dash='dot'), showlegend=False, name=f'Proiezione X di {pt_lbl}'))

                    # Mostra proiezione "ghost" per Y
                    if "calc_Y" in mission and f"Y_{pt_lbl}" not in st.session_state.solved_values:
                        fig.add_trace(go.Scatter(x=[0, p['x']], y=[p['y'], p['y']], mode='lines', line=dict(color='orange', width=2, dash='dot'), showlegend=False, name=f'Proiezione Y di {pt_lbl}'))

            # Mostra i delta sia se sto calcolando la distanza, sia se sto calcolando l'azimut
            is_dist_mission = "seg_dist" in st.session_state.current_mission
            is_az_mission = "seg_az" in st.session_state.current_mission
            
            if is_dist_mission or is_az_mission:
                parts = st.session_state.current_mission.split("_")
                p1_lbl, p2_lbl = parts[2], parts[3]
                p1 = st.session_state.points[p1_lbl]
                p2 = st.session_state.points[p2_lbl]
                
                # Check method choice for visualization style
                method_choice = st.session_state.get('ang_method_choice', '')
                
                if is_dist_mission and "Carnot" in method_choice:
                    # --- VISUALIZZAZIONE METODO DIRETTO (CARNOT) ---
                    # Triangolo O-P1-P2
                    fig.add_trace(go.Scatter(x=[0, p1['x'], p2['x'], 0], y=[0, p1['y'], p2['y'], 0], fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', mode='none', showlegend=False))
                    
                    # Nord (Origine)
                    fig.add_trace(go.Scatter(x=[0, 0], y=[0, radius*1.5], mode='lines', line=dict(color='gray', width=1, dash='dash'), showlegend=False))

                    # Azimut 1 (P1) - Ciano
                    deg1 = math.degrees(p1['alpha']) % 360
                    ax1, ay1 = get_azimuth_arc_points_topo([0,0], radius*0.3, 0, deg1)
                    fig.add_trace(go.Scatter(x=ax1, y=ay1, mode='lines', line=dict(color='#00bcd4', width=2), showlegend=False))
                    if len(ax1) > 0:
                         fig.add_trace(go.Scatter(x=[ax1[-1]], y=[ay1[-1]], mode='markers', marker=dict(symbol='arrow-bar-up', size=8, color='#00bcd4', angle=180), showlegend=False))
                         fig.add_annotation(x=ax1[len(ax1)//2], y=ay1[len(ay1)//2], text=f"θ<sub>{p1_lbl}</sub>", font=dict(color="#00bcd4", size=14), showarrow=False, xshift=10, yshift=10, bgcolor="rgba(255,255,255,0.4)")

                    # Azimut 2 (P2) - Arancione
                    deg2 = math.degrees(p2['alpha']) % 360
                    ax2, ay2 = get_azimuth_arc_points_topo([0,0], radius*0.4, 0, deg2)
                    fig.add_trace(go.Scatter(x=ax2, y=ay2, mode='lines', line=dict(color='#ff9800', width=2), showlegend=False))
                    if len(ax2) > 0:
                         fig.add_trace(go.Scatter(x=[ax2[-1]], y=[ay2[-1]], mode='markers', marker=dict(symbol='arrow-bar-up', size=8, color='#ff9800', angle=180), showlegend=False))
                         fig.add_annotation(x=ax2[len(ax2)//2], y=ay2[len(ay2)//2], text=f"θ<sub>{p2_lbl}</sub>", font=dict(color="#ff9800", size=14), showarrow=False, xshift=10, yshift=10, bgcolor="rgba(255,255,255,0.4)")

                    # Raggi O-P1 e O-P2 (Coordinati con gli azimut)
                    fig.add_trace(go.Scatter(x=[0, p1['x']], y=[0, p1['y']], mode='lines', line=dict(color='#00bcd4', width=2, dash='dash'), showlegend=False))
                    fig.add_trace(go.Scatter(x=[0, p2['x']], y=[0, p2['y']], mode='lines', line=dict(color='#ff9800', width=2, dash='dash'), showlegend=False))
                    
                    # Etichette Raggi
                    rot1 = get_text_rotation({'x':0, 'y':0}, p1)
                    rot2 = get_text_rotation({'x':0, 'y':0}, p2)
                    fig.add_annotation(x=p1['x']/2, y=p1['y']/2, text=f"<b>{p1['r']:.2f}</b>", textangle=rot1, font=dict(color="#00bcd4", size=12), showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                    fig.add_annotation(x=p2['x']/2, y=p2['y']/2, text=f"<b>{p2['r']:.2f}</b>", textangle=rot2, font=dict(color="#ff9800", size=12), showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                    
                    # Origine
                    fig.add_annotation(x=0, y=0, text="O", font=dict(color="black", size=20, weight="bold"), yshift=-20, showarrow=False)
                    
                    # Arco Angolo Delta Theta (Filled)
                    diff = (deg2 - deg1) % 360
                    if diff <= 180: start_a, end_a = deg1, deg2
                    else: start_a, end_a = deg2, deg1
                    
                    ax, ay = get_azimuth_arc_points_topo([0,0], radius*0.2, start_a, end_a)
                    
                    # Riempimento colore angolo
                    fill_x = [0] + ax + [0]
                    fill_y = [0] + ay + [0]
                    fig.add_trace(go.Scatter(x=fill_x, y=fill_y, fill='toself', fillcolor='rgba(211, 84, 0, 0.2)', mode='none', showlegend=False))
                    
                    fig.add_trace(go.Scatter(x=ax, y=ay, mode='lines', line=dict(color='#d35400', width=2), showlegend=False))
                    if len(ax) > 0:
                        mid = len(ax)//2
                        fig.add_annotation(x=ax[mid], y=ay[mid], text="Δθ", font=dict(color="#d35400", size=14), showarrow=False, bgcolor="rgba(255,255,255,0.6)")
                    
                    # Segmento P1-P2 (Target)
                    fig.add_trace(go.Scatter(x=[p1['x'], p2['x']], y=[p1['y'], p2['y']], mode='lines', line=dict(color='green', width=3), showlegend=False))

                else:
                    # --- VISUALIZZAZIONE STANDARD (CARTESIANA / PITAGORA) ---
                    corner_x, corner_y = p2['x'], p1['y']
                    
                    # Triangolo Colorato
                    fig.add_trace(go.Scatter(x=[p1['x'], corner_x, p2['x'], p1['x']], y=[p1['y'], corner_y, p2['y'], p1['y']], fill='toself', fillcolor='rgba(255, 235, 59, 0.3)', mode='none', showlegend=False))
                    
                    # Etichette Delta
                    dx = p2['x'] - p1['x']; dy = p2['y'] - p1['y']
                    dx_text = f"ΔX = {p2['x']:.2f} - ({p1['x']:.2f}) = {dx:.2f}"
                    dy_text = f"ΔY = {p2['y']:.2f} - ({p1['y']:.2f}) = {dy:.2f}"
                    fig.add_annotation(x=(p1['x']+corner_x)/2, y=p1['y'], text=dx_text, font=dict(color="green", size=10, weight="bold"), yshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                    fig.add_annotation(x=corner_x, y=(p1['y']+p2['y'])/2, text=dy_text, font=dict(color="magenta", size=10, weight="bold"), xshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                    
                    # Cateti Tratteggiati
                    fig.add_trace(go.Scatter(x=[p1['x'], p2['x']], y=[p1['y'], p2['y']], mode='lines', line=dict(color='black', width=3, dash='dash'), opacity=0.3, showlegend=False)) # Ipotenusa ghost
                    fig.add_trace(go.Scatter(x=[p1['x'], corner_x], y=[p1['y'], corner_y], mode='lines', line=dict(color='green', width=2, dash='dot'), showlegend=False)) # Cateto Delta X
                    fig.add_trace(go.Scatter(x=[corner_x, p2['x']], y=[corner_y, p2['y']], mode='lines', line=dict(color='magenta', width=2, dash='dot'), showlegend=False)) # Cateto Delta Y
                
                # Visualizza arco azimut se è una missione azimut
                if is_az_mission:
                    # Disegna solo se NON siamo nel workflow interattivo per questo vertice
                    if not (st.session_state.az_workflow['active'] and st.session_state.az_workflow['vertex'] == p1_lbl):
                        draw_azimuth_visuals(fig, p1_lbl, p2_lbl, radius, 0)

            if "calc_ang" in st.session_state.current_mission:
                pt_lbl = st.session_state.current_mission.split("_")[2]
                
                # Get meta info to see if it's a partial angle
                specific_method = st.session_state.get('specific_method_choice')
                q_mission = get_strategies_for_mission(st.session_state.current_mission, specific_method)
                meta = q_mission.get('meta')

                if meta and meta.get('type') in ['carnot_tri', 'sines_tri', 'sub_tri']:
                    # It's a partial angle mission. Draw only that wedge.
                    pA, pB = meta['pA'], meta['pB']
                    draw_angle_wedge(fig, pt_lbl, "rgba(255, 165, 0, 0.2)", radius, p_from_lbl=pA, p_to_lbl=pB)
                else:
                    # Default: draw the full angle wedge for the mission
                    draw_angle_wedge(fig, pt_lbl, "rgba(255, 165, 0, 0.2)", radius)
                
                if st.session_state.ang_method_choice == "Calcolo con Azimut vertici":
                    pass # Visuals handled by az_workflow
                    
    # 4. Punti
        pts_lbls = sorted(st.session_state.points.keys()) 
        
        if st.session_state.projections_visible:
            x_levels = calculate_label_levels([(p['x'], lbl) for lbl, p in st.session_state.points.items() if f"X_{lbl}" in st.session_state.solved_values])
            y_levels = calculate_label_levels([(p['y'], lbl) for lbl, p in st.session_state.points.items() if f"Y_{lbl}" in st.session_state.solved_values])

        for lbl, p in st.session_state.points.items():
            pos = get_point_label_pos(p['x'], p['y'])
            
            h_lines = [f"<b>Punto {lbl}</b>"]
            if f"X_{lbl}" in st.session_state.solved_values: h_lines.append(f"X: {st.session_state.solved_values[f'X_{lbl}']}")
            if f"Y_{lbl}" in st.session_state.solved_values: h_lines.append(f"Y: {st.session_state.solved_values[f'Y_{lbl}']}")
            
            fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers+text', 
                                     marker=dict(size=12, color='blue', line=dict(width=2, color='white')), 
                                     text=[lbl], textposition=pos, showlegend=False, 
                                     hoverinfo='text', hovertext="<br>".join(h_lines)))
            
            if st.session_state.projections_visible:
                # Proiezione e valore X (se calcolato)
                if f"X_{lbl}" in st.session_state.solved_values:
                    # Non ridisegnare la linea se è la missione corrente (ci pensa il blocco mission)
                    if not (st.session_state.current_mission and f"calc_X_{lbl}" in st.session_state.current_mission):
                        fig.add_trace(go.Scatter(x=[p['x'], p['x']], y=[0, p['y']], mode='lines', line=dict(color='red', width=1, dash='dash'), showlegend=False))
                    
                    level = x_levels.get(lbl, 0)
                    fig.add_annotation(
                        x=p['x'], y=0,
                        text=f"{st.session_state.solved_values[f'X_{lbl}']}",
                        showarrow=False, yshift=-15 - level * 20,  # Negativo per posizionare SOTTO l'asse
                        font=dict(color="red", size=10),
                        bgcolor="rgba(255,255,255,0.7)"
                    )
                
                # Proiezione e valore Y (se calcolato)
                if f"Y_{lbl}" in st.session_state.solved_values:
                    # Non ridisegnare la linea se è la missione corrente
                    if not (st.session_state.current_mission and f"calc_Y_{lbl}" in st.session_state.current_mission):
                        fig.add_trace(go.Scatter(x=[0, p['x']], y=[p['y'], p['y']], mode='lines', line=dict(color='red', width=1, dash='dash'), showlegend=False))
                    
                    level = y_levels.get(lbl, 0)
                    fig.add_annotation(
                        x=0, y=p['y'],
                        text=f"{st.session_state.solved_values[f'Y_{lbl}']}",
                        showarrow=False, xshift=-25 - (level * 40),
                        textangle= -90,
                        font=dict(color="red", size=10),
                        bgcolor="rgba(255,255,255,0.7)"
                    )
                
    # 5. Linee Segmenti
        for item in list(st.session_state.solved_items):
            if item.startswith("Seg_") or item.startswith("SegDist_"):
                # Estraiamo i nomi dei punti (es. da SegDist_A_B o Seg_A_B)
                parts = item.split("_")
                if len(parts) < 3: continue
                p1_lbl, p2_lbl = parts[1], parts[2]
                
                if p1_lbl in st.session_state.points and p2_lbl in st.session_state.points:
                    p1 = st.session_state.points[p1_lbl]
                    p2 = st.session_state.points[p2_lbl]
                    
                    # Disegna la linea verde del segmento
                    fig.add_trace(go.Scatter(
                        x=[p1['x'], p2['x']], 
                        y=[p1['y'], p2['y']], 
                        mode='lines', 
                        line=dict(color='green', width=2), 
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Verifichiamo se abbiamo il valore della distanza per scrivere l'etichetta
                    dist_key = f"SegDist_{p1_lbl}_{p2_lbl}"
                    dist_key_rev = f"SegDist_{p2_lbl}_{p1_lbl}"
                    
                    val = st.session_state.solved_values.get(dist_key) or st.session_state.solved_values.get(dist_key_rev)
                    
                    if val:
                        mx, my = (p1['x'] + p2['x']) / 2, (p1['y'] + p2['y']) / 2
                        rot_angle = get_text_rotation(p1, p2)
                        
                        # Determiniamo il nome del lato (a, b, c...) per il triangolo/poligono
                        lbl_text = val
                        try:
                            n = len(pts_lbls)
                            i1 = pts_lbls.index(p1_lbl)
                            i2 = pts_lbls.index(p2_lbl)
                            # Se i punti sono consecutivi nel poligono
                            if (i1 + 1) % n == i2: side_name = chr(ord('a') + i1)
                            elif (i2 + 1) % n == i1: side_name = chr(ord('a') + i2)
                            else: side_name = None # Diagonale
                            
                            if side_name: lbl_text = f"{side_name} = {val}"
                        except:
                            pass
                        
                        # Aggiunge l'etichetta sul disegno
                        fig.add_annotation(
                            x=mx, y=my, 
                            text=lbl_text, 
                            textangle=rot_angle, 
                            font=dict(color="green", size=11, weight="bold"), 
                            bgcolor="white", 
                            borderpad=2, 
                            showarrow=False
                        )            

        # 6. Watermarks for Triangulation
        if len(pts_lbls) == 4:
            p_map = st.session_state.points
            for p1_lbl, p3_lbl in itertools.combinations(pts_lbls, 2):
                idx1 = pts_lbls.index(p1_lbl)
                idx3 = pts_lbls.index(p3_lbl)
                
                # Identifica se è una diagonale (non adiacenti)
                if abs(idx1 - idx3) % (len(pts_lbls)-1) > 1:
                    d_key1 = f"SegDist_{p1_lbl}_{p3_lbl}"
                    d_key2 = f"SegDist_{p3_lbl}_{p1_lbl}"
                    
                    if d_key1 in st.session_state.solved_values or d_key2 in st.session_state.solved_values:
                        other_pts = [p for p in pts_lbls if p not in [p1_lbl, p3_lbl]]
                        p2_lbl, p4_lbl = other_pts[0], other_pts[1]
                        
                        # Baricentri
                        c1_x = (p_map[p1_lbl]['x'] + p_map[p2_lbl]['x'] + p_map[p3_lbl]['x']) / 3
                        c1_y = (p_map[p1_lbl]['y'] + p_map[p2_lbl]['y'] + p_map[p3_lbl]['y']) / 3
                        c2_x = (p_map[p1_lbl]['x'] + p_map[p4_lbl]['x'] + p_map[p3_lbl]['x']) / 3
                        c2_y = (p_map[p1_lbl]['y'] + p_map[p4_lbl]['y'] + p_map[p3_lbl]['y']) / 3
                        
                        fig.add_annotation(x=c1_x, y=c1_y, text="1", font=dict(size=80, color="rgba(0,0,0,0.08)"), showarrow=False)
                        fig.add_annotation(x=c2_x, y=c2_y, text="2", font=dict(size=80, color="rgba(0,0,0,0.08)"), showarrow=False)
                        break
        # [SEZIONE 3: INIEZIONE GRAFICA AGGIORNATA PER GESTIRE IL PENDING]
        if st.session_state.az_workflow['active']:
            wf = st.session_state.az_workflow
            v = wf['vertex']
            orig = st.session_state.points[v]
            r_vis = max_dim * 0.2 

            # Nord sempre visibile
            fig.add_trace(go.Scatter(x=[orig['x'], orig['x']], y=[orig['y'], orig['y'] + r_vis*1.2], mode='lines', line=dict(color='gray', dash='dash'), name='Nord'))

            # --- GESTIONE AZIMUT 1 (CIANO) ---
            # Caso A: Già calcolato (Step > 1) o confermato
            if wf['step'] >= 2:
                ax, ay = get_azimuth_arc_points_topo([orig['x'], orig['y']], r_vis, 0, wf['az1_val'])
                fig.add_trace(go.Scatter(x=ax, y=ay, mode='lines', line=dict(color='#00bcd4', width=3), name='Az 1'))
                if ax: fig.add_trace(go.Scatter(x=[ax[-1]], y=[ay[-1]], mode='markers', marker=dict(symbol='arrow-bar-up', size=10, color='#00bcd4', angle=180), showlegend=False))
                p1 = st.session_state.points[wf['side1']]
                fig.add_trace(go.Scatter(x=[orig['x'], p1['x']], y=[orig['y'], p1['y']], mode='lines', line=dict(color='#00bcd4', width=1, dash='dot'), showlegend=False))
            
            # Caso B: In fase di calcolo (Pending) - Step 1
            elif wf['step'] == 1 and wf['pending_target']:
                # Calcoliamo le coordinate "volanti" solo per il disegno, senza salvare il valore
                pt_target = st.session_state.points[wf['pending_target']]
                dx, dy = pt_target['x'] - orig['x'], pt_target['y'] - orig['y']
                temp_az = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                
                # Disegno arco "Ghost" o attivo
                ax, ay = get_azimuth_arc_points_topo([orig['x'], orig['y']], r_vis, 0, temp_az)
                fig.add_trace(go.Scatter(x=ax, y=ay, mode='lines', line=dict(color='#00bcd4', width=3, dash='dot'), name='Az 1 (Calcolo...)'))
                if ax: fig.add_trace(go.Scatter(x=[ax[-1]], y=[ay[-1]], mode='markers', marker=dict(symbol='arrow-bar-up', size=10, color='#00bcd4', angle=180), showlegend=False))
                fig.add_trace(go.Scatter(x=[orig['x'], pt_target['x']], y=[orig['y'], pt_target['y']], mode='lines', line=dict(color='#00bcd4', width=1, dash='dot'), showlegend=False))
                
                # --- VISUALIZZAZIONE TRIANGOLO DELTA (Step 1) ---
                corner_x, corner_y = pt_target['x'], orig['y']
                fig.add_trace(go.Scatter(x=[orig['x'], corner_x, pt_target['x'], orig['x']], y=[orig['y'], corner_y, pt_target['y'], orig['y']], fill='toself', fillcolor='rgba(255, 235, 59, 0.3)', mode='none', showlegend=False))
                fig.add_trace(go.Scatter(x=[orig['x'], corner_x], y=[orig['y'], corner_y], mode='lines', line=dict(color='green', width=2, dash='dot'), showlegend=False))
                fig.add_trace(go.Scatter(x=[corner_x, pt_target['x']], y=[corner_y, pt_target['y']], mode='lines', line=dict(color='magenta', width=2, dash='dot'), showlegend=False))
                fig.add_annotation(x=(orig['x']+corner_x)/2, y=orig['y'], text=f"ΔX={dx:.2f}", font=dict(color="green", size=10, weight="bold"), yshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                fig.add_annotation(x=corner_x, y=(orig['y']+pt_target['y'])/2, text=f"ΔY={dy:.2f}", font=dict(color="magenta", size=10, weight="bold"), xshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")


            # --- GESTIONE AZIMUT 2 (ARANCIONE) ---
            # Caso A: Già calcolato
            if wf['step'] >= 3:
                r_vis2 = r_vis * 1.2
                ax2, ay2 = get_azimuth_arc_points_topo([orig['x'], orig['y']], r_vis2, 0, wf['az2_val'])
                fig.add_trace(go.Scatter(x=ax2, y=ay2, mode='lines', line=dict(color='#ff9800', width=3), name='Az 2'))
                if ax2: fig.add_trace(go.Scatter(x=[ax2[-1]], y=[ay2[-1]], mode='markers', marker=dict(symbol='arrow-bar-up', size=10, color='#ff9800', angle=180), showlegend=False))
                p2 = st.session_state.points[wf['side2']]
                fig.add_trace(go.Scatter(x=[orig['x'], p2['x']], y=[orig['y'], p2['y']], mode='lines', line=dict(color='#ff9800', width=1, dash='dot'), showlegend=False))

            # Caso B: In fase di calcolo (Pending) - Step 2
            elif wf['step'] == 2 and wf['pending_target']:
                r_vis2 = r_vis * 1.2
                pt_target = st.session_state.points[wf['pending_target']]
                dx, dy = pt_target['x'] - orig['x'], pt_target['y'] - orig['y']
                temp_az = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                
                ax2, ay2 = get_azimuth_arc_points_topo([orig['x'], orig['y']], r_vis2, 0, temp_az)
                fig.add_trace(go.Scatter(x=ax2, y=ay2, mode='lines', line=dict(color='#ff9800', width=3, dash='dot'), name='Az 2 (Calcolo...)'))
                if ax2: fig.add_trace(go.Scatter(x=[ax2[-1]], y=[ay2[-1]], mode='markers', marker=dict(symbol='arrow-bar-up', size=10, color='#ff9800', angle=180), showlegend=False))
                fig.add_trace(go.Scatter(x=[orig['x'], pt_target['x']], y=[orig['y'], pt_target['y']], mode='lines', line=dict(color='#ff9800', width=1, dash='dot'), showlegend=False))

                # --- VISUALIZZAZIONE TRIANGOLO DELTA (Step 2) ---
                corner_x, corner_y = pt_target['x'], orig['y']
                fig.add_trace(go.Scatter(x=[orig['x'], corner_x, pt_target['x'], orig['x']], y=[orig['y'], corner_y, pt_target['y'], orig['y']], fill='toself', fillcolor='rgba(255, 235, 59, 0.3)', mode='none', showlegend=False))
                fig.add_trace(go.Scatter(x=[orig['x'], corner_x], y=[orig['y'], corner_y], mode='lines', line=dict(color='green', width=2, dash='dot'), showlegend=False))
                fig.add_trace(go.Scatter(x=[corner_x, pt_target['x']], y=[corner_y, pt_target['y']], mode='lines', line=dict(color='magenta', width=2, dash='dot'), showlegend=False))
                fig.add_annotation(x=(orig['x']+corner_x)/2, y=orig['y'], text=f"ΔX={dx:.2f}", font=dict(color="green", size=10, weight="bold"), yshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                fig.add_annotation(x=corner_x, y=(orig['y']+pt_target['y'])/2, text=f"ΔY={dy:.2f}", font=dict(color="magenta", size=10, weight="bold"), xshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")

        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    else:
        st.info("Carica i dati.")

# --- TUTOR LOGIC (STRICT DEPENDENCY) ---
def get_available_targets():
    pts_keys = sorted(st.session_state.points.keys())
    
    # 1. Punti
    opt_punti = []
    for p in pts_keys:
        missing_cart = (f"X_{p}" not in st.session_state.solved_items) or (f"Y_{p}" not in st.session_state.solved_items)
        missing_pol = (f"Dist_{p}" not in st.session_state.solved_items) or (f"Az_{p}" not in st.session_state.solved_items)
        
        # Mostra il punto se:
        # 1) Mancano coordinate polari (caso normale)
        # 2) Mancano coordinate cartesiane (anche se hai le polari, potrebbero servire per azimut)
        if missing_pol or missing_cart:
            opt_punti.append(f"Punto {p}")
            
    # 2. Segmenti (Logica Rigida: Distanza -> Azimut)
    opt_seg = []
    if len(pts_keys) >= 2:
        # Distanze (Non direzionali - Combinazioni)
        for c in list(itertools.combinations(pts_keys, 2)):
            # Controlla se la distanza esiste in una delle due direzioni
            if f"SegDist_{c[0]}_{c[1]}" not in st.session_state.solved_items and f"SegDist_{c[1]}_{c[0]}" not in st.session_state.solved_items:
                p1, p2 = c[0], c[1]
                
                # Verifica se almeno uno dei due metodi è disponibile
                # Metodo 1: Cartesiano (Pitagora) - richiede X e Y di entrambi i punti
                cart_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items and
                          f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                
                # Metodo 2: Polare (Carnot) - richiede Dist e Az di entrambi i punti
                pol_ok = (f"Dist_{p1}" in st.session_state.solved_items and f"Az_{p1}" in st.session_state.solved_items and
                         f"Dist_{p2}" in st.session_state.solved_items and f"Az_{p2}" in st.session_state.solved_items)
                
                # Mostra l'opzione solo se almeno uno dei due metodi è disponibile
                if cart_ok or pol_ok:
                    opt_seg.append(f"Lunghezza {c[0]}{c[1]}")
        
        # Azimut (Direzionali - Permutazioni)
        # IMPORTANTE: Gli azimut dei segmenti richiedono le coordinate cartesiane (X,Y) 
        # perché si calcolano con arctan(ΔX/ΔY)
        for p1 in pts_keys:
            for p2 in pts_keys:
                if p1 == p2: continue
                dist_known = f"SegDist_{p1}_{p2}" in st.session_state.solved_items or f"SegDist_{p2}_{p1}" in st.session_state.solved_items
                az_known = f"SegAz_{p1}_{p2}" in st.session_state.solved_items
                
                # Verifica che entrambi i punti abbiano coordinate cartesiane
                cart_p1_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items)
                cart_p2_ok = (f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                
                # Mostra l'azimut solo se: 1) distanza nota, 2) azimut non ancora calcolato, 3) coordinate cartesiane disponibili
                if dist_known and not az_known and cart_p1_ok and cart_p2_ok:
                    opt_seg.append(f"Azimut {p1} → {p2}")


    # 3. Angoli (Logica Migliorata)
    opt_ang = []
    if len(pts_keys) >= 3:
        greeks_latex = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon"]
        greeks_map = {lbl: sym for lbl, sym in zip(pts_keys, greeks_latex)}

        # Trova la diagonale risolta (se esiste)
        solved_diagonal = None
        diagonal_endpoints = set()
        triangles = {} # {1: frozenset, 2: frozenset}
        
        if len(pts_keys) == 4:
            for p1, p3 in itertools.combinations(pts_keys, 2):
                if abs(pts_keys.index(p1) - pts_keys.index(p3)) % 3 > 1:
                    if f"SegDist_{p1}_{p3}" in st.session_state.solved_values or f"SegDist_{p3}_{p1}" in st.session_state.solved_values:
                        solved_diagonal = tuple(sorted((p1, p3)))
                        diagonal_endpoints = {p1, p3}
                        other_verts = sorted([p for p in pts_keys if p not in diagonal_endpoints])
                        triangles[1] = frozenset({p1, other_verts[0], p3})
                        triangles[2] = frozenset({p1, other_verts[1], p3})
                        break

        for p in pts_keys:
            if f"Ang_{p}" in st.session_state.solved_items:
                continue

            greek_p_latex = greeks_map.get(p, f"\\alpha_{{{p}}}")
            greek_p_unicode = latex_to_unicode_greek(greek_p_latex)

            # Caso 1: L'angolo 'p' è un vertice della diagonale (quindi è potenzialmente diviso)
            if p in diagonal_endpoints:
                # Trova i due angoli parziali associati
                other_diag_pt = next(iter(diagonal_endpoints - {p}))
                
                # Partiale 1
                verts1 = triangles[1] - {p}
                k_part1 = f"PartAng_{p}_{min(*verts1)}_{max(*verts1)}"
                
                # Partiale 2
                verts2 = triangles[2] - {p}
                k_part2 = f"PartAng_{p}_{min(*verts2)}_{max(*verts2)}"

                part1_solved = k_part1 in st.session_state.solved_items
                part2_solved = k_part2 in st.session_state.solved_items

                if part1_solved and part2_solved:
                    opt_ang.append(f"Angolo completo {greek_p_unicode} (Somma) in {p}")
                else:
                    if not part1_solved:
                        opt_ang.append(f"Angolo parziale {greek_p_unicode}₁ in {p}")
                    if not part2_solved:
                        opt_ang.append(f"Angolo parziale {greek_p_unicode}₂ in {p}")
            
            # Caso 2: L'angolo 'p' non è sulla diagonale, OPPURE non ci sono diagonali. È un angolo intero.
            else:
                can_solve_full = False
                
                # Sub-caso A: Il poligono è un triangolo e tutti i lati sono noti.
                if len(pts_keys) == 3:
                    s1, s2, s3 = [c for c in itertools.combinations(pts_keys, 2)]
                    sides_ok = True
                    for v1, v2 in [s1, s2, s3]:
                        if not (f"SegDist_{v1}_{v2}" in st.session_state.solved_values or f"SegDist_{v2}_{v1}" in st.session_state.solved_values):
                            sides_ok = False
                            break
                    if sides_ok: can_solve_full = True

                # Sub-caso B: Vertice "libero" in un quadrilatero diviso (es. A in ABCD con diagonale BD)
                elif solved_diagonal:
                    triangle_of_p = triangles[1] if p in triangles[1] else (triangles[2] if p in triangles[2] else None)
                    if triangle_of_p:
                        sides_ok = True
                        for v1, v2 in itertools.combinations(triangle_of_p, 2):
                            if not (f"SegDist_{v1}_{v2}" in st.session_state.solved_values or f"SegDist_{v2}_{v1}" in st.session_state.solved_values):
                                sides_ok = False; break
                        if sides_ok: can_solve_full = True
                
                # Sub-caso C: Differenza di Azimut (sempre possibile se i segmenti adiacenti sono noti)
                idx = pts_keys.index(p)
                prev_p = pts_keys[idx-1]; next_p = pts_keys[(idx+1)%len(pts_keys)]
                if (f"SegAz_{p}_{prev_p}" in st.session_state.solved_items and f"SegAz_{p}_{next_p}" in st.session_state.solved_items):
                    can_solve_full = True

                if can_solve_full:
                    opt_ang.append(f"Angolo interno in {p}")
    
    opt_poly = []
    if len(pts_keys) >= 3:
        known_angles = [x for x in st.session_state.solved_items if x.startswith("Ang_")]
        
        # Verifica completezza lati perimetro
        perimeter_complete = True
        for i in range(len(pts_keys)):
            p1, p2 = pts_keys[i], pts_keys[(i+1)%len(pts_keys)]
            if f"SegDist_{p1}_{p2}" not in st.session_state.solved_items and f"SegDist_{p2}_{p1}" not in st.session_state.solved_items:
                perimeter_complete = False; break
        
        can_area = (len(known_angles) == len(pts_keys))
        can_perim = perimeter_complete
        
        if (can_area and "Area_Poly" not in st.session_state.solved_items) or (can_perim and "Perim_Poly" not in st.session_state.solved_items):
            opt_poly.append("Intero Poligono")
             
    return opt_punti + opt_seg + opt_ang + list(set(opt_poly))

# [Motore di Calcolo identico alla v65 - Copiato per completezza]
def check_goal_feasibility(subject, action):
    if "Intero Poligono" in subject:
        if action == "Area": return True, "OK", "calc_poly_area"
        if action == "Perimetro": return True, "OK", "calc_poly_perim"
    
    # Gestione unificata per tutti i tipi di angolo
    if "Angolo" in subject:
        pt = subject.split(" in ")[-1]
        return True, "OK", f"calc_ang_{pt}"

    if subject.startswith("Punto"):
        target = subject.split(" ")[1]
        if action == "Coord. X": return True, "OK", f"calc_X_{target}"
        if action == "Coord. Y": return True, "OK", f"calc_Y_{target}"
        if action == "Distanza (da O)": return True, "OK", f"calc_dist_{target}"
        if action == "Azimut (da O)": return True, "OK", f"calc_az_{target}"
    if subject.startswith("Lunghezza"):
        p1, p2 = subject.split()[1][0], subject.split()[1][1]
        return True, "OK", f"seg_dist_{p1}_{p2}"
    if subject.startswith("Azimut"):
        parts = subject.replace("Azimut ", "").split(" → ")
        p1, p2 = parts[0], parts[1]
        return True, "OK", f"seg_az_{p1}_{p2}"
    return False, "Non disponibile", None
with col_tutor:
    st.subheader("🎯 Tutor")
    
    # Pulsante "Termina Esercizio" sempre visibile
    st.markdown("### 🏁 Consegna")
    
    cognome_sanitized = st.session_state.get('student_surname', '').strip().replace(" ", "_")
    nome_sanitized = st.session_state.get('student_name', '').strip().replace(" ", "_")
    classe_sanitized = st.session_state.get('student_class', '').strip().replace(" ", "_")
    
    cognome_original = st.session_state.get('student_surname', '').strip()
    nome_original = st.session_state.get('student_name', '').strip()

    if cognome_sanitized and nome_sanitized:
        now = datetime.now()
        # Formato: COGNOME_Nome_CLASSE_gg_mm_aaaa_hh_mm
        data_ora = now.strftime("%d_%m_%Y_%H_%M")
        
        # Cognome in maiuscolo, Nome con prima lettera maiuscola
        cognome_upper = cognome_sanitized.upper()
        nome_capitalized = nome_sanitized.capitalize()
        
        if classe_sanitized:
            # Con classe: ROSSI_Mario_3A_07_02_2025_14_30
            report_filename = f"{cognome_upper}_{nome_capitalized}_{classe_sanitized}_{data_ora}.pdf"
        else:
            # Senza classe: ROSSI_Mario_07_02_2025_14_30
            report_filename = f"{cognome_upper}_{nome_capitalized}_{data_ora}.pdf"
    else:
        report_filename = "geosolver_report.pdf"

    # === GENERAZIONE PDF (Metodo Browser - Universalmente Compatibile) ===
    if st.session_state.log or st.session_state.points:
        st.markdown("---")
        
        # Genera nome file: Cognome_Nome_gg_mm_aaaa_hh_mm.pdf
        now = datetime.now()
        cognome = st.session_state.student_surname if st.session_state.student_surname else "Studente"
        nome = st.session_state.student_name if st.session_state.student_name else "Anonimo"
        pdf_filename = f"{cognome}_{nome}_{now.strftime('%d_%m_%Y_%H_%M')}.pdf"
        
        # Bottone per generare e scaricare PDF
        if st.button("📥 Scarica Report PDF", type="primary", use_container_width=True):
            # Verifica prerequisiti
            if not REPORTLAB_AVAILABLE:
                st.error("❌ ReportLab non installato!")
                st.code("pip3 install reportlab pillow", language="bash")
                st.stop()
            
            with st.spinner("⏳ Generazione PDF in corso..."):
                pdf_bytes = generate_pdf_with_reportlab(
                    st.session_state.log,
                    st.session_state.solved_values,
                    cognome,
                    nome,
                    st.session_state.get('current_fig', None)
                )
                
                if pdf_bytes:
                    st.success(f"✅ PDF generato! ({len(pdf_bytes) // 1024} KB)")
                    st.download_button(
                        label="💾 Salva PDF",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Errore nella generazione del PDF")
                    st.warning("**Controlla il terminale per dettagli dell'errore**")
                    with st.expander("🔧 Soluzioni"):
                        st.markdown("""
                        **Installa le librerie necessarie:**
                        ```bash
                        pip3 install reportlab pillow kaleido
                        ```
                        
                        **Se kaleido non funziona:**
                        - Il PDF verrà generato comunque senza grafico
                        - Su Mac 12: kaleido potrebbe non funzionare, usa Mac 13+
                        
                        **Verifica installazione:**
                        ```python
                        import reportlab
                        print("ReportLab OK")
                        ```
                        """)
        
        if st.session_state.log:
            steps_count = len(st.session_state.log)
            errors_count = sum(1 for entry in st.session_state.log if entry.get('is_error', False))
            st.caption(f"📊 Passaggi: {steps_count} | ❌ Errori: {errors_count}")
    else:
        st.info("ℹ️ Carica un esercizio per iniziare")
    
    st.divider()
    
    # Pulsante Annulla (solo se c'è un log)
    if st.session_state.log:
        st.markdown("### ⚙️ Controlli")
        st.button("↩️ Annulla ultima operazione", on_click=undo_last_action, use_container_width=True)
    
    if st.session_state.last_calc_msg:
        st.success("✅ **Fatto:**")
        st.latex(st.session_state.last_calc_msg.replace("°", r"^{\circ}"))

    # [SEZIONE 4: INIEZIONE LOGICA INTERATTIVA (Step-by-Step)]
    # --- BLOCCO WORKFLOW AZIMUT AGGIORNATO (DIDATTICO) ---
    if st.session_state.az_workflow['active']:
        wf = st.session_state.az_workflow
        v = wf['vertex']
        
        st.info(f"🔵 **Procedura Azimut: Vertice {v}**")
        
        # Funzione helper interna per gestire il quiz
        def handle_azimuth_quiz(step_num, az_color_name, side_key_in_state, val_key_in_state):
            target = wf['pending_target']
            
            st.markdown(f"**Step {step_num} (Calcolo):**")
            st.write(f"Hai scelto la direzione **{v} → {target}** ({az_color_name}).")
            st.write("Quale formula usiamo per calcolare questo Azimut?")
            
           # Genera opzioni se non esistono (o se è cambiato il target)
            if not wf.get('quiz_options'):
                # Dati corretti per visualizzazione LaTeX
                # NOTA: I simboli $ all'inizio e alla fine dicono a Streamlit di renderizzare la formula
                correct_latex = rf"$({v}{target}) = \arctan(\Delta X / \Delta Y)$"
                wrong1 = rf"$({v}{target}) = \arctan(\Delta Y / \Delta X)$"
                wrong2 = rf"$({v}{target}) = \sin(\Delta X \cdot \Delta Y)$"
                
                # Salviamo nello stato QUAL È la risposta giusta, così il controllo non sbaglia mai
                wf['current_correct_ans'] = correct_latex
                wf['quiz_options'] = get_shuffled_options(correct_latex, [wrong1, wrong2])
            
            # Mostra Form
            with st.form(f"quiz_az_{step_num}"):
                ans = st.radio("Scegli la formula:", wf['quiz_options'])
                
                if st.form_submit_button("Verifica e Calcola"):
                    # CONFRONTO DIRETTO: Controlla se la stringa scelta è identica a quella salvata come corretta
                    if ans == wf.get('current_correct_ans'):
                        # Calcolo Reale
                        p_start = st.session_state.points[v]
                        p_end = st.session_state.points[target]
                        dx = p_end['x'] - p_start['x']
                        dy = p_end['y'] - p_start['y']
                        
                        # Logica Quadranti
                        raw_atan = math.atan2(dx, dy)
                        deg = (math.degrees(raw_atan) + 360) % 360
                        
                        # Aggiorna stato workflow
                        wf[side_key_in_state] = target
                        wf[val_key_in_state] = deg
                        
                        # Salva nel sistema globale
                        k_az = f"SegAz_{v}_{target}"
                        st.session_state.solved_items.add(k_az)
                        st.session_state.solved_values[k_az] = format_angle_output(math.radians(deg), st.session_state.input_interpretation)
                        
                        # Display corretto in base all'unità e Report Dettagliato
                        unit = st.session_state.input_interpretation
                        val_disp = deg * (200/180) if unit == AngleUnit.GON else deg
                        sym = r"^g" if unit == AngleUnit.GON else r"^{\circ}"
                        
                        st.session_state.last_calc_msg = rf"({v}{target}) = {val_disp:.4f}{sym}"
                        
                        # Costruzione Report Dettagliato (Azimut)
                        raw_atan_calc = math.atan(dx/dy) if abs(dy) > 1e-9 else (math.pi/2 if dx > 0 else -math.pi/2)
                        raw_atan_fmt = format_angle_output(raw_atan_calc, unit, latex=True)

                        quad_name, corr_str = "?", ""
                        if dx >= 0 and dy >= 0:
                            quad_name, corr_str = r"I^{\circ} (\Delta X+, \Delta Y+)", ""
                        elif dx >= 0 and dy < 0:
                            quad_name, corr_str = r"II^{\circ} (\Delta X+, \Delta Y-)", (r"+ 180^{\circ}" if unit == AngleUnit.DEG else (r"+ 200^g" if unit == AngleUnit.GON else r"+ \pi"))
                        elif dx < 0 and dy < 0:
                            quad_name, corr_str = r"III^{\circ} (\Delta X-, \Delta Y-)", (r"+ 180^{\circ}" if unit == AngleUnit.DEG else (r"+ 200^g" if unit == AngleUnit.GON else r"+ \pi"))
                        else:
                            quad_name, corr_str = r"IV^{\circ} (\Delta X-, \Delta Y+)", (r"+ 360^{\circ}" if unit == AngleUnit.DEG else (r"+ 400^g" if unit == AngleUnit.GON else r"+ 2\pi"))
                        
                        final_az_fmt = format_angle_output(math.radians(deg), unit, latex=True)

                        latex_msg_for_report = rf"""\begin{{aligned}}
\Delta X &= {p_end['x']:.2f} - ({p_start['x']:.2f}) = \mathbf{{{dx:.2f}}} \\
\Delta Y &= {p_end['y']:.2f} - ({p_start['y']:.2f}) = \mathbf{{{dy:.2f}}} \\
\text{{Quadrante}} &: {quad_name} \\
\alpha_{{calc}} &= \arctan(\Delta X / \Delta Y) = {raw_atan_fmt} \\
({v}{target}) &= {raw_atan_fmt} \quad {corr_str} \\
&= \mathbf{{{final_az_fmt}}}
\end{{aligned}}"""

                        # Log
                        st.session_state.log.append({
                            'action': f"Azimut {v}-{target}",
                            'method': ans.replace("$", ""), # Rimuovi i $ per il log pulito
                            'result': latex_msg_for_report,
                            'desc_verbose': f"Calcolo azimut passo-passo nel workflow grafico."
                        })
                        
                        # Pulizia e Avanzamento
                        wf['step'] += 1
                        wf['pending_target'] = None
                        wf['quiz_options'] = None
                        wf['current_correct_ans'] = None # Reset risposta corretta
                        st.balloons()
                        st.rerun()
                    else:
                        # Registra l'errore nel log
                        st.session_state.log.append({
                            'action': f"Azimut {v}-{target}",
                            'method': ans.replace("$", ""),
                            'result': r'\text{ERRORE: Formula errata}',
                            'desc_verbose': f"Hai scelto una formula sbagliata: {ans}. La formula corretta era: {wf.get('current_correct_ans', 'N/A')}",
                            'added_items': [],
                            'added_values': [],
                            'is_error': True
                        })
                        st.error("Formula errata. Ricorda: l'azimut topografico è l'arcotangente di Delta X su Delta Y.")
        # --- LOGICA DEGLI STEP ---

        # STEP 1A: SELEZIONE PRIMO LATO
        if wf['step'] == 1 and wf['pending_target'] is None:
            st.markdown(f"**Step 1:** Seleziona il **primo lato** uscente da {v} (Ciano):")
            # Usa i valid_neighbors se disponibili, altrimenti tutti i punti tranne v
            valid_neighbors = wf.get('valid_neighbors', [])
            if valid_neighbors:
                neighbors = valid_neighbors
            else:
                neighbors = [k for k in st.session_state.points.keys() if k != v]
            
            cols = st.columns(len(neighbors))
            for i, n_lbl in enumerate(neighbors):
                if cols[i].button(f"Verso {n_lbl}", key=f"btn_az1_{n_lbl}"):
                    wf['pending_target'] = n_lbl
                    st.rerun()

        # STEP 1B: CALCOLO PRIMO LATO (QUIZ)
        elif wf['step'] == 1 and wf['pending_target'] is not None:
            handle_azimuth_quiz(1, "Arco Ciano", 'side1', 'az1_val')
            if st.button("🔙 Cambia lato"): # Opzione per tornare indietro
                wf['pending_target'] = None
                wf['quiz_options'] = None
                st.rerun()

        # STEP 2A: SELEZIONE SECONDO LATO
        elif wf['step'] == 2 and wf['pending_target'] is None:
            st.success(f"✅ Azimut 1 ({wf['side1']}) calcolato.")
            st.markdown(f"**Step 2:** Seleziona il **secondo lato** (Arancione):")
            # Usa i valid_neighbors se disponibili, altrimenti tutti i punti tranne v e side1
            valid_neighbors = wf.get('valid_neighbors', [])
            if valid_neighbors:
                neighbors = [k for k in valid_neighbors if k != wf['side1']]
            else:
                neighbors = [k for k in st.session_state.points.keys() if k != v and k != wf['side1']]
            
            cols = st.columns(len(neighbors))
            for i, n_lbl in enumerate(neighbors):
                if cols[i].button(f"Verso {n_lbl}", key=f"btn_az2_{n_lbl}"):
                    wf['pending_target'] = n_lbl
                    st.rerun()

        # STEP 2B: CALCOLO SECONDO LATO (QUIZ)
        elif wf['step'] == 2 and wf['pending_target'] is not None:
            handle_azimuth_quiz(2, "Arco Arancione", 'side2', 'az2_val')
            if st.button("🔙 Cambia lato"):
                wf['pending_target'] = None
                wf['quiz_options'] = None
                st.rerun()

        # STEP 3: CALCOLO FINALE DIFFERENZA
        elif wf['step'] == 3:
            az1, az2 = wf['az1_val'], wf['az2_val']
            az1_fmt = format_angle_output(math.radians(az1), st.session_state.input_interpretation)
            az2_fmt = format_angle_output(math.radians(az2), st.session_state.input_interpretation)
            
            st.markdown(f"""
            Abbiamo trovato i due orientamenti:
            - <b style='color:#00bcd4'>Azimut 1 ({v}→{wf['side1']}): {az1_fmt}</b>
            - <b style='color:#ff9800'>Azimut 2 ({v}→{wf['side2']}): {az2_fmt}</b>
            """, unsafe_allow_html=True)
            
            st.write("Ora calcoliamo l'angolo interno.")
            
            with st.form("final_calc"):
                st.write("Come troviamo l'angolo tra i due azimut?")
                # Opzioni logiche
                opts = [
                    "Sottrazione: |Azimut 1 - Azimut 2|",
                    "Somma: Azimut 1 + Azimut 2",
                    "Prodotto: Azimut 1 * Azimut 2"
                ]
                ans = st.radio("Operazione:", opts)
                
                if st.form_submit_button("Calcola Angolo Interno"):
                    if "Sottrazione" in ans:
                        # Gestione Unità per il calcolo visuale
                        unit = st.session_state.input_interpretation
                        factor = 200/180 if unit == AngleUnit.GON else 1.0
                        full_circle = 400 if unit == AngleUnit.GON else 360
                        sym = r"^g" if unit == AngleUnit.GON else r"^{\circ}"
                        
                        az1_disp = az1 * factor
                        az2_disp = az2 * factor
                        
                        diff_disp = abs(az1_disp - az2_disp)
                        if diff_disp > full_circle / 2: diff_disp = full_circle - diff_disp
                        
                        # Determina la chiave corretta: PartAng se è parziale, Ang se completo
                        # Usa il suffisso salvato nel workflow (più affidabile di selected_partial_suffix)
                        partial_suffix = st.session_state.az_workflow.get('partial_suffix')
                        is_partial = partial_suffix is not None
                        
                        if is_partial and len(pts_lbls := sorted(st.session_state.points.keys())) == 4:
                            # Angolo parziale - usa PartAng_{v}_{pt1}_{pt2}
                            # I due punti sono side1 e side2 (ordinati alfabeticamente)
                            pt1, pt2 = sorted([wf['side1'], wf['side2']])
                            final_key = f"PartAng_{v}_{pt1}_{pt2}"
                            
                            # Salva anche i metadata del parziale
                            if 'partial_angle_metadata' not in st.session_state:
                                st.session_state.partial_angle_metadata = {}
                            st.session_state.partial_angle_metadata[final_key] = {'suffix': partial_suffix}
                        else:
                            # Angolo completo
                            final_key = f"Ang_{v}"
                        
                        st.session_state.solved_items.add(final_key)
                        
                        # Salvataggio valore - SEMPRE usa l'angolo interno (< 180° o < 200g)
                        diff_deg = abs(az1 - az2)
                        
                        # Normalizza per ottenere sempre l'angolo INTERNO
                        if diff_deg > 180:
                            diff_deg = 360 - diff_deg
                        
                        # IMPORTANTE: Verifica che l'angolo sia ragionevole per un triangolo/poligono
                        # Un angolo interno deve essere < 180° (200g)
                        # Se dopo il calcolo è ancora > 180°, c'è un errore di orientamento
                        
                        rad_val = math.radians(diff_deg)
                        
                        st.session_state.solved_values[final_key] = format_angle_output(rad_val, st.session_state.input_interpretation)
                        
                        # Etichetta Greca
                        pts_lbls = sorted(st.session_state.points.keys())
                        idx = pts_lbls.index(v)
                        greeks = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta"]
                        greek_label = greeks[idx] if idx < len(greeks) else rf"\alpha_{{{v}}}"
                        
                        # Aggiungi suffisso se è parziale
                        if is_partial and partial_suffix:
                            greek_label += partial_suffix.replace("_", "")  # "_1" → "1"
                        
                        st.session_state.last_calc_msg = rf"{greek_label} = |{az1_disp:.4f}{sym} - {az2_disp:.4f}{sym}| = {diff_disp:.4f}{sym}"
                        
                        # Report Dettagliato (Sottrazione)
                        az1_str = format_angle_output(math.radians(az1), unit, latex=True)
                        az2_str = format_angle_output(math.radians(az2), unit, latex=True)
                        final_res_str = format_angle_output(rad_val, unit, latex=True)
                        
                        latex_msg_sub = rf"""\begin{{aligned}}
{greek_label} &= |({v}{wf['side1']}) - ({v}{wf['side2']})| \\
&= |{az1_str} - {az2_str}| \\
&= \mathbf{{{final_res_str}}}
\end{{aligned}}"""

                        st.session_state.log.append({
                            'action': f"Calcolo {final_key}", 
                            'method': rf"{greek_label} = |({v}{wf['side1']}) - ({v}{wf['side2']})|", 
                            'result': latex_msg_sub,
                            'desc_verbose': "Calcolo angolo interno tramite differenza di azimut."
                        })
                        
                        # Reset
                        st.session_state.az_workflow = {'active': False, 'vertex': None, 'step': 0, 'side1': None, 'side2': None, 'az1_val': None, 'az2_val': None, 'pending_target': None, 'quiz_options': None}
                        st.balloons()
                        st.rerun()
                    else:
                        # Registra l'errore nel log
                        pts_lbls = sorted(st.session_state.points.keys())
                        idx = pts_lbls.index(v)
                        greeks = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta"]
                        greek_label = greeks[idx] if idx < len(greeks) else rf"\alpha_{{{v}}}"
                        
                        st.session_state.log.append({
                            'action': f"Calcolo Ang_{v}",
                            'method': ans,
                            'result': r'\text{ERRORE: Operazione errata}',
                            'desc_verbose': f"Hai scelto l'operazione sbagliata: {ans}. L'angolo interno si calcola con la sottrazione dei due azimut: |Azimut 1 - Azimut 2|",
                            'added_items': [],
                            'added_values': [],
                            'is_error': True
                        })
                        st.error("No. L'angolo compreso è la differenza tra le due direzioni.")

    # Logica Standard (Eseguita solo se az_workflow non è attivo)
    elif not st.session_state.points:
        st.info("Carica dati.")
    elif st.session_state.angle_workflow_target and not st.session_state.current_mission:
        # --- GESTIONE FLUSSO AUTOMATICO AZIMUT ---
        pt = st.session_state.angle_workflow_target
        pts_lbls = sorted(st.session_state.points.keys())
        idx = pts_lbls.index(pt)
        prev = pts_lbls[idx-1]; nex = pts_lbls[(idx+1)%len(pts_lbls)]
        
        # Verifica quali azimut mancano
        az_prev_known = f"SegAz_{prev}_{pt}" in st.session_state.solved_items or f"SegAz_{pt}_{prev}" in st.session_state.solved_items
        az_nex_known = f"SegAz_{pt}_{nex}" in st.session_state.solved_items or f"SegAz_{nex}_{pt}" in st.session_state.solved_items
        
        if az_prev_known and az_nex_known:
            # Entrambi noti -> Calcolo Angolo Finale
            st.session_state.current_mission = f"calc_ang_{pt}"
            st.session_state.ang_method_choice = "Calcolo con Azimut vertici"
            st.session_state.current_options = None
            st.rerun()
        elif not az_prev_known and not az_nex_known:
            # Entrambi mancano (Caso iniziale o reset anomalo) -> Lascia scegliere all'utente nel blocco standard
            st.session_state.angle_workflow_target = None
            st.rerun()
        else:
            # Ne manca uno -> Imposta automaticamente l'altro
            missing_mission = None
            if not az_prev_known: missing_mission = f"seg_az_{pt}_{prev}"
            else: missing_mission = f"seg_az_{pt}_{nex}"
            
            st.session_state.current_mission = missing_mission
            st.session_state.current_options = None
            st.rerun()
    else:
        avail = get_available_targets()
        
        # Se l'utente ha cliccato "SÌ, calcola coordinate", forza la visualizzazione dei punti
        if not avail and st.session_state.get('force_show_coords', False):
            pts_keys = sorted(st.session_state.points.keys())
            avail = []
            for p in pts_keys:
                if f"X_{p}" not in st.session_state.solved_items or f"Y_{p}" not in st.session_state.solved_items:
                    avail.append(f"Punto {p}")
            # Reset del flag
            if avail:
                st.info("📐 **Calcolo Coordinate Cartesiane per gli Azimut**")
                st.caption("Calcola le coordinate (X,Y) dei punti per poter poi calcolare gli azimut dei segmenti")
        
        if not avail: 
            st.success("🎉 Hai completato l'esercizio!")
            
            # SUGGERIMENTO PROATTIVO: Pulsante per calcolare azimut
            pts_keys = sorted(st.session_state.points.keys())
            if len(pts_keys) >= 2:
                # Controlla se ci sono azimut dei segmenti che potrebbero essere calcolati
                potential_azimuths = []
                missing_coords_for_azimuths = []
                
                for p1 in pts_keys:
                    for p2 in pts_keys:
                        if p1 == p2: continue
                        dist_known = f"SegDist_{p1}_{p2}" in st.session_state.solved_items or f"SegDist_{p2}_{p1}" in st.session_state.solved_items
                        az_known = f"SegAz_{p1}_{p2}" in st.session_state.solved_items
                        cart_p1_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items)
                        cart_p2_ok = (f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                        
                        if dist_known and not az_known:
                            potential_azimuths.append(f"{p1}→{p2}")
                            if not (cart_p1_ok and cart_p2_ok):
                                for p in [p1, p2]:
                                    if not (f"X_{p}" in st.session_state.solved_items and f"Y_{p}" in st.session_state.solved_items):
                                        if p not in missing_coords_for_azimuths:
                                            missing_coords_for_azimuths.append(p)
                
                if potential_azimuths and missing_coords_for_azimuths:
                    st.divider()
                    st.write("**💡 Vuoi calcolare anche gli Azimut dei segmenti?**")
                    st.caption(f"Per calcolare gli azimut ({', '.join(potential_azimuths[:3])}) servono le coordinate cartesiane (X,Y)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ SÌ, calcola coordinate cartesiane", use_container_width=True, type="primary"):
                            # Forza il ricaricamento mostrando i punti mancanti
                            # Semplice trick: resettiamo avail così il sistema ricarica la lista
                            st.session_state.force_show_coords = True
                            st.rerun()
                    with col2:
                        if st.button("❌ NO, scarica il report", use_container_width=True):
                            st.session_state.force_show_coords = False
                            st.rerun()
        else:
            # Mostra messaggio informativo se ci sono segmenti non disponibili per mancanza coordinate
            pts_keys = sorted(st.session_state.points.keys())
            if len(pts_keys) >= 2:
                missing_segments = []
                missing_azimuths = []
                
                # Controlla lunghezze mancanti
                for c in list(itertools.combinations(pts_keys, 2)):
                    if f"SegDist_{c[0]}_{c[1]}" not in st.session_state.solved_items and f"SegDist_{c[1]}_{c[0]}" not in st.session_state.solved_items:
                        p1, p2 = c[0], c[1]
                        cart_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items and
                                  f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                        pol_ok = (f"Dist_{p1}" in st.session_state.solved_items and f"Az_{p1}" in st.session_state.solved_items and
                                 f"Dist_{p2}" in st.session_state.solved_items and f"Az_{p2}" in st.session_state.solved_items)
                        if not cart_ok and not pol_ok:
                            missing_segments.append(f"{p1}{p2}")
                
                # Controlla azimut mancanti per mancanza coordinate cartesiane
                # Separa in due categorie: con distanza nota vs senza distanza
                azimuths_need_coords = []  # Hanno distanza ma mancano coord
                azimuths_need_dist = []    # Mancano sia distanza che coord
                
                for p1 in pts_keys:
                    for p2 in pts_keys:
                        if p1 == p2: continue
                        dist_known = f"SegDist_{p1}_{p2}" in st.session_state.solved_items or f"SegDist_{p2}_{p1}" in st.session_state.solved_items
                        az_known = f"SegAz_{p1}_{p2}" in st.session_state.solved_items
                        cart_p1_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items)
                        cart_p2_ok = (f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                        
                        if not az_known:
                            if dist_known and not (cart_p1_ok and cart_p2_ok):
                                # Hanno la distanza, servono solo le coordinate
                                azimuths_need_coords.append(f"{p1}→{p2}")
                            elif not dist_known:
                                # Mancano sia distanza che coordinate (probabile)
                                azimuths_need_dist.append(f"{p1}→{p2}")
                
                # Mostra messaggi informativi con priorità
                if missing_segments:
                    st.info(f"💡 **Step 1 - Lunghezze:** Per calcolare {', '.join(missing_segments)}, calcola prima le coordinate (X,Y) o polari (Distanza, Azimut) dei punti.")
                
                if azimuths_need_coords:
                    st.success(f"✅ **Prossimo Step - Azimut dei segmenti:** Hai già calcolato la distanza {', '.join(azimuths_need_coords)}! Per calcolare gli azimut di questi segmenti, devi ora calcolare le coordinate cartesiane (X,Y) dei punti coinvolti. Gli azimut si calcolano con arctan(ΔX/ΔY).")
                
                if azimuths_need_dist and not azimuths_need_coords:
                    st.info(f"📐 **Per gli azimut:** Prima calcola le distanze {', '.join(azimuths_need_dist[:3])}{'...' if len(azimuths_need_dist) > 3 else ''}, poi le coordinate cartesiane (X,Y).")
            
            sel_subj = st.selectbox("1. Calcola:", avail)
            possible_actions = []
            if "Angolo" in sel_subj:
                # Mostra sempre tutte e 3 le categorie
                possible_actions.append("Calcolo con i Teoremi")
                possible_actions.append("Calcolo con Azimut vertici")
                possible_actions.append("Sottrazione Angolare")
            elif "Punto" in sel_subj: 
                pt = sel_subj.split()[1]
                if f"X_{pt}" not in st.session_state.solved_items: possible_actions.append("Coord. X")
                if f"Y_{pt}" not in st.session_state.solved_items: possible_actions.append("Coord. Y")
                if f"Dist_{pt}" not in st.session_state.solved_items: possible_actions.append("Distanza (da O)")
                if f"Az_{pt}" not in st.session_state.solved_items: possible_actions.append("Azimut (da O)")
            elif "Lunghezza" in sel_subj:
                p1, p2 = sel_subj.split()[1][0], sel_subj.split()[1][1]
                cart_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items and
                           f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                pol_ok = (f"Dist_{p1}" in st.session_state.solved_items and f"Az_{p1}" in st.session_state.solved_items and
                          f"Dist_{p2}" in st.session_state.solved_items and f"Az_{p2}" in st.session_state.solved_items)
                
                # Mostra tutti i metodi disponibili (senza preferenze)
                if pol_ok and cart_ok:
                    # Entrambi disponibili - lascia scegliere
                    possible_actions.extend(["Metodo Diretto (Carnot)", "Metodo Indiretto (Cartesiano)"])
                elif pol_ok:
                    # Solo polari - puoi comunque usare entrambi (convertendo a cartesiane)
                    possible_actions.extend(["Metodo Diretto (Carnot)", "Metodo Indiretto (Cartesiano)"])
                elif cart_ok:
                    # Solo cartesiane - solo Pitagora disponibile
                    possible_actions.append("Metodo Indiretto (Cartesiano)")
            elif "Poligono" in sel_subj: 
                pts_keys = sorted(st.session_state.points.keys())
                known_angles = [x for x in st.session_state.solved_items if x.startswith("Ang_")]
                
                perimeter_complete = True
                for i in range(len(pts_keys)):
                    p1, p2 = pts_keys[i], pts_keys[(i+1)%len(pts_keys)]
                    if f"SegDist_{p1}_{p2}" not in st.session_state.solved_items and f"SegDist_{p2}_{p1}" not in st.session_state.solved_items:
                        perimeter_complete = False; break
                
                if "Area_Poly" not in st.session_state.solved_items and len(known_angles) == len(pts_keys): possible_actions.append("Area")
                if "Perim_Poly" not in st.session_state.solved_items and perimeter_complete: possible_actions.append("Perimetro")
            else: possible_actions = ["Calcola"]
            
            if len(possible_actions) == 0: sel_act = None
            elif len(possible_actions) == 1: sel_act = possible_actions[0]; st.info(f"📍 Azione unica disponibile: **{sel_act}**")
            else: sel_act = st.radio("2. Scegli metodo:", possible_actions, horizontal=True, key=f"radio_{len(st.session_state.log)}")
            
            if st.button("Procedi"):
                if sel_act:
                    code = None
                    # Special handling for the new angle flow
                    if "Angolo" in sel_subj:
                        # Estrai il vertice (ultima parola dopo "in")
                        if " in " in sel_subj:
                            pt = sel_subj.split(" in ")[-1]
                        else:
                            pt = sel_subj.split(" ")[-1]
                        
                        # Controlla se è un angolo parziale e memorizza quale
                        if "parziale" in sel_subj:
                            # Estrai il suffisso (₁ o ₂) dal simbolo greco
                            if "₁" in sel_subj:
                                st.session_state.selected_partial_suffix = "_1"
                            elif "₂" in sel_subj:
                                st.session_state.selected_partial_suffix = "_2"
                        else:
                            st.session_state.selected_partial_suffix = None
                        
                        code = f"calc_ang_{pt}"
                        # Store the category choice. Reusing ang_method_choice for the category.
                        st.session_state.ang_method_choice = sel_act 
                    elif "Lunghezza" in sel_subj:
                        p1, p2 = sel_subj.split()[1][0], sel_subj.split()[1][1]
                        
                        # Non salvare più la preferenza - lascia sempre scegliere
                        
                        if sel_act == "Metodo Indiretto (Cartesiano)":
                            cart_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items and
                                       f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                            if not cart_ok:
                                # Attiva proiezioni se non visibili
                                if not st.session_state.projections_visible:
                                    activate_projections()
                                
                                # Trova tutte le coordinate mancanti
                                missing = []
                                for p in [p1, p2]:
                                    if f"X_{p}" not in st.session_state.solved_items:
                                        missing.append(f"X_{p}")
                                    if f"Y_{p}" not in st.session_state.solved_items:
                                        missing.append(f"Y_{p}")
                                
                                if missing:
                                    st.warning(f"⚠️ **Prerequisito per Metodo Cartesiano:**")
                                    st.info(f"Per calcolare la distanza {p1}-{p2} con Pitagora, servono le coordinate cartesiane di entrambi i punti.")
                                    st.write(f"**Coordinate mancanti:** {', '.join(missing)}")
                                    st.write("💡 **Suggerimento:** Calcola prima tutte le coordinate mancanti, poi torna a calcolare la distanza.")
                                    
                                    # Mostra bottone per calcolare la prima coordinata mancante
                                    first_missing = missing[0]
                                    coord_type = "X" if "X_" in first_missing else "Y"
                                    point = first_missing.split("_")[1]
                                    
                                    if st.button(f"✅ Calcola {coord_type} di {point}"):
                                        st.session_state.current_mission = f"calc_{coord_type}_{point}"
                                        st.session_state.current_options = None
                                        st.rerun()
                                    
                                    # Non procedere con il calcolo della distanza
                                    code = None
                                else:
                                    code = f"seg_dist_{p1}_{p2}"
                            else:
                                code = f"seg_dist_{p1}_{p2}"
                        else:
                            # Metodo Diretto (Carnot) - non serve controllo prerequisiti
                            code = f"seg_dist_{p1}_{p2}"
                        
                        if code:
                            st.session_state.ang_method_choice = sel_act
                    else:
                        ok, msg, code_from_func = check_goal_feasibility(sel_subj, sel_act)
                        if ok:
                            code = code_from_func
                            st.session_state.specific_method_choice = None
                            # Attiva automaticamente le proiezioni se si calcolano coordinate cartesiane
                            if code and ("calc_X" in code or "calc_Y" in code) and not st.session_state.projections_visible:
                                activate_projections()
                        # Clear the choice for non-angle calculations
                        st.session_state.ang_method_choice = None

                    if code:
                        st.session_state.current_mission = code
                        st.session_state.current_options = None 
                        st.session_state.az_segments_confirmed = False
                        st.session_state.specific_method_choice = None # CLEAR IT
                        st.rerun()

            if st.session_state.current_mission:
                st.divider()
                
                # --- HEADER MISSIONE ---
                q_desc = get_strategies_for_mission(st.session_state.current_mission).get('desc', 'Missione')
                st.markdown(f"### 🎯 Obiettivo: {q_desc}")
                
                method_filter = None
                q = {}
                num_steps_offset = 0
                show_strategy_form = True

                if "seg_dist" in st.session_state.current_mission:
                    q_check = get_strategies_for_mission(st.session_state.current_mission)
                    all_methods = q_check.get("available_methods", [])
                    category = st.session_state.get('ang_method_choice')
                    
                    # Verifica che la categoria scelta sia disponibile
                    if category and category in all_methods:
                        method_filter = category
                        # Applica subito il filtro
                        q = get_strategies_for_mission(st.session_state.current_mission, method_filter)
                    else:
                        # Nessun filtro o filtro non valido - usa il primo metodo disponibile
                        q = get_strategies_for_mission(st.session_state.current_mission)

                elif "calc_ang" in st.session_state.current_mission:
                    q_check = get_strategies_for_mission(st.session_state.current_mission)
                    all_methods = q_check.get("available_methods", [])
                    category = st.session_state.get('ang_method_choice')

                    methods_in_category = []
                    if category == "Calcolo con i Teoremi":
                        # Include Carnot, Seni, (ma NON Sottrazione angolare che è una categoria a sé)
                        methods_in_category = [m for m in all_methods if any(x in m for x in ['Carnot', 'Seni', 'Somma'])]
                    elif category == "Calcolo con Azimut vertici":
                        methods_in_category = [m for m in all_methods if m == 'Azimut']
                    elif category == "Sottrazione Angolare":
                        # Solo il metodo Sottrazione
                        methods_in_category = [m for m in all_methods if 'Sottrazione' in m]
                    
                    if not methods_in_category:
                        if category == "Calcolo con Azimut vertici":
                            pt = st.session_state.current_mission.split("_")[2]
                            pts_keys = sorted(st.session_state.points.keys())
                            idx = pts_keys.index(pt)
                            
                            # Determina i lati dell'angolo
                            # Se è un angolo parziale, usa i vertici del triangolo specifico (con diagonale)
                            # Altrimenti usa prev/next del poligono completo
                            
                            is_partial = st.session_state.get('selected_partial_suffix') is not None
                            
                            if is_partial and len(pts_keys) == 4:
                                # Quadrilatero con angolo parziale - trova la diagonale e il triangolo
                                suffix = st.session_state.selected_partial_suffix  # "_1" o "_2"
                                
                                # Trova la diagonale esistente
                                diagonal_found = False
                                for p1_lbl, p3_lbl in itertools.combinations(pts_keys, 2):
                                    idx1 = pts_keys.index(p1_lbl)
                                    idx3 = pts_keys.index(p3_lbl)
                                    
                                    # È una diagonale? (non adiacenti)
                                    if abs(idx1 - idx3) % (len(pts_keys)-1) > 1:
                                        d_key1 = f"SegDist_{p1_lbl}_{p3_lbl}"
                                        d_key2 = f"SegDist_{p3_lbl}_{p1_lbl}"
                                        
                                        if d_key1 in st.session_state.solved_items or d_key2 in st.session_state.solved_items:
                                            # Trovata la diagonale tra p1_lbl e p3_lbl
                                            other_pts = [p for p in pts_keys if p not in [p1_lbl, p3_lbl]]
                                            p2_lbl, p4_lbl = other_pts[0], other_pts[1]
                                            
                                            # Triangolo "1": (p1, p2, p3) 
                                            # Triangolo "2": (p1, p4, p3)
                                            triangle_1_set = {p1_lbl, p2_lbl, p3_lbl}
                                            triangle_2_set = {p1_lbl, p4_lbl, p3_lbl}
                                            
                                            # Determina quale triangolo contiene pt con suffisso corretto
                                            if suffix == "_1" and pt in triangle_1_set:
                                                # Triangolo 1
                                                triangle_verts = sorted([v for v in triangle_1_set if v != pt])
                                                prev_pt = triangle_verts[0]
                                                next_pt = triangle_verts[1]
                                                diagonal_found = True
                                                break
                                            elif suffix == "_2" and pt in triangle_2_set:
                                                # Triangolo 2
                                                triangle_verts = sorted([v for v in triangle_2_set if v != pt])
                                                prev_pt = triangle_verts[0]
                                                next_pt = triangle_verts[1]
                                                diagonal_found = True
                                                break
                                
                                if not diagonal_found:
                                    # Fallback: usa prev/next standard
                                    prev_pt = pts_keys[idx-1]
                                    next_pt = pts_keys[(idx+1)%len(pts_keys)]
                            else:
                                # Angolo completo o triangolo semplice: usa prev/next del poligono
                                prev_pt = pts_keys[idx-1]
                                next_pt = pts_keys[(idx+1)%len(pts_keys)]
                            
                            # Verifica che i punti adiacenti abbiano le coordinate cartesiane
                            # (servono per calcolare gli azimut dei segmenti)
                            missing_coords = []
                            for p in [pt, prev_pt, next_pt]:
                                if f"X_{p}" not in st.session_state.solved_items or f"Y_{p}" not in st.session_state.solved_items:
                                    missing_coords.append(p)
                            
                            if missing_coords:
                                st.error(f"⛔ **Prerequisito mancante per il Metodo Azimut!**")
                                st.warning(f"Per calcolare l'angolo in {pt} con il metodo degli azimut, devi PRIMA calcolare le coordinate cartesiane (X,Y) dei punti coinvolti nell'angolo.")
                                st.info(f"🎯 **Punti che richiedono coordinate cartesiane:** {', '.join(missing_coords)}")
                                st.write(f"**Perché?** Il metodo richiede gli azimut dei lati ({pt}→{prev_pt}) e ({pt}→{next_pt}), che si calcolano con arctan(ΔX/ΔY).")
                                st.write("💡 **Alternativa:** Puoi calcolare prima le coordinate cartesiane, oppure scegliere 'Calcolo con i Teoremi' che può usare le coordinate polari!")
                                
                                if st.button("❌ Annulla e scegli altro metodo"):
                                    st.session_state.current_mission = None
                                    st.rerun()
                                
                                show_strategy_form = False
                            else:
                                st.info("🌟 **Modalità Interattiva - Metodo Azimut**")
                                if is_partial:
                                    st.write(f"Per questo angolo parziale useremo il metodo degli azimut con i vertici del triangolo: **{pt}, {prev_pt}, {next_pt}**")
                                else:
                                    st.write("Per questo calcolo useremo il metodo visuale passo-passo degli azimut.")
                                
                                # ATTIVAZIONE NUOVO WORKFLOW
                                if st.button("✅ Avvia Procedura Guidata"):
                                    st.session_state.az_workflow['active'] = True
                                    st.session_state.az_workflow['vertex'] = pt
                                    st.session_state.az_workflow['step'] = 1
                                    # Salva il suffisso parziale nel workflow per preservarlo
                                    st.session_state.az_workflow['partial_suffix'] = st.session_state.get('selected_partial_suffix')
                                    # Salva i punti adiacenti per limitare la selezione ai lati corretti
                                    st.session_state.az_workflow['valid_neighbors'] = [prev_pt, next_pt]
                                    st.session_state.current_mission = None # Pulisce la missione vecchia
                                    st.rerun()
                                    
                                show_strategy_form = False
                        else:
                            # Messaggio specifico per Sottrazione Angolare
                            if category == "Sottrazione Angolare":
                                st.error("⛔ **Sottrazione Angolare non disponibile!**")
                                st.warning(f"Per usare la Sottrazione Angolare, devi PRIMA calcolare gli altri 2 angoli del triangolo corrente.")
                                st.info("💡 **Suggerimento:** Calcola prima 2 angoli usando 'Calcolo con i Teoremi' o 'Calcolo con Azimut vertici', poi potrai usare la Sottrazione Angolare per il terzo.")
                                
                                if st.button("❌ Annulla e scegli altro metodo"):
                                    st.session_state.current_mission = None
                                    st.rerun()
                            else:
                                st.warning(f"Nessun metodo di tipo '{category}' è attualmente applicabile per questo angolo.")
                                st.session_state.current_mission = None; st.rerun()
                    elif len(methods_in_category) == 1:
                        method_filter = methods_in_category[0]
                        st.session_state.specific_method_choice = method_filter
                        st.caption(f"Metodo Selezionato: **{method_filter}**")
                        
                        if category == "Calcolo con Azimut vertici":
                             pass
                    else:
                        # Mostra label appropriato in base alla categoria
                        if category == "Calcolo con i Teoremi":
                            st.write("3. Scegli il Teorema specifico:")
                            label_text = "Teorema:"
                        elif category == "Sottrazione Angolare":
                            st.write("3. Scegli la formula corretta:")
                            label_text = "Formula:"
                        else:
                            st.write("3. Scegli il metodo:")
                            label_text = "Metodo:"
                        
                        # Aggiungi key univoco basato sulla missione per forzare reset
                        radio_key = f"theorem_choice_{st.session_state.current_mission}"
                        
                        # Mostra il valore corrente se esiste
                        current_choice = st.session_state.get('specific_method_choice')
                        if current_choice and current_choice in methods_in_category:
                            default_index = methods_in_category.index(current_choice)
                        else:
                            default_index = 0
                        
                        method_filter = st.radio(
                            label_text, 
                            methods_in_category, 
                            index=default_index,
                            horizontal=True, 
                            label_visibility="collapsed", 
                            key=radio_key
                        )
                        
                        # Se la selezione è cambiata, aggiorna e forza rerun
                        if method_filter != st.session_state.get('specific_method_choice'):
                            st.session_state.specific_method_choice = method_filter
                            st.session_state.current_options = None  # Reset opzioni per rigenerare
                            st.rerun()
                        
                        num_steps_offset = 1
                    
                    # Assicuriamoci che method_filter sia impostato dalla session_state se non lo è già
                    if not method_filter:
                        method_filter = st.session_state.get('specific_method_choice')
                    
                    if method_filter:
                        q = get_strategies_for_mission(st.session_state.current_mission, method_filter)
                    else:
                        q = get_strategies_for_mission(st.session_state.current_mission)
                else:
                    q = get_strategies_for_mission(st.session_state.current_mission)
                
                # Salva q in session_state per riuso
                st.session_state.current_question_data = q
                
                if show_strategy_form and not q:
                    if st.session_state.current_mission:
                        st.error("Errore interno: Strategia non trovata.")
                        st.session_state.current_mission = None
                        st.rerun()

                if st.session_state.current_mission and show_strategy_form: # Check if mission was cancelled or form hidden
                    if not st.session_state.current_options:
                        corrects = q.get('correct_list', [q.get('correct')])
                        if not corrects or corrects[0] is None:
                            st.warning("Nessuna strategia valida trovata per il metodo selezionato.")
                            st.session_state.current_mission = None
                            st.rerun()
                        else:
                            import random
                            wrongs = q.get('wrongs', [])
                            
                            # Rimuovi duplicati dalle opzioni corrette e sbagliate
                            corrects = list(set(corrects))
                            wrongs = list(set(wrongs))
                            
                            # Rimuovi dalle sbagliate quelle che sono già nelle corrette
                            wrongs = [w for w in wrongs if w not in corrects]
                            
                            # Assicurati di avere sempre esattamente 3 opzioni DIVERSE
                            all_opts = []
                            
                            # Aggiungi le corrette
                            all_opts.extend(corrects)
                            
                            # Aggiungi opzioni sbagliate fino ad avere 3 opzioni totali
                            needed = 3 - len(all_opts)
                            if needed > 0 and wrongs:
                                # Prendi al massimo 'needed' opzioni sbagliate casuali
                                available_wrongs = min(len(wrongs), needed)
                                all_opts.extend(random.sample(wrongs, available_wrongs))
                            
                            # Se ancora non abbiamo 3 opzioni diverse, crea opzioni generiche
                            generic_wrongs = [
                                "Formula non applicabile in questo caso",
                                "Metodo alternativo non corretto",
                                "Approccio errato per questo tipo di problema"
                            ]
                            idx = 0
                            while len(all_opts) < 3:
                                if generic_wrongs[idx] not in all_opts:
                                    all_opts.append(generic_wrongs[idx])
                                idx = (idx + 1) % len(generic_wrongs)
                            
                            # Verifica che siano tutte diverse (rimozione duplicati finali)
                            all_opts = list(dict.fromkeys(all_opts))  # Mantiene l'ordine e rimuove duplicati
                            
                            # Prendi solo le prime 3 opzioni
                            all_opts = all_opts[:3]
                            
                            # Mescola le opzioni
                            random.shuffle(all_opts)
                            
                            st.session_state.current_options = all_opts
                    
                    with st.form("strat"):
                        st.write(f"{3 + num_steps_offset}. Strategia:")
                        ans = st.radio("Formula:", st.session_state.current_options)
                        if st.form_submit_button("Calcola"):
                            # Rigenera q al momento del calcolo per evitare problemi di stato
                            # Questo garantisce che q contenga il meta corretto
                            # Per seg_dist usa ang_method_choice, per calc_ang usa specific_method_choice
                            if "seg_dist" in st.session_state.current_mission:
                                method_filter_on_calc = st.session_state.get('ang_method_choice')
                            else:
                                method_filter_on_calc = st.session_state.get('specific_method_choice')
                            
                            q = get_strategies_for_mission(st.session_state.current_mission, method_filter_on_calc)
                            
                            corrects = q.get('correct_list', [q['correct']])
                            
                            # Confronto robusto: controlla sia la stringa esatta che la stringa normalizzata
                            def normalize_formula(s):
                                """Rimuove spazi extra e simboli $ per confronto robusto"""
                                if not s:
                                    return ""
                                # Rimuovi $ all'inizio e alla fine
                                s = s.strip()
                                if s.startswith('$') and s.endswith('$'):
                                    s = s[1:-1]
                                # Rimuovi spazi multipli
                                import re
                                s = re.sub(r'\s+', ' ', s)
                                return s.strip()
                            
                            ans_normalized = normalize_formula(ans)
                            corrects_normalized = [normalize_formula(c) for c in corrects]
                            
                            # Verifica se la risposta è corretta (confronto esatto o normalizzato)
                            is_correct = ans in corrects or ans_normalized in corrects_normalized
                            
                            # DEBUG: Se la risposta è sbagliata, mostra cosa è stato confrontato
                            if not is_correct:
                                # Mostra debug solo per sviluppo - rimuovere in produzione
                                with st.expander("🔍 Debug - Confronto Formula"):
                                    st.write("**Risposta selezionata:**", ans)
                                    st.write("**Risposta normalizzata:**", ans_normalized)
                                    st.write("**Risposte corrette:**", corrects)
                                    st.write("**Risposte corrette normalizzate:**", corrects_normalized)
                            
                            if is_correct:
                                # Snapshot per Undo
                                pre_items = st.session_state.solved_items.copy()
                                pre_values = set(st.session_state.solved_values.keys())
                                # --- MOTORE DI CALCOLO COMPLETO ---
                                parts = st.session_state.current_mission.split("_")
                                act = parts[0]+"_"+parts[1]
                                
                                
                                unit = st.session_state.input_interpretation
                                latex_msg = "Calcolo OK" 
                                descr_text = "Calcolo standard."

                                if act=="calc_X":
                                    pt_lbl = parts[2]; pt = st.session_state.points[pt_lbl]
                                    val = pt['x']; k=f"X_{pt_lbl}"
                                    st.session_state.solved_items.add(k); st.session_state.solved_values[k] = format_coord(val)
                                    ang_fmt = format_angle_output(pt['alpha'], unit, latex=True)
                                    latex_msg = rf"X_{{{pt_lbl}}} = {pt['r']:.2f} \cdot \sin({ang_fmt}) \quad = \quad \mathbf{{{format_coord(val)}}}"
                                    descr_text = f"Proiezione del vettore $O{pt_lbl}$ sull'asse delle ascisse (Est)."
                                elif act=="calc_Y":
                                    pt_lbl = parts[2]; pt = st.session_state.points[pt_lbl]
                                    val = pt['y']; k=f"Y_{pt_lbl}"
                                    st.session_state.solved_items.add(k); st.session_state.solved_values[k] = format_coord(val)
                                    ang_fmt = format_angle_output(pt['alpha'], unit, latex=True)
                                    latex_msg = rf"Y_{{{pt_lbl}}} = {pt['r']:.2f} \cdot \cos({ang_fmt}) \quad = \quad \mathbf{{{format_coord(val)}}}"
                                    descr_text = f"Proiezione del vettore $O{pt_lbl}$ sull'asse delle ordinate (Nord)."
                                elif act=="calc_dist":
                                    pt_lbl = parts[2]; pt = st.session_state.points[pt_lbl]
                                    val = pt['r']; k=f"Dist_{pt_lbl}"
                                    st.session_state.solved_items.add(k); st.session_state.solved_values[k] = f"{val:.4f}"
                                    latex_msg = rf"\overline{{O{pt_lbl}}} = \sqrt{{{format_coord(pt['x'])}^2 + {format_coord(pt['y'])}^2}} \quad = \quad \mathbf{{{val:.4f}}}"
                                    descr_text = f"Calcolo della distanza dall'origine (modulo)."
                                elif act=="calc_az":
                                    pt_lbl = parts[2]; pt = st.session_state.points[pt_lbl]
                                    x, y = pt['x'], pt['y']
                                    
                                    # Logica Quadranti e Correzione
                                    raw_atan = math.atan(x/y) if abs(y) > 1e-9 else (math.pi/2 if x > 0 else -math.pi/2)
                                    raw_atan_fmt = format_angle_output(raw_atan, unit, latex=True)
                                    
                                    quad_name, corr_str = "?", ""
                                    if x >= 0 and y >= 0: quad_name, corr_str = r"I^{\circ} \quad (X+, Y+)", ""
                                    elif x >= 0 and y < 0: quad_name, corr_str = r"II^{\circ} \quad (X+, Y-)", (r"+ 180^{\circ}" if unit == AngleUnit.DEG else (r"+ 200^g" if unit == AngleUnit.GON else r"+ \pi"))
                                    elif x < 0 and y < 0: quad_name, corr_str = r"III^{\circ} \quad (X-, Y-)", (r"+ 180^{\circ}" if unit == AngleUnit.DEG else (r"+ 200^g" if unit == AngleUnit.GON else r"+ \pi"))
                                    else: quad_name, corr_str = r"IV^{\circ} \quad (X-, Y+)", (r"+ 360^{\circ}" if unit == AngleUnit.DEG else (r"+ 400^g" if unit == AngleUnit.GON else r"+ 2\pi"))

                                    val = pt['alpha']; k=f"Az_{pt_lbl}"
                                    st.session_state.solved_items.add(k); st.session_state.solved_values[k] = format_angle_output(val, unit)
                                    latex_msg = rf"(O{pt_lbl}) = \arctan\left(\frac{{{format_coord(pt['x'])}}}{{{format_coord(pt['y'])}}}\right) \quad = \quad \mathbf{{{format_angle_output(val, unit, latex=True)}}}"
                                    descr_text = f"Calcolo dell'azimut rispetto all'origine."
                                    latex_msg = rf"""\begin{{aligned}}
\text{{Quadrante}} &: {quad_name} \\
\alpha_{{calc}} &= \arctan(X/Y) = {raw_atan_fmt} \\
(O{pt_lbl}) &= {raw_atan_fmt} \quad \mathbf{{{corr_str}}} \\
&= \mathbf{{{format_angle_output(val, unit, latex=True)}}}
\end{{aligned}}"""
                                    descr_text = f"Calcolo Azimut O-{pt_lbl} con correzione quadrante."
                                elif act=="seg_dist":
                                    p1, p2 = parts[2], parts[3]
                                    
                                    if "Carnot" in ans:
                                        # Metodo Diretto
                                        r1 = st.session_state.points[p1]['r']
                                        r2 = st.session_state.points[p2]['r']
                                        a1 = st.session_state.points[p1]['alpha']
                                        a2 = st.session_state.points[p2]['alpha']
                                        diff_alpha = a2 - a1
                                        val = math.sqrt(r1**2 + r2**2 - 2*r1*r2*math.cos(diff_alpha))
                                        
                                        k = f"SegDist_{p1}_{p2}"
                                        st.session_state.solved_items.add(f"Seg_{p1}_{p2}"); st.session_state.solved_items.add(k); st.session_state.solved_values[k] = f"{val:.4f}"
                                        
                                        a1_fmt = format_angle_output(a1, unit, latex=True)
                                        a2_fmt = format_angle_output(a2, unit, latex=True)
                                        latex_msg = rf"""\begin{{aligned}}
\overline{{O{p1}}} &= {r1:.4f}, \quad \overline{{O{p2}}} = {r2:.4f} \\
\Delta \theta &= \theta_{{{p2}}} - \theta_{{{p1}}} = {a2_fmt} - {a1_fmt} \\
\overline{{{p1}{p2}}} &= \sqrt{{ {r1:.2f}^2 + {r2:.2f}^2 - 2 \cdot {r1:.2f} \cdot {r2:.2f} \cdot \cos(\Delta \theta) }} \\
\overline{{{p1}{p2}}} &= \mathbf{{{val:.4f}}}
\end{{aligned}}"""
                                        descr_text = f"Calcolo distanza {p1}-{p2} con Teorema di Carnot (Metodo Diretto)."
                                    else:
                                        # Metodo Indiretto (Cartesiano)
                                        x1, y1 = st.session_state.points[p1]['x'], st.session_state.points[p1]['y']
                                        x2, y2 = st.session_state.points[p2]['x'], st.session_state.points[p2]['y']
                                        dx_val = x2 - x1; dy_val = y2 - y1
                                        val = math.sqrt(dx_val**2 + dy_val**2)
                                        k = f"SegDist_{p1}_{p2}"
                                        st.session_state.solved_items.add(f"Seg_{p1}_{p2}"); st.session_state.solved_items.add(k); st.session_state.solved_values[k] = f"{val:.4f}"
                                        latex_msg = rf"""\begin{{aligned}}
\Delta X &= {format_coord(x2)} - {format_coord(x1)} = \mathbf{{{dx_val:.4f}}} \\
\Delta Y &= {format_coord(y2)} - {format_coord(y1)} = \mathbf{{{dy_val:.4f}}} \\
\overline{{{p1}{p2}}} &= \sqrt{{\Delta X^2 + \Delta Y^2}} \\
&= \sqrt{{({dx_val:.4f})^2 + ({dy_val:.4f})^2}} \\
&= \mathbf{{{val:.4f}}}
\end{{aligned}}"""
                                    descr_text = f"Calcolo della distanza tra {p1} e {p2} applicando il teorema di Pitagora alle differenze di coordinate."
                                elif act=="seg_az":
                                    p1, p2 = parts[2], parts[3]
                                    x1, y1 = st.session_state.points[p1]['x'], st.session_state.points[p1]['y']
                                    x2, y2 = st.session_state.points[p2]['x'], st.session_state.points[p2]['y']
                                    dx_val = x2 - x1; dy_val = y2 - y1
                                    
                                    # Logica Quadranti e Correzione
                                    raw_atan = math.atan(dx_val/dy_val) if abs(dy_val) > 1e-9 else (math.pi/2 if dx_val > 0 else -math.pi/2)
                                    raw_atan_fmt = format_angle_output(raw_atan, unit, latex=True)
                                    
                                    quad_name, corr_str = "?", ""
                                    if dx_val >= 0 and dy_val >= 0:
                                        quad_name = r"I^{\circ} \quad (\Delta X+, \Delta Y+)"
                                        corr_str = "" # Nessuna correzione
                                    elif dx_val >= 0 and dy_val < 0:
                                        quad_name = r"II^{\circ} \quad (\Delta X+, \Delta Y-)"
                                        corr_str = r"+ 180^{\circ}" if unit == AngleUnit.DEG else (r"+ 200^g" if unit == AngleUnit.GON else r"+ \pi")
                                    elif dx_val < 0 and dy_val < 0:
                                        quad_name = r"III^{\circ} \quad (\Delta X-, \Delta Y-)"
                                        corr_str = r"+ 180^{\circ}" if unit == AngleUnit.DEG else (r"+ 200^g" if unit == AngleUnit.GON else r"+ \pi")
                                    else: # dx < 0, dy > 0
                                        quad_name = r"IV^{\circ} \quad (\Delta X-, \Delta Y+)"
                                        corr_str = r"+ 360^{\circ}" if unit == AngleUnit.DEG else (r"+ 400^g" if unit == AngleUnit.GON else r"+ 2\pi")

                                    az = (math.atan2(dx_val, dy_val)+2*math.pi)%(2*math.pi)
                                    k = f"SegAz_{p1}_{p2}"
                                    st.session_state.solved_items.add(k); st.session_state.solved_values[k] = format_angle_output(az, unit)
                                    
                                    latex_msg = rf"""\begin{{aligned}}
\Delta X &= {format_coord(x2)} - {format_coord(x1)} = \mathbf{{{format_coord(dx_val)}}} \\
\Delta Y &= {format_coord(y2)} - {format_coord(y1)} = \mathbf{{{format_coord(dy_val)}}} \\
\text{{Quadrante}} &: {quad_name} \\
\alpha_{{calc}} &= \arctan\left(\frac{{\Delta X}}{{\Delta Y}}\right) = {raw_atan_fmt} \\
({p1}{p2}) &= \arctan\left(\frac{{\Delta X}}{{\Delta Y}}\right) \\
&= {raw_atan_fmt} \quad {corr_str} \\
({p1}{p2}) &= {raw_atan_fmt} \quad \mathbf{{{corr_str}}} \\
&= \mathbf{{{format_angle_output(az, unit, latex=True)}}}
\end{{aligned}}"""
                                    descr_text = f"Calcolo Azimut {p1}-{p2} con analisi del quadrante e correzione."
                                elif act=="calc_poly":
                                    pts_lbls = sorted(st.session_state.points.keys())
                                    if parts[2] == "area":
                                        area = 0.0
                                        x = [st.session_state.points[l]['x'] for l in pts_lbls]
                                        y = [st.session_state.points[l]['y'] for l in pts_lbls]
                                        n = len(pts_lbls)
                                        for i in range(n): area += x[i] * (y[(i+1)%n] - y[(i-1)%n])
                                        area = abs(area)/2.0
                                        st.session_state.solved_items.add("Area_Poly"); st.session_state.solved_values["Area"] = f"{area:.2f}"
                                        latex_msg = rf"\text{{Area}} = \mathbf{{{area:.2f}}}\, m^2"
                                        descr_text = "Calcolo superficie con formula di Gauss."
                                    elif parts[2] == "perim":
                                        perim = 0.0
                                        for i in range(len(pts_lbls)):
                                            p1 = st.session_state.points[pts_lbls[i]]
                                            p2 = st.session_state.points[pts_lbls[(i+1)%len(pts_lbls)]]
                                            perim += math.sqrt((p2['x']-p1['x'])**2 + (p2['y']-p1['y'])**2)
                                        st.session_state.solved_items.add("Perim_Poly"); st.session_state.solved_values["Perimetro"] = f"{perim:.2f}"
                                        latex_msg = rf"\text{{Perimetro}} = \mathbf{{{perim:.2f}}}\, m"
                                        descr_text = "Sommatoria lati."
                                elif act=="calc_ang":
                                    # DEBUG IMMEDIATO
                                    
                                    pt = parts[2]
                                    # NON aggiungere Ang_{pt} qui! Viene aggiunto solo per angoli completi o sum_partials
                                    pts_lbls = sorted(st.session_state.points.keys())
                                    # La rimozione dei parziali viene gestita in sum_partials
                                    idx = pts_lbls.index(pt)
                                    prev_lbl = pts_lbls[idx-1]
                                    nex_lbl = pts_lbls[(idx+1)%len(pts_lbls)]
                                    prev = st.session_state.points[prev_lbl]
                                    curr = st.session_state.points[pt]
                                    nex = st.session_state.points[nex_lbl]
                                    
                                    greeks_latex = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta"]
                                    greek_symbol = greeks_latex[idx] if idx < len(greeks_latex) else f"\\alpha_{{{pt}}}"

                                    # DEBUG PRIMA DI TUTTO

                                    # --- GESTIONE METODI AVANZATI (META) ---
                                    if q.get('meta'):
                                        meta = q['meta']
                                        m_type = meta['type']
                                        
                                        # DEBUG CRITICO
                                        
                                        if m_type == "carnot_tri":
                                            pA, pB = meta['pA'], meta['pB']
                                            s_adj1 = float(st.session_state.solved_values[f"SegDist_{pt}_{pA}" if f"SegDist_{pt}_{pA}" in st.session_state.solved_values else f"SegDist_{pA}_{pt}"])
                                            s_adj2 = float(st.session_state.solved_values[f"SegDist_{pt}_{pB}" if f"SegDist_{pt}_{pB}" in st.session_state.solved_values else f"SegDist_{pB}_{pt}"])
                                            s_opp = float(st.session_state.solved_values[f"SegDist_{pA}_{pB}" if f"SegDist_{pA}_{pB}" in st.session_state.solved_values else f"SegDist_{pB}_{pA}"])
                                            
                                            cos_val = (s_adj1**2 + s_adj2**2 - s_opp**2) / (2 * s_adj1 * s_adj2)
                                            cos_val = max(-1, min(1, cos_val))
                                            alpha_rad = math.acos(cos_val)
                                            
                                            # Salva come parziale
                                            k_part = f"PartAng_{pt}_{min(pA, pB)}_{max(pA, pB)}"
                                            st.session_state.solved_items.add(k_part)
                                            st.session_state.solved_values[k_part] = format_angle_output(alpha_rad, unit)
                                            
                                            # Se suffix è vuoto E siamo in un triangolo (non quadrilatero),
                                            # significa che questo angolo copre l'intero vertice
                                            # Quindi è anche l'angolo completo del poligono
                                            if (not meta.get('sub') or meta['sub'] == '') and len(pts_lbls) == 3:
                                                st.session_state.solved_items.add(f"Ang_{pt}")
                                                st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(alpha_rad, unit)
                                            
                                            # Salva metadata per visualizzazione corretta
                                            st.session_state.partial_angle_metadata[k_part] = {
                                                'suffix': meta['sub'],
                                                'pA': pA,
                                                'pB': pB
                                            }
                                            
                                            # Se l'angolo calcolato non è parziale, salvalo anche come angolo completo del vertice
                                            if not meta.get('sub'):
                                                st.session_state.solved_items.add(f"Ang_{pt}")
                                                st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(alpha_rad, unit)
                                            
                                            sub_sym = greek_symbol + meta['sub']
                                            latex_msg = rf"{sub_sym} = \arccos({cos_val:.4f}) = \mathbf{{{format_angle_output(alpha_rad, unit, latex=True)}}}"
                                            descr_text = f"Calcolo angolo parziale {sub_sym} con Carnot."

                                        elif m_type == "sines_tri":
                                            # Implementazione Sines Parziale
                                            pA, pB, other = meta['pA'], meta['pB'], meta['other']
                                            
                                            # L'angolo di 'other' NEL TRIANGOLO CORRENTE (pt, pA, pB)
                                            triangle_vertices = {pt, pA, pB}
                                            other_vertices = [v for v in triangle_vertices if v != other]
                                            
                                            # Costruisci la chiave dell'angolo nel triangolo corrente
                                            k_other_part = f"PartAng_{other}_{min(other_vertices)}_{max(other_vertices)}"
                                            ang_val_str = st.session_state.solved_values.get(k_other_part)
                                            
                                            # Se non trovato con questa chiave esatta, cerca con ricerca robusta
                                            if not ang_val_str:
                                                # Cerca qualsiasi angolo parziale di 'other' nel triangolo corrente
                                                for k, v in st.session_state.solved_values.items():
                                                    if k.startswith(f"PartAng_{other}_"):
                                                        parts_k = k.split("_")
                                                        if len(parts_k) >= 4:
                                                            v1, v2 = parts_k[2], parts_k[3]
                                                            if v1 in triangle_vertices and v2 in triangle_vertices:
                                                                ang_val_str = v
                                                                k_other_part = k
                                                                break
                                            
                                            # Fallback: angolo completo (solo per triangoli)
                                            if not ang_val_str and len(pts_lbls) == 3:
                                                ang_val_str = st.session_state.solved_values.get(f"Ang_{other}")
                                                if ang_val_str:
                                                    k_other_part = f"Ang_{other}"
                                            
                                            if ang_val_str:
                                                ang_val_rad = parse_angle(ang_val_str.split()[0], unit)
                                                
                                                # Teorema dei Seni: sin(A)/a = sin(B)/b
                                                # sin(A) = a * sin(B) / b
                                                # dove A è l'angolo in pt, B è l'angolo in other
                                                # a = lato opposto a pt = segmento tra pA e pB
                                                # b = lato opposto a other
                                                
                                                # Lato opposto a pt (il vertice dove calcoliamo l'angolo)
                                                s_opp_pt_key = f"SegDist_{pA}_{pB}" if f"SegDist_{pA}_{pB}" in st.session_state.solved_values else f"SegDist_{pB}_{pA}"
                                                s_opp_pt = float(st.session_state.solved_values[s_opp_pt_key])
                                                
                                                # Lato opposto a other (il vertice dove conosciamo l'angolo)
                                                # Se other = pA, il lato opposto è pt-pB
                                                # Se other = pB, il lato opposto è pt-pA
                                                if other == pA:
                                                    opp_v1, opp_v2 = pt, pB
                                                else:  # other == pB
                                                    opp_v1, opp_v2 = pt, pA
                                                
                                                s_opp_other_key = f"SegDist_{opp_v1}_{opp_v2}" if f"SegDist_{opp_v1}_{opp_v2}" in st.session_state.solved_values else f"SegDist_{opp_v2}_{opp_v1}"
                                                s_opp_other = float(st.session_state.solved_values[s_opp_other_key])
                                                
                                                # Calcolo: sin(pt_angle) = s_opp_pt * sin(other_angle) / s_opp_other
                                                sin_val = (s_opp_pt * math.sin(ang_val_rad)) / s_opp_other
                                                sin_val = max(-1, min(1, sin_val))
                                                alpha_rad = math.asin(sin_val)
                                                
                                                # IMPORTANTE: Per angoli ottusi, arcsin restituisce sempre l'angolo acuto
                                                # Dobbiamo verificare se l'angolo dovrebbe essere ottuso
                                                # Usando la somma degli angoli: se other_angle + alpha > 180°, c'è un problema
                                                # Oppure usiamo il fatto che se il lato opposto è il più lungo, l'angolo è il più grande
                                                if s_opp_pt > s_opp_other and alpha_rad < ang_val_rad:
                                                    # L'angolo dovrebbe essere maggiore, quindi è ottuso
                                                    alpha_rad = math.pi - alpha_rad
                                                
                                                k_part = f"PartAng_{pt}_{min(pA, pB)}_{max(pA, pB)}"
                                                st.session_state.solved_items.add(k_part)
                                                st.session_state.solved_values[k_part] = format_angle_output(alpha_rad, unit)
                                                
                                                # NON salvare come angolo completo per quadrilateri divisi
                                                # L'angolo completo viene calcolato solo con "Somma Parziali"
                                                
                                                # Salva metadata per visualizzazione corretta
                                                st.session_state.partial_angle_metadata[k_part] = {
                                                    'suffix': meta['sub'],
                                                    'pA': pA,
                                                    'pB': pB
                                                }
                                                
                                                # Se l'angolo calcolato non è parziale, salvalo anche come angolo completo del vertice
                                                if not meta.get('sub'):
                                                    st.session_state.solved_items.add(f"Ang_{pt}")
                                                    st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(alpha_rad, unit)
                                                
                                                latex_msg = rf"\sin({greek_symbol}{meta['sub']}) = \frac{{{s_opp_pt:.2f} \cdot \sin({format_angle_output(ang_val_rad, unit, latex=True)})}}{{{s_opp_other:.2f}}} = {sin_val:.4f} \Rightarrow \mathbf{{{format_angle_output(alpha_rad, unit, latex=True)}}}"
                                                descr_text = "Calcolo angolo con Teorema dei Seni."
                                            else:
                                                # Errore: angolo non trovato
                                                st.error(f"❌ Angolo in {other} nel triangolo {pt}-{pA}-{pB} non trovato!")
                                                latex_msg = rf"\text{{Errore: angolo in {other} non trovato}}"
                                                descr_text = f"Impossibile calcolare: angolo in {other} non disponibile."

                                        elif m_type == "sub_tri":
                                            pA, pB, k_A, k_B = meta['pA'], meta['pB'], meta.get('k_A'), meta.get('k_B')
                                            
                                            val_A_str = st.session_state.solved_values.get(k_A)
                                            val_B_str = st.session_state.solved_values.get(k_B)
                                            
                                            if not val_A_str or not val_B_str:
                                                st.error(f"❌ Errore interno: impossibile trovare i valori per le chiavi {k_A} o {k_B}.")
                                                latex_msg = r"\text{Errore: angoli non trovati}"
                                                descr_text = "Calcolo fallito per mancanza di angoli."
                                            else:
                                                val_A = parse_angle(val_A_str.split()[0], unit)
                                                val_B = parse_angle(val_B_str.split()[0], unit)
                                            
                                                tot_rad = math.pi
                                                alpha_rad = tot_rad - (val_A + val_B)
                                                
                                                k_part = f"PartAng_{pt}_{min(pA, pB)}_{max(pA, pB)}"
                                                st.session_state.solved_items.add(k_part)
                                                st.session_state.solved_values[k_part] = format_angle_output(alpha_rad, unit)
                                                
                                                if (not meta.get('sub') or meta['sub'] == '') and len(pts_lbls) == 3:
                                                    st.session_state.solved_items.add(f"Ang_{pt}")
                                                    st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(alpha_rad, unit)
                                                
                                                st.session_state.partial_angle_metadata[k_part] = {'suffix': meta['sub'], 'pA': pA, 'pB': pB}
                                                
                                                # Funzione interna per generare il simbolo greco corretto
                                                def get_greek_from_key(key):
                                                    v_parts = key.split('_'); vertex = v_parts[1]
                                                    v_idx = pts_lbls.index(vertex)
                                                    v_greek = greeks_latex[v_idx] if v_idx < len(greeks_latex) else r"\alpha"
                                                    if key.startswith("PartAng_"):
                                                        v_suffix = st.session_state.partial_angle_metadata.get(key, {}).get('suffix', '')
                                                        return v_greek + v_suffix
                                                    return v_greek
                                                
                                                greek_A_sym = get_greek_from_key(k_A)
                                                greek_B_sym = get_greek_from_key(k_B)
                                                
                                                tot_disp = r"180^{\circ}" if unit == AngleUnit.DEG else r"200^g"
                                                latex_msg = rf"{greek_symbol}{meta['sub']} = {tot_disp} - ({greek_A_sym} + {greek_B_sym}) = \mathbf{{{format_angle_output(alpha_rad, unit, latex=True)}}}"
                                                descr_text = "Calcolo angolo parziale per differenza nel triangolo."

                                        elif m_type == "sum_partials":
                                            val = 0.0
                                            for k, v in st.session_state.solved_values.items():
                                                if k.startswith(f"PartAng_{pt}_"):
                                                    val += parse_angle(v.split()[0], unit)
                                            
                                            # Rimuovi i parziali dai solved items per pulizia
                                            for k in list(st.session_state.solved_items):
                                                if k.startswith(f"PartAng_{pt}_"):
                                                    st.session_state.solved_items.discard(k)
                                            
                                            # Aggiungi l'angolo completo a solved_items E solved_values
                                            st.session_state.solved_items.add(f"Ang_{pt}")
                                            st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(val, unit)
                                            latex_msg = rf"{greek_symbol} = \sum \alpha_i = \mathbf{{{format_angle_output(val, unit, latex=True)}}}"
                                            descr_text = "Somma degli angoli parziali."

                                    # --- METODI STANDARD (Azimut) ---
                                    elif "Azimut" in ans:
                                        # Metodo Azimut
                                        az_in = math.atan2(prev['x']-curr['x'], prev['y']-curr['y'])
                                        az_out = math.atan2(nex['x']-curr['x'], nex['y']-curr['y'])
                                        
                                        # Normalizzazione 0-2pi per coerenza visuale
                                        az_in = (az_in + 2*math.pi) % (2*math.pi)
                                        az_out = (az_out + 2*math.pi) % (2*math.pi)
                                        
                                        raw_diff = az_out - az_in
                                        
                                        # Stringhe per report dettagliato
                                        az_out_str = format_angle_output(az_out, unit, True)
                                        az_in_str = format_angle_output(az_in, unit, True)
                                        
                                        generic_formula = rf"{greek_symbol} = ({pt}{nex_lbl}) - ({pt}{prev_lbl})"
                                        numeric_formula = rf"{az_out_str} - {az_in_str}"
                                        descr_text = f"Calcolo angolo interno {greek_symbol} come differenza tra Azimut Uscita e Azimut Entrata."

                                        if raw_diff < 0:
                                            step1_str = format_angle_output(raw_diff, unit, latex=True) # Sarà negativo
                                            full_circle_str = r"360^{\circ}" if unit == AngleUnit.DEG else r"400^g"
                                            final_val = raw_diff + (2*math.pi)
                                            final_str = format_angle_output(final_val, unit, latex=True)
                                            
                                            latex_msg = rf"""\begin{{aligned}} {generic_formula} &= {numeric_formula} \\ &= {step1_str} \quad (\text{{Neg}}!) \\ &= {step1_str} + {full_circle_str} \\ &= \mathbf{{{final_str}}} \end{{aligned}}"""
                                            st.session_state.solved_items.add(f"Ang_{pt}")
                                            st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(final_val, unit)
                                        else:
                                            final_val = raw_diff
                                            st.session_state.solved_items.add(f"Ang_{pt}")
                                            st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(final_val, unit)
                                            final_str = format_angle_output(final_val, unit, latex=True)
                                            latex_msg = rf"""\begin{{aligned}} {generic_formula} &= {numeric_formula} \\ &= \mathbf{{{final_str}}} \end{{aligned}}"""

                                # Se abbiamo concluso il calcolo dell'angolo target del workflow, chiudiamo il flusso
                                if act == "calc_ang" and st.session_state.angle_workflow_target == parts[2]:
                                    st.session_state.angle_workflow_target = None

                                # Calcola differenze per Undo
                                added_items = list(st.session_state.solved_items - pre_items)
                                added_values = list(set(st.session_state.solved_values.keys()) - pre_values)

                                st.session_state.log.append({'action': q['desc'], 'method': ans, 'result': latex_msg, 'desc_verbose': descr_text, 'added_items': added_items, 'added_values': added_values})
                                st.session_state.last_calc_msg = latex_msg
                                st.session_state.current_mission = None
                                st.session_state.selected_partial_suffix = None  # Reset dopo calcolo
                                st.rerun()
                            else: 
                                # Registra l'errore nel log
                                st.session_state.log.append({
                                    'action': q['desc'], 
                                    'method': ans, 
                                    'result': '\\text{ERRORE: Strategia errata}', 
                                    'desc_verbose': f'Hai scelto una formula sbagliata: {ans}. La formula corretta era tra: {", ".join(corrects)}',
                                    'added_items': [],
                                    'added_values': [],
                                    'is_error': True
                                })
                                st.error("Strategia errata.")
