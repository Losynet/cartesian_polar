import streamlit as st
import pandas as pd
import numpy as np
import math
import itertools
from collections import defaultdict
import plotly.graph_objects as go

# --- CONFIGURAZIONE ---
st.set_page_config(layout="wide", page_title="GeoSolver v67 - Prof. Losenno")

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
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
st.markdown("<div class='prof-title'>📐 GeoSolver Ultimate</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Progetto didattico del Prof. G. Losenno</div>", unsafe_allow_html=True)

# --- UTILS ---
class AngleUnit:
    GON = "Gon"
    DEG = "Deg"
    RAD = "Rad"

def parse_angle(val, mode):
    try:
        val = float(val)
        if mode == AngleUnit.GON: return val * (math.pi / 200.0)
        elif mode == AngleUnit.DEG: return math.radians(val)
        elif mode == AngleUnit.RAD: return val
    except: return 0.0

def format_angle_output(rad_val, unit_mode, latex=False):
    deg_sym = "^{\circ}" if latex else "°"
    if unit_mode == AngleUnit.RAD: return f"{rad_val:.4f} rad"
    elif unit_mode == AngleUnit.DEG: return f"{math.degrees(rad_val):.4f}{deg_sym}"
    elif unit_mode == AngleUnit.GON: return f"{(rad_val * 200.0 / math.pi):.4f}g"
    return str(rad_val)

def format_coord(val): return f"{val:.2f}"

def get_shuffled_options(correct, wrongs):
    import random
    opts = [correct] + wrongs
    random.shuffle(opts)
    return opts

# --- REPORT HTML ---
def generate_html_report(log_entries, solved_values):
    rows_html = ""
    for i, entry in enumerate(log_entries):
        clean_res = entry['result']
        clean_method = entry['method'].replace(r"$", "")
        desc_text = entry.get('desc_verbose', 'Calcolo eseguito.')
        
        rows_html += f"""
        <div class="entry">
            <div class="entry-header">Step {i+1}: {entry['action']}</div>
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
            .entry {{ border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 25px; background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.05); overflow: hidden; }}
            .entry-header {{ background: #2980b9; color: white; padding: 10px 15px; font-weight: bold; font-size: 1.1em; }}
            .entry-body {{ padding: 20px; }}
            .description {{ background: #fffbe6; border-left: 4px solid #f1c40f; padding: 15px; margin-bottom: 15px; font-style: italic; color: #555; }}
            .formula-row {{ margin-bottom: 10px; color: #7f8c8d; font-weight: bold; }}
            .math-box {{ font-family: 'Times New Roman', serif; font-size: 1.2em; background: #f8f9fa; padding: 15px; border: 1px dashed #ccc; border-radius: 5px; overflow-x: auto; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 30px; border: 1px solid #ddd; }}
            td, th {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #f2f2f2; color: #2c3e50; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>📐 GeoSolver - Report Tecnico</h1>
        <p><b>Studente:</b> ___________________ &nbsp;|&nbsp; <b>Docente:</b> Prof. G. Losenno</p>
        <hr>
        {rows_html}
        <h3>Riepilogo Risultati</h3>
        <table>
            <tr><th>Grandezza</th><th>Valore</th></tr>
            {''.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in solved_values.items()])}
        </table>
    </body>
    </html>
    """
    return html

# --- STATE ---
if 'raw_data' not in st.session_state: st.session_state.raw_data = [] 
if 'points' not in st.session_state: st.session_state.points = {} 
if 'log' not in st.session_state: st.session_state.log = []
if 'projections_visible' not in st.session_state: st.session_state.projections_visible = False 
if 'solved_items' not in st.session_state: st.session_state.solved_items = set() 
if 'solved_values' not in st.session_state: st.session_state.solved_values = {} 
if 'current_options' not in st.session_state: st.session_state.current_options = None
if 'current_mission' not in st.session_state: st.session_state.current_mission = None
if 'last_calc_msg' not in st.session_state: st.session_state.last_calc_msg = None
if 'input_interpretation' not in st.session_state: st.session_state.input_interpretation = AngleUnit.GON 
if 'ang_method_choice' not in st.session_state: st.session_state.ang_method_choice = None
if 'az_segments_confirmed' not in st.session_state: st.session_state.az_segments_confirmed = False
if 'angle_workflow_target' not in st.session_state: st.session_state.angle_workflow_target = None
if 'dist_method_preference' not in st.session_state: st.session_state.dist_method_preference = None

# [SEZIONE 1: NUOVO STATO AZIMUT AGGIORNATO CON PENDING TARGET]
if 'az_workflow' not in st.session_state:
    st.session_state.az_workflow = {
        'active': False,
        'vertex': None,
        'step': 0,      # 0:Start, 1:Lato 1, 2:Lato 2, 3:Diff
        'side1': None,
        'side2': None,
        'az1_val': None,
        'az2_val': None,
        'pending_target': None, # Il punto che stiamo calcolando ORA (fase quiz)
        'quiz_options': None    # Le opzioni del quiz corrente
    }

def undo_last_action():
    if st.session_state.log:
        last = st.session_state.log.pop()
        for k in last.get('added_items', []): st.session_state.solved_items.discard(k)
        for k in last.get('added_values', []): st.session_state.solved_values.pop(k, None)
        st.session_state.last_calc_msg = None
        st.session_state.current_mission = None
        st.session_state.az_workflow['active'] = False
        st.rerun()

def recalculate_points():
    st.session_state.points = {}
    st.session_state.solved_items = set()
    st.session_state.solved_values = {}
    st.session_state.az_workflow = {'active': False, 'vertex': None, 'step': 0, 'side1': None, 'side2': None, 'az1_val': None, 'az2_val': None, 'pending_target': None, 'quiz_options': None}
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
        with col1: pt_lbl = st.text_input("Etichetta", "P")
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

def draw_angle_wedge(fig, pt_lbl, color, radius, text=None):
    pts_lbls = sorted(st.session_state.points.keys())
    try:
        idx = pts_lbls.index(pt_lbl)
        curr = st.session_state.points[pt_lbl]
        prev = st.session_state.points[pts_lbls[idx-1]]
        nex = st.session_state.points[pts_lbls[(idx+1)%len(pts_lbls)]]
        
        az_in = math.atan2(prev['x']-curr['x'], prev['y']-curr['y'])
        az_out = math.atan2(nex['x']-curr['x'], nex['y']-curr['y'])
        
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
            fig.add_annotation(x=txt_x, y=txt_y, text=text, showarrow=False, font=dict(color="black", size=14, weight="bold"))
            
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
                        if idx < len(greeks): ang_lbl = greeks[idx]
                    except: pass
                draw_angle_wedge(fig, pt_lbl, "rgba(0,128,0,0.15)", radius, text=ang_lbl)

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
                pts_lbls = sorted(st.session_state.points.keys())
                idx = pts_lbls.index(pt_lbl)
                prev = pts_lbls[idx-1]; nex = pts_lbls[(idx+1)%len(pts_lbls)]
                
                if st.session_state.ang_method_choice == "Calcolo con Azimut vertici":
                    # Verifica stato risoluzione
                    az_prev_known = f"SegAz_{pt_lbl}_{prev}" in st.session_state.solved_items
                    az_nex_known = f"SegAz_{pt_lbl}_{nex}" in st.session_state.solved_items
                    
                    if az_prev_known and az_nex_known:
                        pass # Lascia fare alla Sezione 2 (Solved Items)
                    else:
                        # Fase di scelta: mostra solo quello selezionato nel radio button
                        sel = st.session_state.get("az_fix_radio")
                        if sel:
                            try:
                                # sel format: "Azimut A-B"
                                parts = sel.split(" ")[1].split("-")
                                if parts[0] == pt_lbl:
                                    draw_azimuth_visuals(fig, parts[0], parts[1], radius, 0)
                            except: pass
                
                draw_angle_wedge(fig, pt_lbl, "rgba(255, 165, 0, 0.2)", radius)
                    
        # 4. Punti
        for lbl, p in st.session_state.points.items():
            pos = get_point_label_pos(p['x'], p['y'])
            
            h_lines = [f"<b>Punto {lbl}</b>"]
            if f"X_{lbl}" in st.session_state.solved_values: h_lines.append(f"X: {st.session_state.solved_values[f'X_{lbl}']}")
            if f"Y_{lbl}" in st.session_state.solved_values: h_lines.append(f"Y: {st.session_state.solved_values[f'Y_{lbl}']}")
            if f"Dist_{lbl}" in st.session_state.solved_values: h_lines.append(f"Dist: {st.session_state.solved_values[f'Dist_{lbl}']}")
            if f"Az_{lbl}" in st.session_state.solved_values: h_lines.append(f"Az: {st.session_state.solved_values[f'Az_{lbl}']}")
            
            fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers+text', marker=dict(size=12, color='blue', line=dict(width=2, color='white')), text=[lbl], textposition=pos, textfont=dict(size=12, color="darkblue", weight="bold"), showlegend=False, hoverinfo='text', hovertext="<br>".join(h_lines)))
            if st.session_state.projections_visible:
                x_levels = calculate_label_levels([(p['x'], lbl) for lbl, p in st.session_state.points.items()])
                y_levels = calculate_label_levels([(p['y'], lbl) for lbl, p in st.session_state.points.items()])
                fig.add_trace(go.Scatter(x=[p['x'], p['x'], 0], y=[0, p['y'], p['y']], mode='lines', line=dict(color='red', width=1, dash='dash'), showlegend=False))
                lvl_x = x_levels.get(lbl, 0); pixel_offset_x = 30 + (lvl_x * 25)
                txt_x = st.session_state.solved_values.get(f"X_{lbl}", f"X_{lbl}"); color_x = 'red' if f"X_{lbl}" in st.session_state.solved_items else 'gray'
                fig.add_annotation(x=p['x'], y=0, text=txt_x, textangle=0, font=dict(size=10, color=color_x, weight='bold'), showarrow=True, arrowhead=0, arrowcolor="gray", arrowwidth=1, ax=0, ay=pixel_offset_x, bgcolor="rgba(255,255,255,0.8)", borderpad=1)
                lvl_y = y_levels.get(lbl, 0); pixel_offset_y = 30 + (lvl_y * 25)
                txt_y = st.session_state.solved_values.get(f"Y_{lbl}", f"Y_{lbl}"); color_y = 'red' if f"Y_{lbl}" in st.session_state.solved_items else 'gray'
                fig.add_annotation(x=0, y=p['y'], text=txt_y, textangle=0, font=dict(size=10, color=color_y, weight='bold'), showarrow=True, arrowhead=0, arrowcolor="gray", arrowwidth=1, ax=-pixel_offset_y, ay=0, bgcolor="rgba(255,255,255,0.8)", borderpad=1)

        # 5. Linee Segmenti
        pts_lbls = sorted(st.session_state.points.keys())
        for item in st.session_state.solved_items:
            if item.startswith("Seg_") and "SegDist" not in item:
                parts = item.split("_"); p1 = st.session_state.points[parts[1]]; p2 = st.session_state.points[parts[2]]
                
                h_txt = f"<b>Segmento {parts[1]}-{parts[2]}</b>"
                if f"SegDist_{parts[1]}_{parts[2]}" in st.session_state.solved_values:
                    h_txt += f"<br>Lunghezza: {st.session_state.solved_values[f'SegDist_{parts[1]}_{parts[2]}']}"
                if f"SegAz_{parts[1]}_{parts[2]}" in st.session_state.solved_values:
                    h_txt += f"<br>Azimut ({parts[1]}{parts[2]}): {st.session_state.solved_values[f'SegAz_{parts[1]}_{parts[2]}']}"
                if f"SegAz_{parts[2]}_{parts[1]}" in st.session_state.solved_values:
                    h_txt += f"<br>Azimut ({parts[2]}{parts[1]}): {st.session_state.solved_values[f'SegAz_{parts[2]}_{parts[1]}']}"
                
                fig.add_trace(go.Scatter(x=[p1['x'], p2['x']], y=[p1['y'], p2['y']], mode='lines', line=dict(color='green', width=2), showlegend=False, hoverinfo='text', text=h_txt))
                if f"SegDist_{parts[1]}_{parts[2]}" in st.session_state.solved_values:
                    mx, my = (p1['x']+p2['x'])/2, (p1['y']+p2['y'])/2
                    val = st.session_state.solved_values[f"SegDist_{parts[1]}_{parts[2]}"]
                    
                    lbl_text = val
                    side_name = None
                    
                    if len(pts_lbls) == 3:
                        # Triangolo: lato opposto al vertice
                        rem = [x for x in pts_lbls if x not in [parts[1], parts[2]]]
                        if len(rem) == 1: side_name = rem[0].lower()
                    elif len(pts_lbls) > 3:
                        # Poligono: sequenziale (AB=a, BC=b...)
                        try:
                            i1 = pts_lbls.index(parts[1]); i2 = pts_lbls.index(parts[2]); n = len(pts_lbls)
                            if (i1 + 1) % n == i2: side_name = chr(ord('a') + i1)
                            elif (i2 + 1) % n == i1: side_name = chr(ord('a') + i2)
                        except: pass
                    
                    if side_name: lbl_text = f"{side_name} = {val}"
                    
                    rot_angle = get_text_rotation(p1, p2)
                    fig.add_annotation(x=mx, y=my, text=lbl_text, textangle=rot_angle, font=dict(color="green", size=11, weight="bold"), bgcolor="white", borderpad=2, showarrow=False)

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
        # Se ho le coordinate polari (missing_pol=False), nascondo il Punto per forzare la scelta strategica sul segmento
        if missing_pol:
            opt_punti.append(f"Punto {p}")
            
    # 2. Segmenti (Logica Rigida: Distanza -> Azimut)
    opt_seg = []
    if len(pts_keys) >= 2:
        # Distanze (Non direzionali - Combinazioni)
        for c in list(itertools.combinations(pts_keys, 2)):
            # Controlla se la distanza esiste in una delle due direzioni
            if f"SegDist_{c[0]}_{c[1]}" not in st.session_state.solved_items and f"SegDist_{c[1]}_{c[0]}" not in st.session_state.solved_items:
                opt_seg.append(f"Lunghezza {c[0]}{c[1]}")
        
        # Azimut (Direzionali - Permutazioni)
        for p1 in pts_keys:
            for p2 in pts_keys:
                if p1 == p2: continue
                dist_known = f"SegDist_{p1}_{p2}" in st.session_state.solved_items or f"SegDist_{p2}_{p1}" in st.session_state.solved_items
                az_known = f"SegAz_{p1}_{p2}" in st.session_state.solved_items
                if dist_known and not az_known:
                    opt_seg.append(f"Azimut {p1} → {p2}")

    # 3. Angoli (Logica Rigida: Azimut In/Out noti)
    opt_ang = []
    if len(pts_keys) >= 3:
        # Verifica preliminare per strategie triangolo
        known_sides_count = 0
        if len(pts_keys) == 3:
            for i in range(len(pts_keys)):
                p1, p2 = pts_keys[i], pts_keys[(i+1)%3]
                if f"SegDist_{p1}_{p2}" in st.session_state.solved_values or f"SegDist_{p2}_{p1}" in st.session_state.solved_values:
                    known_sides_count += 1

        for p in pts_keys:
             if f"Ang_{p}" not in st.session_state.solved_items:
                 idx = pts_keys.index(p)
                 prev_p = pts_keys[idx-1]; next_p = pts_keys[(idx+1)%len(pts_keys)]
                 
                 # 1. Differenza Azimut
                 has_az_in = f"SegAz_{prev_p}_{p}" in st.session_state.solved_items or f"SegAz_{p}_{prev_p}" in st.session_state.solved_items
                 has_az_out = f"SegAz_{p}_{next_p}" in st.session_state.solved_items or f"SegAz_{next_p}_{p}" in st.session_state.solved_items
                 
                 # 2. Carnot / Teoremi (Richiede triangolo chiuso: Lati adiacenti + Lato opposto/Diagonale)
                 s_adj1 = f"SegDist_{p}_{next_p}" in st.session_state.solved_items or f"SegDist_{next_p}_{p}" in st.session_state.solved_items
                 s_adj2 = f"SegDist_{p}_{prev_p}" in st.session_state.solved_items or f"SegDist_{prev_p}_{p}" in st.session_state.solved_items
                 s_opp = f"SegDist_{prev_p}_{next_p}" in st.session_state.solved_items or f"SegDist_{next_p}_{prev_p}" in st.session_state.solved_items
                 
                 can_theorems = (s_adj1 and s_adj2 and s_opp)
                 
                 # 3. Teorema dei Seni (Lato opposto noto + Altra coppia Lato/Angolo nota)
                 can_sines = False
                 if len(pts_keys) == 3:
                     opp_side_key = f"SegDist_{prev_p}_{next_p}" if f"SegDist_{prev_p}_{next_p}" in st.session_state.solved_values else f"SegDist_{next_p}_{prev_p}"
                     if opp_side_key in st.session_state.solved_values:
                         for other in pts_keys:
                             if other == p: continue
                             if f"Ang_{other}" in st.session_state.solved_values:
                                 o_idx = pts_keys.index(other); o_prev = pts_keys[o_idx-1]; o_next = pts_keys[(o_idx+1)%3]
                                 o_opp_key = f"SegDist_{o_prev}_{o_next}" if f"SegDist_{o_prev}_{o_next}" in st.session_state.solved_values else f"SegDist_{o_next}_{o_prev}"
                                 if o_opp_key in st.session_state.solved_values: can_sines = True; break

                 if (has_az_in and has_az_out) or can_theorems or can_sines:
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
    if "Angolo interno" in subject:
        pt = subject.split(" ")[-1]; return True, "OK", f"calc_ang_{pt}"
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

def get_strategies_for_mission(mission_code, method_filter=None):
    parts = mission_code.split("_"); act = parts[0] + "_" + parts[1]
    q = {}
    if act == "calc_X": 
        pt = parts[2]
        q['correct'] = rf"$X_{{{pt}}} = \overline{{O{pt}}} \cdot \sin(O{pt})$"
        q['wrongs'] = [rf"$X_{{{pt}}} = \overline{{O{pt}}} \cdot \cos(O{pt})$"]
        q['latex'] = rf"X_{{{pt}}} = \overline{{O{pt}}} \cdot \sin(O{pt})"
        q['desc'] = rf"Calcolo X di {pt}"
    elif act == "calc_Y": 
        pt = parts[2]
        q['correct'] = rf"$Y_{{{pt}}} = \overline{{O{pt}}} \cdot \cos(O{pt})$"
        q['wrongs'] = [rf"$Y_{{{pt}}} = \overline{{O{pt}}} \cdot \sin(O{pt})$"]
        q['latex'] = rf"Y_{{{pt}}} = \overline{{O{pt}}} \cdot \cos(O{pt})"
        q['desc'] = rf"Calcolo Y di {pt}"
    elif act == "calc_dist":
        pt = parts[2]
        q['correct'] = rf"$\overline{{O{pt}}} = \sqrt{{X_{{{pt}}}^2 + Y_{{{pt}}}^2}}$"
        q['wrongs'] = [rf"$\overline{{O{pt}}} = X_{{{pt}}} + Y_{{{pt}}}$"]
        q['latex'] = rf"\overline{{O{pt}}} = \sqrt{{X_{{{pt}}}^2 + Y_{{{pt}}}^2}}"
        q['desc'] = rf"Calcolo Distanza O-{pt}"
    elif act == "calc_az":
        pt = parts[2]
        q['correct'] = rf"$(O{pt}) = \arctan(X_{{{pt}}} / Y_{{{pt}}})$"
        q['wrongs'] = [rf"$(O{pt}) = \arctan(Y_{{{pt}}} / X_{{{pt}}})$"]
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
                "wrongs": [rf"Carnot: $\overline{{{p1}{p2}}} = \sqrt{{ \overline{{O{p1}}}^2 + \overline{{O{p2}}}^2 + 2 \cdot \overline{{O{p1}}} \cdot \overline{{O{p2}}} \cdot \cos(\Delta \theta) }}$"]
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
        q['wrongs'] = [r"$\sin(\Delta X)$"]
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

        # Sottrazione (se triangolo e 2 angoli noti)
        if len(pts_lbls) == 3:
            known_angs = [k for k in st.session_state.solved_items if k.startswith("Ang_") and k != f"Ang_{pt}"]
            if len(known_angs) == 2:
                tot = "180^{\circ}" if st.session_state.input_interpretation == AngleUnit.DEG else "200^g"
                others_list = []
                for k in known_angs:
                    lbl = k.split('_')[1]
                    try:
                        i_lbl = pts_lbls.index(lbl)
                        g_lbl = greeks_latex[i_lbl] if i_lbl < len(greeks_latex) else rf"\alpha_{{{lbl}}}"
                    except: g_lbl = rf"\alpha_{{{lbl}}}"
                    others_list.append(g_lbl)
                others = "+".join(others_list)
                methods["Somma Int."] = {
                    "correct": rf"Somma Int.: ${greek_pt} = {tot} - ({others})$",
                    "wrongs": [rf"Somma Int.: ${greek_pt} = {tot} + ({others})$"]
                }
        
        # Teorema del Coseno (Carnot)
        s_adj1_key = f"SegDist_{pt}_{nex}" if f"SegDist_{pt}_{nex}" in st.session_state.solved_values else f"SegDist_{nex}_{pt}"
        s_adj2_key = f"SegDist_{pt}_{prev}" if f"SegDist_{pt}_{prev}" in st.session_state.solved_values else f"SegDist_{prev}_{pt}"
        s_opp_key = f"SegDist_{prev}_{nex}" if f"SegDist_{prev}_{nex}" in st.session_state.solved_values else f"SegDist_{nex}_{prev}"

        if s_adj1_key in st.session_state.solved_values and s_adj2_key in st.session_state.solved_values:
            if s_opp_key in st.session_state.solved_values or len(pts_lbls) >= 4:
                methods["Carnot"] = {
                    "correct": rf"Carnot: ${greek_pt} = \arccos\left(\frac{{\overline{{{pt}{nex}}}^2 + \overline{{{pt}{prev}}}^2 - \overline{{{prev}{nex}}}^2}}{{2 \cdot \overline{{{pt}{nex}}} \cdot \overline{{{pt}{prev}}}}}\right)$",
                    "wrongs": [rf"Carnot: ${greek_pt} = \arccos\left(\frac{{\overline{{{pt}{nex}}}^2 + \overline{{{pt}{prev}}}^2 + \overline{{{prev}{nex}}}^2}}{{2 \cdot \overline{{{pt}{nex}}} \cdot \overline{{{pt}{prev}}}}}\right)$", rf"Carnot: ${greek_pt} = \arcsin(...)$"]
                }

        # Teorema dei Seni
        if len(pts_lbls) == 3:
            opp_side_key = f"SegDist_{prev}_{nex}" if f"SegDist_{prev}_{nex}" in st.session_state.solved_values else f"SegDist_{nex}_{prev}"
            if opp_side_key in st.session_state.solved_values:
                 for other in pts_lbls:
                     if other == pt: continue
                     if f"Ang_{other}" in st.session_state.solved_values:
                         o_idx = pts_lbls.index(other); o_prev = pts_lbls[o_idx-1]; o_next = pts_lbls[(o_idx+1)%3]
                         o_opp_key = f"SegDist_{o_prev}_{o_next}" if f"SegDist_{o_prev}_{o_next}" in st.session_state.solved_values else f"SegDist_{o_next}_{o_prev}"
                         if o_opp_key in st.session_state.solved_values:
                             # Formula: sin(alpha) = a * sin(beta) / b
                             # a = opp_side_key (lato opposto a pt)
                             # beta = Ang_{other}
                             # b = o_opp_key (lato opposto a other)
                             o_idx = pts_lbls.index(other)
                             greek_other = greeks_latex[o_idx] if o_idx < len(greeks_latex) else rf"\alpha_{{{other}}}"
                             methods["Seni"] = {
                                 "correct": rf"Seni: $\sin({greek_pt}) = \frac{{\overline{{{prev}{nex}}} \cdot \sin({greek_other})}}{{\overline{{{o_prev}{o_next}}}}}$",
                                 "wrongs": [rf"Seni: $\sin({greek_pt}) = \frac{{\overline{{{o_prev}{o_next}}} \cdot \sin({greek_other})}}{{\overline{{{prev}{nex}}}}}$"]
                             }
                             break

        q['available_methods'] = list(methods.keys())
        
        # Selezione Metodo
        selected_data = None
        if method_filter and method_filter in methods:
            selected_data = methods[method_filter]
        elif len(methods) > 0:
            # Fallback: prendi il primo o tutti (qui prendiamo il primo per coerenza se non filtrato)
            selected_data = list(methods.values())[0]
            
        if selected_data:
            q['correct'] = selected_data['correct']
            q['correct_list'] = [selected_data['correct']]
            q['wrongs'] = selected_data['wrongs']
        else:
            q['correct'] = "N/A"; q['correct_list'] = []; q['wrongs'] = []

        q['latex'] = rf"{greek_pt} = ({pt}{nex}) - ({pt}{prev})"
        q['desc'] = f"Calcolo Angolo in {pt}"
    return q

with col_tutor:
    st.subheader("Tutor")
    if st.session_state.log:
        html_report = generate_html_report(st.session_state.log, st.session_state.solved_values)
        st.download_button("📄 Report", data=html_report, file_name="geosolver_report.html", mime="text/html")
        st.button("↩️ Annulla ultima operazione", on_click=undo_last_action)
    
    if st.session_state.last_calc_msg:
        st.success("✅ **Fatto:**")
        st.latex(st.session_state.last_calc_msg.replace("°", "^{\circ}"))

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
                        sym = "^g" if unit == AngleUnit.GON else "^{\circ}"
                        
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
                        st.error("Formula errata. Ricorda: l'azimut topografico è l'arcotangente di Delta X su Delta Y.")
        # --- LOGICA DEGLI STEP ---

        # STEP 1A: SELEZIONE PRIMO LATO
        if wf['step'] == 1 and wf['pending_target'] is None:
            st.markdown(f"**Step 1:** Seleziona il **primo lato** uscente da {v} (Ciano):")
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
            # Escludi il lato già calcolato
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
                        sym = "^g" if unit == AngleUnit.GON else "^{\circ}"
                        
                        az1_disp = az1 * factor
                        az2_disp = az2 * factor
                        
                        diff_disp = abs(az1_disp - az2_disp)
                        if diff_disp > full_circle / 2: diff_disp = full_circle - diff_disp
                        
                        final_key = f"Ang_{v}"
                        st.session_state.solved_items.add(final_key)
                        # Salvataggio valore (riconversione in radianti dal valore visualizzato o dall'originale)
                        diff_deg = abs(az1 - az2)
                        if diff_deg > 180: diff_deg = 360 - diff_deg
                        rad_val = math.radians(diff_deg)
                        
                        st.session_state.solved_values[final_key] = format_angle_output(rad_val, st.session_state.input_interpretation)
                        
                        # Etichetta Greca
                        pts_lbls = sorted(st.session_state.points.keys())
                        idx = pts_lbls.index(v)
                        greeks = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta"]
                        greek_label = greeks[idx] if idx < len(greeks) else rf"\alpha_{{{v}}}"
                        
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
        if not avail: 
            st.success("Hai completato l'esercizio!")
        else:
            sel_subj = st.selectbox("1. Calcola:", avail)
            possible_actions = []
            if "Angolo" in sel_subj:
                possible_actions.append("Calcolo con i Teoremi")
                possible_actions.append("Calcolo con Azimut vertici")
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
                
                # Applica preferenza se esistente
                pref = st.session_state.dist_method_preference
                if pref and pol_ok:
                    # Se abbiamo una preferenza e i dati polari lo permettono (o permettono la scelta)
                    # Filtriamo le azioni. Nota: Cartesian è sempre possibile se pol_ok (via conversione)
                    if pref == "Metodo Diretto (Carnot)": possible_actions.append(pref)
                    elif pref == "Metodo Indiretto (Cartesiano)": possible_actions.append(pref)
                elif pol_ok:
                    # Nessuna preferenza, offri entrambe
                    possible_actions.extend(["Metodo Diretto (Carnot)", "Metodo Indiretto (Cartesiano)"])
                elif cart_ok:
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
                        pt = sel_subj.split(" ")[-1]
                        code = f"calc_ang_{pt}"
                        # Store the category choice. Reusing ang_method_choice for the category.
                        st.session_state.ang_method_choice = sel_act 
                    elif "Lunghezza" in sel_subj:
                        p1, p2 = sel_subj.split()[1][0], sel_subj.split()[1][1]
                        
                        # Salva la preferenza per i prossimi calcoli di lunghezza
                        if not st.session_state.dist_method_preference:
                            st.session_state.dist_method_preference = sel_act
                        
                        if sel_act == "Metodo Indiretto (Cartesiano)":
                            cart_ok = (f"X_{p1}" in st.session_state.solved_items and f"Y_{p1}" in st.session_state.solved_items and
                                       f"X_{p2}" in st.session_state.solved_items and f"Y_{p2}" in st.session_state.solved_items)
                            if not cart_ok:
                                if not st.session_state.projections_visible: activate_projections()
                                target_mission = None
                                for p in [p1, p2]:
                                    if f"X_{p}" not in st.session_state.solved_items: target_mission = f"calc_X_{p}"; break
                                    if f"Y_{p}" not in st.session_state.solved_items: target_mission = f"calc_Y_{p}"; break
                                if target_mission:
                                    st.session_state.current_mission = target_mission
                                    st.session_state.current_options = None
                                    st.rerun()

                        code = f"seg_dist_{p1}_{p2}"
                        st.session_state.ang_method_choice = sel_act
                    else:
                        ok, msg, code_from_func = check_goal_feasibility(sel_subj, sel_act)
                        if ok:
                            code = code_from_func
                            # Attiva automaticamente le proiezioni se si calcolano coordinate cartesiane
                            if code and ("calc_X" in code or "calc_Y" in code) and not st.session_state.projections_visible:
                                activate_projections()
                        # Clear the choice for non-angle calculations
                        st.session_state.ang_method_choice = None

                    if code:
                        st.session_state.current_mission = code
                        st.session_state.current_options = None 
                        st.session_state.az_segments_confirmed = False
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
                    if category and category in all_methods: method_filter = category
                    if method_filter: q = get_strategies_for_mission(st.session_state.current_mission, method_filter)
                    else: q = get_strategies_for_mission(st.session_state.current_mission)

                if "calc_ang" in st.session_state.current_mission:
                    q_check = get_strategies_for_mission(st.session_state.current_mission)
                    all_methods = q_check.get("available_methods", [])
                    category = st.session_state.get('ang_method_choice')

                    methods_in_category = []
                    if category == "Calcolo con i Teoremi":
                        methods_in_category = [m for m in all_methods if m in ['Carnot', 'Seni', 'Somma Int.']]
                    elif category == "Calcolo con Azimut vertici":
                        methods_in_category = [m for m in all_methods if m == 'Azimut']
                    
                    if not methods_in_category:
                        if category == "Calcolo con Azimut vertici":
                            pt = st.session_state.current_mission.split("_")[2]
                            
                            st.info("🌟 **Modalità Interattiva**")
                            st.write("Per questo calcolo useremo il metodo visuale passo-passo.")
                            
                            # ATTIVAZIONE NUOVO WORKFLOW
                            if st.button("Avvia Procedura Guidata"):
                                st.session_state.az_workflow['active'] = True
                                st.session_state.az_workflow['vertex'] = pt
                                st.session_state.az_workflow['step'] = 1
                                st.session_state.current_mission = None # Pulisce la missione vecchia
                                st.rerun()
                                
                            show_strategy_form = False
                        else:
                            st.warning(f"Nessun metodo di tipo '{category}' è attualmente applicabile per questo angolo.")
                            st.session_state.current_mission = None; st.rerun()
                    elif len(methods_in_category) == 1:
                        method_filter = methods_in_category[0]
                        st.caption(f"Metodo Selezionato: **{method_filter}**")
                        
                        if category == "Calcolo con Azimut vertici":
                             # FALLBACK SE QUALCOSA VA STORTO, MA DOVREBBE PRENDERE L'IF SOPRA
                             pass
                    else:
                        st.write("3. Scegli il Teorema specifico:")
                        method_filter = st.radio("Teorema:", methods_in_category, horizontal=True, label_visibility="collapsed")
                        num_steps_offset = 1
                    
                    if method_filter:
                        q = get_strategies_for_mission(st.session_state.current_mission, method_filter)
                else:
                    q = get_strategies_for_mission(st.session_state.current_mission)
                
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
                            all_opts = corrects + q.get('wrongs', [])
                            import random; random.shuffle(all_opts)
                            st.session_state.current_options = all_opts
                    
                    with st.form("strat"):
                        st.write(f"{3 + num_steps_offset}. Strategia:")
                        ans = st.radio("Formula:", st.session_state.current_options)
                        if st.form_submit_button("Calcola"):
                            corrects = q.get('correct_list', [q['correct']])
                            if ans in corrects:
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
                                    pt = parts[2]
                                    st.session_state.solved_items.add(f"Ang_{pt}")
                                    pts_lbls = sorted(st.session_state.points.keys())
                                    idx = pts_lbls.index(pt)
                                    prev_lbl = pts_lbls[idx-1]
                                    nex_lbl = pts_lbls[(idx+1)%len(pts_lbls)]
                                    prev = st.session_state.points[prev_lbl]
                                    curr = st.session_state.points[pt]
                                    nex = st.session_state.points[nex_lbl]
                                    
                                    # Gestione Metodo Sottrazione
                                    greeks_latex = ["\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta"]
                                    greek_symbol = greeks_latex[idx] if idx < len(greeks_latex) else f"\\alpha_{{{pt}}}"

                                    if "180" in ans or "200" in ans:
                                        az_in = math.atan2(prev['x']-curr['x'], prev['y']-curr['y'])
                                        az_out = math.atan2(nex['x']-curr['x'], nex['y']-curr['y'])
                                        final_val = (az_out - az_in) % (2*math.pi)
                                        tot_str = "180^{\circ}" if unit == AngleUnit.DEG else "200^g"
                                        prev_idx = pts_lbls.index(prev_lbl)
                                        next_idx = pts_lbls.index(nex_lbl)
                                        prev_greek = greeks_latex[prev_idx] if prev_idx < len(greeks_latex) else f"\\alpha_{{{prev_lbl}}}"
                                        next_greek = greeks_latex[next_idx] if next_idx < len(greeks_latex) else f"\\alpha_{{{nex_lbl}}}"
                                        
                                        latex_msg = rf"{greek_symbol} = {tot_str} - ({prev_greek} + {next_greek}) \quad = \quad \mathbf{{{format_angle_output(final_val, unit, latex=True)}}}"
                                        descr_text = f"Calcolo per differenza angolare (somma interna)."
                                        st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(final_val, unit)
                                    elif "arccos" in ans:
                                        # Carnot
                                        s_adj1 = float(st.session_state.solved_values[f"SegDist_{pt}_{nex_lbl}" if f"SegDist_{pt}_{nex_lbl}" in st.session_state.solved_values else f"SegDist_{nex_lbl}_{pt}"])
                                        s_adj2 = float(st.session_state.solved_values[f"SegDist_{pt}_{prev_lbl}" if f"SegDist_{pt}_{prev_lbl}" in st.session_state.solved_values else f"SegDist_{prev_lbl}_{pt}"])
                                        
                                        # Gestione Diagonale (Opposto)
                                        s_opp_key = f"SegDist_{prev_lbl}_{nex_lbl}" if f"SegDist_{prev_lbl}_{nex_lbl}" in st.session_state.solved_values else f"SegDist_{nex_lbl}_{prev_lbl}"
                                        # Ora garantito dalla logica di get_available_targets
                                        s_opp = float(st.session_state.solved_values[s_opp_key])
                                        diag_note = ""
                                        
                                        cos_val = (s_adj1**2 + s_adj2**2 - s_opp**2) / (2 * s_adj1 * s_adj2)
                                        cos_val = max(-1, min(1, cos_val)) # Clamp
                                        alpha_rad = math.acos(cos_val)
                                        
                                        st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(alpha_rad, unit)
                                        latex_msg = rf"{greek_symbol} = \arccos({cos_val:.4f}) = \mathbf{{{format_angle_output(alpha_rad, unit, latex=True)}}} {diag_note}"
                                        descr_text = "Calcolo angolo interno tramite Teorema del Coseno (Carnot) con triangolazione."
                                    elif "sin" in ans and "frac" in ans:
                                        # Seni
                                        opp_side_val = float(st.session_state.solved_values[f"SegDist_{prev_lbl}_{nex_lbl}" if f"SegDist_{prev_lbl}_{nex_lbl}" in st.session_state.solved_values else f"SegDist_{nex_lbl}_{prev_lbl}"])
                                        
                                        # Trova la coppia nota (Angolo, Lato Opposto)
                                        for other in pts_lbls:
                                            if other == pt: continue
                                            if f"Ang_{other}" in st.session_state.solved_values:
                                                o_idx = pts_lbls.index(other); o_prev = pts_lbls[o_idx-1]; o_next = pts_lbls[(o_idx+1)%3]
                                                o_opp_key = f"SegDist_{o_prev}_{o_next}" if f"SegDist_{o_prev}_{o_next}" in st.session_state.solved_values else f"SegDist_{o_next}_{o_prev}"
                                                if o_opp_key in st.session_state.solved_values:
                                                    # Ricalcola angolo 'other' dai punti per precisione (evita parsing stringa)
                                                    p_o, p_op, p_on = st.session_state.points[other], st.session_state.points[o_prev], st.session_state.points[o_next]
                                                    az_in_o = math.atan2(p_op['x']-p_o['x'], p_op['y']-p_o['y'])
                                                    az_out_o = math.atan2(p_on['x']-p_o['x'], p_on['y']-p_o['y'])
                                                    ang_other_rad = (az_out_o - az_in_o) % (2*math.pi)
                                                    
                                                    other_side_val = float(st.session_state.solved_values[o_opp_key])
                                                    sin_alpha = (opp_side_val * math.sin(ang_other_rad)) / other_side_val
                                                    sin_alpha = max(-1, min(1, sin_alpha))
                                                    alpha_rad = math.asin(sin_alpha)
                                                    
                                                    st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(alpha_rad, unit)
                                                    latex_msg = rf"\sin({greek_symbol}) = {sin_alpha:.4f} \Rightarrow {greek_symbol} = \mathbf{{{format_angle_output(alpha_rad, unit, latex=True)}}}"
                                                    descr_text = "Calcolo angolo interno tramite Teorema dei Seni."
                                                    break
                                    else:
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
                                            full_circle_str = "360^{\circ}" if unit == AngleUnit.DEG else "400^g"
                                            final_val = raw_diff + (2*math.pi)
                                            final_str = format_angle_output(final_val, unit, latex=True)
                                            
                                            latex_msg = rf"""\begin{{aligned}} {generic_formula} &= {numeric_formula} \\ &= {step1_str} \quad (\text{{Neg}}!) \\ &= {step1_str} + {full_circle_str} \\ &= \mathbf{{{final_str}}} \end{{aligned}}"""
                                            st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(final_val, unit)
                                        else:
                                            final_val = raw_diff
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
                                st.rerun()
                            else: st.error("Strategia errata.")
