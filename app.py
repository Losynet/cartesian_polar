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

def undo_last_action():
    if st.session_state.log:
        last = st.session_state.log.pop()
        for k in last.get('added_items', []): st.session_state.solved_items.discard(k)
        for k in last.get('added_values', []): st.session_state.solved_values.pop(k, None)
        st.session_state.last_calc_msg = None
        st.session_state.current_mission = None
        st.rerun()

def recalculate_points():
    st.session_state.points = {}
    st.session_state.solved_items = set()
    st.session_state.solved_values = {}
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

def draw_angle_wedge(fig, pt_lbl, color, radius):
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
        
        fig.add_trace(go.Scatter(x=wedge_x, y=wedge_y, fill="toself", fillcolor=color, line=dict(width=0), showlegend=False, hoverinfo='skip'))
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
    
    arc_x, arc_y = get_arc_path(p1['x'], p1['y'], eff_radius*0.7, 0, az_rad, 40)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color='#d35400', width=2), showlegend=False))
    
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
        for item in st.session_state.solved_items:
            if item.startswith("Ang_"):
                pt_lbl = item.split("_")[1]
                draw_angle_wedge(fig, pt_lbl, "rgba(0,128,0,0.15)", radius)

        # 2. Azimut (Smart Overlap)
        # Contatore per sapere quanti azimut partono da ogni punto
        azimuth_counts = defaultdict(int)
        
        for item in st.session_state.solved_items:
            if item.startswith("SegAz_"):
                parts = item.split("_")
                p1, p2 = parts[1], parts[2]
                
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
                p1 = st.session_state.points[parts[2]]; p2 = st.session_state.points[parts[3]]
                corner_x, corner_y = p2['x'], p1['y']
                
                # Triangolo Colorato
                fig.add_trace(go.Scatter(x=[p1['x'], corner_x, p2['x'], p1['x']], y=[p1['y'], corner_y, p2['y'], p1['y']], fill='toself', fillcolor='rgba(255, 235, 59, 0.3)', mode='none', showlegend=False))
                
                # Etichette Delta
                dx = p2['x'] - p1['x']; dy = p2['y'] - p1['y']
                fig.add_annotation(x=(p1['x']+corner_x)/2, y=p1['y'], text=f"ΔX={dx:.2f}", font=dict(color="green", size=10, weight="bold"), yshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                fig.add_annotation(x=corner_x, y=(p1['y']+p2['y'])/2, text=f"ΔY={dy:.2f}", font=dict(color="magenta", size=10, weight="bold"), xshift=15, showarrow=False, bgcolor="rgba(255,255,255,0.7)")
                
                # Cateti Tratteggiati
                fig.add_trace(go.Scatter(x=[p1['x'], p2['x']], y=[p1['y'], p2['y']], mode='lines', line=dict(color='black', width=3, dash='dash'), opacity=0.3, showlegend=False)) # Ipotenusa ghost
                fig.add_trace(go.Scatter(x=[p1['x'], corner_x, p2['x']], y=[p1['y'], corner_y, p2['y']], mode='lines', line=dict(color='gray', width=2, dash='dot'), showlegend=False)) # Cateti

            if "calc_ang" in st.session_state.current_mission:
                pt_lbl = st.session_state.current_mission.split("_")[2]
                pts_lbls = sorted(st.session_state.points.keys())
                idx = pts_lbls.index(pt_lbl)
                prev = pts_lbls[idx-1]; nex = pts_lbls[(idx+1)%len(pts_lbls)]
                # Visualizza i due azimut che generano l'angolo (In e Out)
                draw_azimuth_visuals(fig, pt_lbl, nex, radius, 0) # Out
                draw_azimuth_visuals(fig, pt_lbl, prev, radius, radius*0.15) # In (Back Azimuth)
                draw_angle_wedge(fig, pt_lbl, "rgba(255, 165, 0, 0.2)", radius)
                    
        # 4. Punti
        for lbl, p in st.session_state.points.items():
            pos = get_point_label_pos(p['x'], p['y'])
            fig.add_trace(go.Scatter(x=[p['x']], y=[p['y']], mode='markers+text', marker=dict(size=12, color='blue', line=dict(width=2, color='white')), text=[lbl], textposition=pos, textfont=dict(size=12, color="darkblue", weight="bold"), showlegend=False))
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
        for item in st.session_state.solved_items:
            if item.startswith("Seg_") and "SegDist" not in item:
                parts = item.split("_"); p1 = st.session_state.points[parts[1]]; p2 = st.session_state.points[parts[2]]
                fig.add_trace(go.Scatter(x=[p1['x'], p2['x']], y=[p1['y'], p2['y']], mode='lines', line=dict(color='green', width=2), showlegend=False))
                if f"SegDist_{parts[1]}_{parts[2]}" in st.session_state.solved_values:
                    mx, my = (p1['x']+p2['x'])/2, (p1['y']+p2['y'])/2
                    val = st.session_state.solved_values[f"SegDist_{parts[1]}_{parts[2]}"]
                    rot_angle = get_text_rotation(p1, p2)
                    fig.add_annotation(x=mx, y=my, text=val, textangle=rot_angle, font=dict(color="green", size=11, weight="bold"), bgcolor="white", borderpad=2, showarrow=False)

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
        if missing_cart or missing_pol:
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
                    opt_seg.append(f"Azimut ({p1}{p2})")

    # 3. Angoli (Logica Rigida: Azimut In/Out noti)
    opt_ang = []
    if len(pts_keys) >= 3:
        for p in pts_keys:
             if f"Ang_{p}" not in st.session_state.solved_items:
                 idx = pts_keys.index(p)
                 prev_p = pts_keys[idx-1]; next_p = pts_keys[(idx+1)%len(pts_keys)]
                 has_az_in = f"SegAz_{prev_p}_{p}" in st.session_state.solved_items or f"SegAz_{p}_{prev_p}" in st.session_state.solved_items
                 has_az_out = f"SegAz_{p}_{next_p}" in st.session_state.solved_items or f"SegAz_{next_p}_{p}" in st.session_state.solved_items
                 if has_az_in and has_az_out: opt_ang.append(f"Angolo interno in {p}")
    
    opt_poly = []
    if len(pts_keys) >= 3:
        known_angles = [x for x in st.session_state.solved_items if x.startswith("Ang_")]
        known_segments = [x for x in st.session_state.solved_items if x.startswith("SegDist_")]
        if len(known_segments) >= 2 and len(known_angles) >= 1:
            if "Area_Poly" not in st.session_state.solved_items: opt_poly.append("Intero Poligono")
        if "Perim_Poly" not in st.session_state.solved_items and len(known_segments) == len(pts_keys):
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
        seg_str = subject.split("(")[1].replace(")","")
        p1, p2 = seg_str[0], seg_str[1]
        return True, "OK", f"seg_az_{p1}_{p2}"
    return False, "Non disponibile", None

def get_strategies_for_mission(mission_code):
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
        q['correct'] = rf"$\overline{{{p1}{p2}}} = \sqrt{{\Delta X^2 + \Delta Y^2}}$"
        q['wrongs'] = [r"Media coordinate", r"Somma differenze"]
        q['latex'] = rf"\overline{{{p1}{p2}}} = \sqrt{{(X_{{{p2}}}-X_{{{p1}}})^2 + (Y_{{{p2}}}-Y_{{{p1}}})^2}}"
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
        
        # Strategie Multiple
        corrects = [rf"$\alpha_{{{pt}}} = ({pt}{nex}) - ({pt}{prev})$"]
        
        # Sottrazione (se triangolo e 2 angoli noti)
        if len(pts_lbls) == 3:
            known_angs = [k for k in st.session_state.solved_items if k.startswith("Ang_") and k != f"Ang_{pt}"]
            if len(known_angs) == 2:
                tot = "180^{\circ}" if st.session_state.input_interpretation == AngleUnit.DEG else "200^g"
                others = "+".join([rf"\alpha_{{{k.split('_')[1]}}}" for k in known_angs])
                corrects.append(rf"$\alpha_{{{pt}}} = {tot} - ({others})$")

        q['correct'] = corrects[0] # Default per compatibilità
        q['correct_list'] = corrects
        q['wrongs'] = [rf"$\alpha_{{{pt}}} = ({pt}{nex}) + ({pt}{prev})$"]
        q['latex'] = rf"\alpha_{{{pt}}} = ({pt}{nex}) - ({pt}{prev})"
        q['desc'] = f"Calcolo Angolo in {pt}"
    return q

with col_tutor:
    st.subheader("🧠 Tutor")
    if st.session_state.log:
        html_report = generate_html_report(st.session_state.log, st.session_state.solved_values)
        st.download_button("📄 Report", data=html_report, file_name="geosolver_report.html", mime="text/html")
        st.button("↩️ Annulla ultima operazione", on_click=undo_last_action)
    
    if st.session_state.last_calc_msg:
        st.success("✅ **Fatto:**")
        st.latex(st.session_state.last_calc_msg.replace("°", "^{\circ}"))

    if not st.session_state.points:
        st.info("Carica dati.")
    elif not st.session_state.projections_visible:
        st.warning("⚠️ **Analisi Preliminare**")
        st.button("Traccia Proiezioni (X, Y)", on_click=activate_projections)
    else:
        avail = get_available_targets()
        if not avail: 
            st.success("Hai completato l'esercizio!")
        else:
            sel_subj = st.selectbox("1. Calcola:", avail)
            possible_actions = []
            if "Punto" in sel_subj: 
                pt = sel_subj.split()[1]
                if f"X_{pt}" not in st.session_state.solved_items: possible_actions.append("Coord. X")
                if f"Y_{pt}" not in st.session_state.solved_items: possible_actions.append("Coord. Y")
                if f"Dist_{pt}" not in st.session_state.solved_items: possible_actions.append("Distanza (da O)")
                if f"Az_{pt}" not in st.session_state.solved_items: possible_actions.append("Azimut (da O)")
            elif "Poligono" in sel_subj: 
                if "Area_Poly" not in st.session_state.solved_items: possible_actions.append("Area")
                if "Perim_Poly" not in st.session_state.solved_items: possible_actions.append("Perimetro")
            elif "Angolo" in sel_subj: possible_actions = ["Valore Angolare"]
            else: possible_actions = ["Calcola"]
            
            if len(possible_actions) == 0: sel_act = None
            elif len(possible_actions) == 1: sel_act = possible_actions[0]; st.info(f"📍 Azione unica disponibile: **{sel_act}**")
            else: sel_act = st.radio("2. Grandezza:", possible_actions, horizontal=True, key=f"radio_{len(st.session_state.log)}")
            
            if st.button("Procedi"):
                if sel_act:
                    ok, msg, code = check_goal_feasibility(sel_subj, sel_act)
                    if ok:
                        st.session_state.current_mission = code
                        st.session_state.current_options = None 
                        st.rerun()

            if st.session_state.current_mission:
                st.divider()
                q = get_strategies_for_mission(st.session_state.current_mission)
                if not q:
                    st.error("Errore interno: Strategia non trovata.")
                    st.session_state.current_mission = None
                    st.rerun()

                if not st.session_state.current_options:
                    # Usa correct_list se esiste, altrimenti fallback su correct singolo
                    corrects = q.get('correct_list', [q['correct']])
                    all_opts = corrects + q['wrongs']
                    import random; random.shuffle(all_opts)
                    st.session_state.current_options = all_opts
                
                with st.form("strat"):
                    st.write("3. Strategia:")
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
                                st.session_state.solved_items.add(k); st.session_state.solved_values[k] = format_coord(val)
                                latex_msg = rf"\overline{{O{pt_lbl}}} = \sqrt{{{format_coord(pt['x'])}^2 + {format_coord(pt['y'])}^2}} \quad = \quad \mathbf{{{format_coord(val)}}}"
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
                                x1, y1 = st.session_state.points[p1]['x'], st.session_state.points[p1]['y']
                                x2, y2 = st.session_state.points[p2]['x'], st.session_state.points[p2]['y']
                                dx_val = x2 - x1; dy_val = y2 - y1
                                val = math.sqrt(dx_val**2 + dy_val**2)
                                k = f"SegDist_{p1}_{p2}"
                                st.session_state.solved_items.add(f"Seg_{p1}_{p2}"); st.session_state.solved_items.add(k); st.session_state.solved_values[k] = format_coord(val)
                                latex_msg = rf"""\begin{{aligned}}
\Delta X &= {format_coord(x2)} - {format_coord(x1)} = \mathbf{{{format_coord(dx_val)}}} \\
\Delta Y &= {format_coord(y2)} - {format_coord(y1)} = \mathbf{{{format_coord(dy_val)}}} \\
\overline{{{p1}{p2}}} &= \sqrt{{\Delta X^2 + \Delta Y^2}} \\
&= \sqrt{{({format_coord(dx_val)})^2 + ({format_coord(dy_val)})^2}} \\
&= \mathbf{{{format_coord(val)}}}
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
                                prev = st.session_state.points[pts_lbls[idx-1]]
                                curr = st.session_state.points[pt]
                                nex = st.session_state.points[pts_lbls[(idx+1)%len(pts_lbls)]]
                                
                                # Gestione Metodo Sottrazione
                                if "180" in ans or "200" in ans:
                                    az_in = math.atan2(prev['x']-curr['x'], prev['y']-curr['y'])
                                    az_out = math.atan2(nex['x']-curr['x'], nex['y']-curr['y'])
                                    final_val = (az_out - az_in) % (2*math.pi)
                                    tot_str = "180^{\circ}" if unit == AngleUnit.DEG else "200^g"
                                    latex_msg = rf"\alpha_{{{pt}}} = {tot_str} - (\alpha_{{{pts_lbls[idx-1]}}} + \alpha_{{{pts_lbls[(idx+1)%len(pts_lbls)]}}}) \quad = \quad \mathbf{{{format_angle_output(final_val, unit, latex=True)}}}"
                                    descr_text = f"Calcolo per differenza angolare (somma interna)."
                                    st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(final_val, unit)
                                else:
                                    # Metodo Azimut
                                    az_in = math.atan2(prev['x']-curr['x'], prev['y']-curr['y'])
                                    az_out = math.atan2(nex['x']-curr['x'], nex['y']-curr['y'])
                                    raw_diff = az_out - az_in
                                    descr_text = f"Angolo in {pt}: differenza Azimut Uscita ({pt}{pts_lbls[(idx+1)%len(pts_lbls)]}) - Azimut Entrata ({pt}{pts_lbls[idx-1]})."
                                    if raw_diff < 0:
                                        step1_str = format_angle_output(raw_diff, unit, latex=True)
                                        step2_str = "360^\circ" if unit == AngleUnit.DEG else "400^g"
                                        final_val = raw_diff + (2*math.pi)
                                        final_str = format_angle_output(final_val, unit, latex=True)
                                        latex_msg = rf"\alpha_{{{pt}}} = {format_angle_output(az_out, unit, True)} - {format_angle_output(az_in, unit, True)} = {step1_str} \quad (\text{{Neg}}!) \quad \Rightarrow \quad {step1_str} + {step2_str} \quad = \quad \mathbf{{{final_str}}}"
                                        st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(final_val, unit)
                                    else:
                                        final_val = raw_diff
                                        st.session_state.solved_values[f"Ang_{pt}"] = format_angle_output(final_val, unit)
                                        latex_msg = rf"\alpha_{{{pt}}} = {format_angle_output(az_out, unit, True)} - {format_angle_output(az_in, unit, True)} \quad = \quad \mathbf{{{format_angle_output(final_val, unit, latex=True)}}}"

                            # Calcola differenze per Undo
                            added_items = list(st.session_state.solved_items - pre_items)
                            added_values = list(set(st.session_state.solved_values.keys()) - pre_values)
                            
                            st.session_state.log.append({'action': q['desc'], 'method': ans, 'result': latex_msg, 'desc_verbose': descr_text, 'added_items': added_items, 'added_values': added_values})
                            st.session_state.last_calc_msg = latex_msg
                            st.session_state.current_mission = None
                            st.rerun()
                        else: st.error("Strategia errata.")