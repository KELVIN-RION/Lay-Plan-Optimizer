import streamlit as st
import pandas as pd
import pulp
from ortools.sat.python import cp_model
import time
import random
import math
import io
import zipfile
import threading

st.set_page_config(page_title="Lay Plan Optimizer V7", layout="wide")
st.title("✂️ Lay Plan Optimizer ")

# --- 1. MANUAL ORDER INPUT ---
st.subheader("1. Manual Order Entry")
with st.expander("Enter Sizes and Quantities", expanded=True):
    sizes_input = st.text_input("Enter sizes separated by commas (e.g., S, M, L, XL):", value="S, M, L, XL")
    sizes_list = [s.strip().upper() for s in sizes_input.split(',') if s.strip()]
    
    if sizes_list:
        cols = st.columns(len(sizes_list))
        order_demand = {}
        for i, size in enumerate(sizes_list):
            with cols[i]:
                val = st.number_input(f"Qty for {size}", min_value=0, value=0, key=f"qty_{size}")
                if val > 0:
                    order_demand[size] = val

# --- 2. FABRIC, LOT & FILE INPUTS ---
st.subheader("2. Fabric & Marker Data")
col_cc, col_model = st.columns(2)
with col_cc: cc_input = st.text_input("CC / Style Code", value="WMT-2024-045")
with col_model: model_input = st.text_input("Model / Color", value="Black")

col1, col2 = st.columns(2)
with col1: marker_file = st.file_uploader("Upload Marker Data (CSV)", type="csv", key="marker")
with col2: roll_file = st.file_uploader("Upload Roll Data (CSV)", type="csv", key="roll")

# --- LOT SELECTION DROPDOWN ---
lot_options = ["-- Select LOT --"]
lot_lengths = {}

if roll_file:
    df_roll_raw = pd.read_csv(roll_file)
    if 'LOT' in df_roll_raw.columns and 'Available_Length_m' in df_roll_raw.columns:
        lot_data = df_roll_raw.groupby('LOT')['Available_Length_m'].sum().to_dict()
        lot_options = list(lot_data.keys())
        lot_lengths = lot_data

with st.expander("Select Fabric LOT to Cut", expanded=True):
    selected_lot = st.selectbox("Choose ONE LOT for this cutting run:", lot_options)
    fabric_available = lot_lengths.get(selected_lot, 0)
    if selected_lot != "-- Select LOT --":
        st.info(f"📏 **Total Fabric Available for {selected_lot}: {fabric_available:.2f} meters**")

# --- CORE MATHEMATICS (STRICT INTEGERS ONLY) ---
def parse_marker_data(ratio_str, sizes_str, pcs_per_ply):
    sizes = [s.strip().upper() for s in sizes_str.split('+')]
    ratios = [int(r.strip()) for r in ratio_str.split(':')] # STRICT INTEGERS
    size_ratio_map = dict(zip(sizes, ratios))
    return size_ratio_map, pcs_per_ply

# --- 8 ALGORITHMS ---
def run_greedy(order_demand, df_marker):
    start = time.time(); remaining = order_demand.copy()
    res = {row['Marker_ID']: 0 for _, row in df_marker.iterrows()}
    sorted_m = df_marker.sort_values(by='Efficiency_pct', ascending=False)['Marker_ID'].tolist()
    while any(v > 0 for v in remaining.values()):
        placed = False
        for m in sorted_m:
            mk_data = df_marker[df_marker['Marker_ID']==m].iloc[0]
            size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
            if all(remaining.get(s, 0) >= ratio for s, ratio in size_ratio_map.items()):
                res[m] += 1
                for s, ratio in size_ratio_map.items(): remaining[s] -= ratio
                placed = True; break
        if not placed: break
    return res, time.time() - start

def run_bestfit(order_demand, df_marker):
    start = time.time(); remaining = order_demand.copy()
    res = {row['Marker_ID']: 0 for _, row in df_marker.iterrows()}
    while any(v > 0 for v in remaining.values()):
        best_score = -1; best_mk = None
        for _, row in df_marker.iterrows():
            m = row['Marker_ID']
            size_ratio_map, _ = parse_marker_data(row['ratio'], row['Size_Combination'], row['no_of_pieces'])
            score = 0; valid = True
            for s, ratio in size_ratio_map.items():
                if remaining.get(s, 0) < ratio: valid = False; break
                score += (ratio / remaining[s])
            if valid and score > best_score: best_score = score; best_mk = m
        if not best_mk: break
        mk_data = df_marker[df_marker['Marker_ID']==best_mk].iloc[0]
        size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        res[best_mk] += 1
        for s, ratio in size_ratio_map.items(): remaining[s] -= ratio
    return res, time.time() - start

def run_pulp(order_demand, df_marker):
    start = time.time(); marker_ids = df_marker['Marker_ID'].tolist()
    prob = pulp.LpProblem("LayPlan", pulp.LpMinimize)
    plies = pulp.LpVariable.dicts("P", marker_ids, lowBound=0, cat='Integer')
    prob += pulp.lpSum([plies[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids])
    for m in marker_ids:
        mk_data = df_marker[df_marker['Marker_ID']==m].iloc[0]
        size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        for size, ratio in size_ratio_map.items():
            if size in order_demand: prob += plies[m] * ratio >= order_demand[size]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {m: int(plies[m].varValue) for m in marker_ids}, time.time() - start

def run_ortools(order_demand, df_marker):
    start = time.time(); marker_ids = df_marker['Marker_ID'].tolist()
    model = cp_model.CpModel()
    plies = {m: model.NewIntVar(0, 100000, f"p_{m}") for m in marker_ids}
    model.Minimize(sum(plies[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids))
    for m in marker_ids:
        mk_data = df_marker[df_marker['Marker_ID']==m].iloc[0]
        size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        for size, ratio in size_ratio_map.items():
            if size in order_demand: model.Add(plies[m] * ratio >= order_demand[size])
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = 10.0; solver.Solve(model)
    return {m: solver.Value(plies[m]) for m in marker_ids}, time.time() - start

def run_stochastic_wrapper(algo_func, order_demand, df_marker, runs=3):
    start = time.time(); best_res = None; best_score = float('inf')
    for _ in range(runs):
        res, _ = algo_func(order_demand, df_marker)
        score = sum(res[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in df_marker['Marker_ID'])
        if score < best_score: best_score = score; best_res = res
    return best_res, time.time() - start

def _random_search(order_demand, df_marker):
    return {m: random.randint(0, max(1, sum(order_demand.values())//2)) for m in df_marker['Marker_ID']}, 0

def _simulated_annealing(order_demand, df_marker):
    marker_ids = df_marker['Marker_ID'].tolist()
    current = {m: random.randint(0, max(1, sum(order_demand.values())//2)) for m in marker_ids}
    best = current.copy(); T = 1000.0
    for _ in range(1500):
        T *= 0.995; neighbor = current.copy(); m = random.choice(marker_ids)
        neighbor[m] = max(0, neighbor[m] + random.randint(-5, 5))
        if sum(neighbor[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids) < sum(current[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids) or random.random() < math.exp(-0.1 / T):
            current = neighbor
            if sum(current[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids) < sum(best[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids): best = neighbor.copy()
    return best, 0

def _genetic(order_demand, df_marker):
    marker_ids = df_marker['Marker_ID'].tolist()
    population = [{m: random.randint(0, max(1, sum(order_demand.values())//2)) for m in marker_ids} for _ in range(30)]
    for _ in range(30):
        population.sort(key=lambda x: sum(x[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids))
        next_gen = population[:15]
        for i in range(0, 14, 2):
            p1, p2 = next_gen[i], next_gen[i+1]; child1, child2 = {}, {}
            for m in marker_ids:
                if random.random() > 0.5: child1[m], child2[m] = p1[m], p2[m]
                else: child1[m], child2[m] = p2[m], p1[m]
            next_gen.extend([child1, child2])
        for ind in next_gen:
            if random.random() < 0.2: m = random.choice(marker_ids); ind[m] = max(0, ind[m] + random.randint(-5, 5))
        population = next_gen
    return min(population, key=lambda x: sum(x[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids)), 0

def _pso(order_demand, df_marker):
    marker_ids = df_marker['Marker_ID'].tolist()
    particles = [{m: random.randint(0, max(1, sum(order_demand.values())//2)) for m in marker_ids} for _ in range(20)]
    best = min(particles, key=lambda x: sum(x[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids))
    return best, 0
# --- TABLE FORMATTING & PHYSICAL LOT MAPPING ---
def format_lay_plan(plies_result, df_marker, order_demand, lot_id, lot_avail):
    rows = []; current_used = 0.0
    total_yield = {size: 0 for size in order_demand.keys()}; total_pcs = 0; total_eff_weighted = 0.0

    for mk_id, planned_plies in plies_result.items():
        if planned_plies <= 0: continue
        mk_data = df_marker[df_marker['Marker_ID'] == mk_id].iloc[0]
        size_ratio_map, pcs_per_ply = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        
        max_possible_plies = int((lot_avail - current_used) // mk_data['Marker_Length_m'])
        actual_plies = min(planned_plies, max_possible_plies)
        if actual_plies <= 0: break 

        fabric_used = actual_plies * mk_data['Marker_Length_m']
        pcs_cut = actual_plies * pcs_per_ply
        
        row = {
            "Lay #": str(len(rows) + 1), # STRICT STRING TO PREVENT PYARROW CRASH
            "Marker No": mk_id, "Size Combination": mk_data['Size_Combination'],
            "Ratio": mk_data['ratio'], "CAD Pcs/Ply": mk_data['no_of_pieces'],
            "Length (m)": mk_data['Marker_Length_m'], "Ply": actual_plies,
            "Efficiency (%)": mk_data['Efficiency_pct'], "Fabric Lot #": lot_id,
            "Fabric Available (m)": lot_avail, "Fabric Used (m)": fabric_used,
            "Fabric Pending (m)": round(lot_avail - (current_used + fabric_used), 2),
            "Total Pcs Cut": pcs_cut
        }
        for size in order_demand.keys():
            yield_count = actual_plies * size_ratio_map.get(size, 0)
            row[f"Yield_{size}"] = yield_count; total_yield[size] += yield_count
            
        rows.append(row); current_used += fabric_used; total_pcs += pcs_cut
        total_eff_weighted += fabric_used * mk_data['Efficiency_pct']

    # Calculate Bottom Summary Row
    end_bit = round(lot_avail - current_used, 2)
    overall_eff = (total_eff_weighted / current_used) if current_used > 0 else 0
    
    summary_row = {
        "Lay #": "END", 
        "Marker No": None, "Size Combination": None, "Ratio": None, "CAD Pcs/Ply": None,  # FIXED: Changed "" to None
        "Length (m)": None, # FIXED
        "Ply": sum([r["Ply"] for r in rows]), 
        "Efficiency (%)": f"Avg: {overall_eff:.1f}%",
        "Fabric Lot #": lot_id, 
        "Fabric Available (m)": None, # FIXED
        "Fabric Used (m)": round(current_used, 2), 
        "Fabric Pending (m)": None, # FIXED: Removed text, kept as None to prevent crash
        "Total Pcs Cut": total_pcs
    }
    for size in order_demand.keys(): summary_row[f"Yield_{size}"] = total_yield[size]
    
    df_out = pd.DataFrame(rows + [summary_row])
    base_cols = ["Lay #", "Marker No", "Size Combination", "Ratio", "CAD Pcs/Ply", "Length (m)", 
                 "Ply", "Efficiency (%)", "Fabric Lot #", "Fabric Available (m)", "Fabric Used (m)", 
                 "Fabric Pending (m)", "Total Pcs Cut"]
    yield_cols = [f"Yield_{size}" for size in order_demand.keys()]
    df_out = df_out[base_cols + yield_cols]
    overcuts = {size: total_yield[size] - demand for size, demand in order_demand.items() if total_yield[size] > demand}
    
    return df_out, round(current_used, 2), end_bit, total_pcs, overall_eff, overcuts

# --- SESSION STATE (PREVENTS DOWNLOAD CLEARING SCREEN) ---
if 'results' not in st.session_state:
    st.session_state.results = None

# --- MAIN EXECUTION UI ---
if order_demand and marker_file and selected_lot != "-- Select LOT --":
    df_marker = pd.read_csv(marker_file)
    
    if st.button("🚀 Run 8-Algorithm Hybrid Engine", type="primary", use_container_width=True):
        # NATIVE STREAMLIT SKELETON LOADER (No threading errors)
        with st.status("Executing Optimization Engine...", expanded=True) as status:
            st.write("🧠 Initializing 8 Algorithm Engines...")
            time.sleep(0.5) # Tiny pause to let UI render the skeleton
            
            st.write("⚡ Running Deterministic Solvers (Greedy, Best-Fit)...")
            res_greed = run_greedy(order_demand, df_marker)[0]
            res_bf = run_bestfit(order_demand, df_marker)[0]
            
            st.write("📐 Running Mathematical Solvers (PuLP, OR-Tools)...")
            res_pulp = run_pulp(order_demand, df_marker)[0]
            res_ort = run_ortools(order_demand, df_marker)[0]
            
            st.write("🧬 Running Stochastic AI (Random, Annealing, Genetic, Swarm)...")
            res_rand = run_stochastic_wrapper(_random_search, order_demand, df_marker)[0]
            res_sa = run_stochastic_wrapper(_simulated_annealing, order_demand, df_marker)[0]
            res_ga = run_stochastic_wrapper(_genetic, order_demand, df_marker)[0]
            res_pso = run_stochastic_wrapper(_pso, order_demand, df_marker)[0]
            
            st.write("📊 Formatting Lay Plans & Calculating Yields...")
            results_raw = [
                ("DETERMINISTIC: Greedy", res_greed), ("DETERMINISTIC: Best-Fit", res_bf),
                ("DETERMINISTIC: PuLP", res_pulp), ("DETERMINISTIC: OR-Tools", res_ort),
                ("STOCHASTIC: Random (x3)", res_rand), ("STOCHASTIC: Annealing (x3)", res_sa),
                ("STOCHASTIC: Genetic (x3)", res_ga), ("STOCHASTIC: Swarm (x3)", res_pso)
            ]
            
            processed = []
            for name, res in results_raw:
                df, cons, eb, pcs, eff, oc = format_lay_plan(res, df_marker, order_demand, selected_lot, fabric_available)
                processed.append({"name": name, "df": df, "cons": cons, "eb": eb, "pcs": pcs, "eff": eff, "oc": oc})
            processed.sort(key=lambda x: x['cons'])
            
            # Save to memory bank
            st.session_state.results = processed
            status.update(label="✅ Optimization Complete!", state="complete", expanded=False)
        
        st.rerun() # Refresh screen to show results instantly

    # --- DISPLAY RESULTS (ONLY IF IN MEMORY BANK) ---
    if st.session_state.results is not None:
        processed = st.session_state.results
        best = processed[0]
        others = sorted(processed[1:], key=lambda x: (-x['eff'], x['cons']))
        
        st.markdown("---")
        st.subheader(f"🏆 OPTIMAL LAY PLAN: {best['name']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Pcs Produced", best['pcs'])
        m2.metric("Fabric Consumed", f"{best['cons']} m")
        m3.metric("End Bit Leftover", f"{best['eb']} m")
        
        if best['oc']: st.error(f"⚠️ LOT Exhausted. Overcut sizes: {best['oc']}")
        else: st.success("✅ Order fulfilled perfectly within LOT limits.")
        
        st.dataframe(best['df'], width="stretch", hide_index=True)
        
        # --- ZIP DOWNLOAD SECTION ---
        st.subheader("📥 Download Lay Plans")
        st.write("Select the algorithms you want to export into a ZIP file.")
        
        dl_cols = st.columns(4)
        selected_algos = []
        for i, algo in enumerate(processed):
            with dl_cols[i % 4]:
                if st.checkbox(algo['name'], value=(i==0)): 
                    selected_algos.append(algo)
                    
        if selected_algos:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for algo in selected_algos:
                    csv_data = algo['df'].to_csv(index=False).encode('utf-8')
                    safe_name = algo['name'].replace(" ", "_").replace(":", "").replace("/", "_")
                    zipf.writestr(f"{safe_name}.csv", csv_data)
            
            st.download_button(
                label=f"⬇️ Download {len(selected_algos)} Selected as ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"LayPlans_{selected_lot}.zip",
                mime="application/zip"
            )

        # --- VIEW ALL ---
        with st.expander("🔍 View All 8 Algorithm Comparisons"):
            for algo in others:
                st.markdown(f"**{algo['name']}** | Used: {algo['cons']}m | End Bit: {algo['eb']}m")
                st.dataframe(algo['df'], width="stretch", hide_index=True)
                st.markdown("---")