import streamlit as st
import pandas as pd
import pulp
from ortools.sat.python import cp_model
import time
import random
import math
import io
import zipfile

st.set_page_config(page_title="Lay Plan Optimizer V8", layout="wide")
st.title("✂️ Lay Plan Optimizer")

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

# --- 2. FILE INPUTS ---
st.subheader("2. Fabric & Marker Data")
col_cc, col_model = st.columns(2)
with col_cc: cc_input = st.text_input("CC / Style Code", value="WMT-2024-045")
with col_model: model_input = st.text_input("Model / Color", value="Black")

col1, col2 = st.columns(2)
with col1: marker_file = st.file_uploader("Upload Marker Data (CSV)", type="csv", key="marker")
with col2: roll_file = st.file_uploader("Upload Roll Data (CSV)", type="csv", key="roll")

# --- AUTO LOT GROUPING ---
lot_dict = {}
if roll_file:
    df_roll_raw = pd.read_csv(roll_file)
    if 'LOT' in df_roll_raw.columns and 'Available_Length_m' in df_roll_raw.columns:
        lot_data = df_roll_raw.groupby('LOT')['Available_Length_m'].sum().to_dict()
        lot_dict = dict(sorted(lot_data.items(), key=lambda item: item[1], reverse=True))
        st.success(f"🧠 Auto-Detected {len(lot_dict)} LOTs. Engine will optimize sequentially.")

# --- CORE MATHEMATICS ---
def parse_marker_data(ratio_str, sizes_str, pcs_per_ply):
    sizes = [s.strip().upper() for s in sizes_str.split('+')]
    ratios = [int(r.strip()) for r in ratio_str.split(':')]
    size_ratio_map = dict(zip(sizes, ratios))
    return size_ratio_map, pcs_per_ply

# --- 8 ALGORITHMS ---
def run_greedy(order_demand, df_marker, lot_length):
    start = time.time(); remaining = order_demand.copy(); current_used = 0.0
    res = {row['Marker_ID']: 0 for _, row in df_marker.iterrows()}
    sorted_m = df_marker.sort_values(by='Efficiency_pct', ascending=False)['Marker_ID'].tolist()
    while any(v > 0 for v in remaining.values()):
        placed = False
        for m in sorted_m:
            mk_data = df_marker[df_marker['Marker_ID']==m].iloc[0]
            size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
            if all(remaining.get(s, 0) >= ratio for s, ratio in size_ratio_map.items()):
                if current_used + mk_data['Marker_Length_m'] <= lot_length:
                    res[m] += 1; current_used += mk_data['Marker_Length_m']
                    for s, ratio in size_ratio_map.items(): remaining[s] -= ratio
                    placed = True; break
        if not placed: break
    return res, time.time() - start

def run_bestfit(order_demand, df_marker, lot_length):
    start = time.time(); remaining = order_demand.copy(); current_used = 0.0
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
            if valid and score > best_score and (current_used + row['Marker_Length_m'] <= lot_length): 
                best_score = score; best_mk = m
        if not best_mk: break
        mk_data = df_marker[df_marker['Marker_ID']==best_mk].iloc[0]
        size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        res[best_mk] += 1; current_used += mk_data['Marker_Length_m']
        for s, ratio in size_ratio_map.items(): remaining[s] -= ratio
    return res, time.time() - start

def run_pulp(order_demand, df_marker, lot_length):
    start = time.time(); marker_ids = df_marker['Marker_ID'].tolist()
    prob = pulp.LpProblem("LayPlan", pulp.LpMinimize)
    plies = pulp.LpVariable.dicts("P", marker_ids, lowBound=0, cat='Integer')
    prob += pulp.lpSum([plies[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids])
    prob += pulp.lpSum([plies[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids]) <= lot_length
    for m in marker_ids:
        mk_data = df_marker[df_marker['Marker_ID']==m].iloc[0]
        size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        for size, ratio in size_ratio_map.items():
            if size in order_demand: prob += plies[m] * ratio >= order_demand[size]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {m: int(plies[m].varValue) for m in marker_ids}, time.time() - start

def run_ortools(order_demand, df_marker, lot_length):
    start = time.time(); marker_ids = df_marker['Marker_ID'].tolist()
    model = cp_model.CpModel()
    plies = {m: model.NewIntVar(0, 100000, f"p_{m}") for m in marker_ids}
    scale = 100 
    scaled_lengths = {m: int(df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] * scale) for m in marker_ids}
    model.Minimize(sum(plies[m] * scaled_lengths[m] for m in marker_ids))
    model.Add(sum(plies[m] * scaled_lengths[m] for m in marker_ids) <= int(lot_length * scale))
    for m in marker_ids:
        mk_data = df_marker[df_marker['Marker_ID']==m].iloc[0]
        size_ratio_map, _ = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        for size, ratio in size_ratio_map.items():
            if size in order_demand: model.Add(plies[m] * ratio >= order_demand[size])
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = 10.0; solver.Solve(model)
    return {m: solver.Value(plies[m]) for m in marker_ids}, time.time() - start

def run_stochastic_wrapper(algo_func, order_demand, df_marker, lot_length, runs=3):
    start = time.time(); best_res = None; best_score = float('inf')
    for _ in range(runs):
        res, _ = algo_func(order_demand, df_marker, lot_length)
        used = sum(res[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in df_marker['Marker_ID'])
        score = used + (10000 if used > lot_length else 0) 
        if score < best_score: best_score = score; best_res = res
    return best_res, time.time() - start

def _random_search(order_demand, df_marker, lot_length):
    return {m: random.randint(0, max(1, int(lot_length // 2))) for m in df_marker['Marker_ID']}, 0

def _simulated_annealing(order_demand, df_marker, lot_length):
    marker_ids = df_marker['Marker_ID'].tolist()
    current = {m: random.randint(0, max(1, int(lot_length // 2))) for m in marker_ids}
    best = current.copy(); T = 1000.0
    for _ in range(1500):
        T *= 0.995; neighbor = current.copy(); m = random.choice(marker_ids)
        neighbor[m] = max(0, neighbor[m] + random.randint(-5, 5))
        curr_used = sum(current[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids)
        next_used = sum(neighbor[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids)
        best_used = sum(best[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids)
        if (next_used <= lot_length and next_used < curr_used) or (next_used <= lot_length and random.random() < math.exp(-0.1 / T)):
            current = neighbor
            if next_used <= lot_length and next_used < best_used: best = neighbor.copy()
    return best, 0

def _genetic(order_demand, df_marker, lot_length):
    marker_ids = df_marker['Marker_ID'].tolist()
    population = [{m: random.randint(0, max(1, int(lot_length // 2))) for m in marker_ids} for _ in range(30)]
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
    valid_pop = [p for p in population if sum(p[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids) <= lot_length]
    return min(valid_pop, key=lambda x: sum(x[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids)) if valid_pop else population[0], 0

def _pso(order_demand, df_marker, lot_length):
    marker_ids = df_marker['Marker_ID'].tolist()
    particles = [{m: random.randint(0, max(1, int(lot_length // 2))) for m in marker_ids} for _ in range(20)]
    valid_pop = [p for p in particles if sum(p[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids) <= lot_length]
    return min(valid_pop, key=lambda x: sum(x[m] * df_marker[df_marker['Marker_ID']==m].iloc[0]['Marker_Length_m'] for m in marker_ids)) if valid_pop else particles[0], 0

    # --- TABLE FORMATTING ---
def format_lot_plan(plies_result, df_marker, remaining_order, lot_id, lot_avail, lay_start_num):
    rows = []; current_used = 0.0
    total_yield = {size: 0 for size in remaining_order.keys()}; total_pcs = 0; total_eff_weighted = 0.0

    for mk_id, planned_plies in plies_result.items():
        if planned_plies <= 0: continue
        mk_data = df_marker[df_marker['Marker_ID'] == mk_id].iloc[0]
        size_ratio_map, pcs_per_ply = parse_marker_data(mk_data['ratio'], mk_data['Size_Combination'], mk_data['no_of_pieces'])
        
        max_possible_plies = int((lot_avail - current_used) // mk_data['Marker_Length_m'])
        actual_plies = min(planned_plies, max_possible_plies)
        if actual_plies <= 0: break 

        fabric_used = actual_plies * mk_data['Marker_Length_m']
        pcs_cut = actual_plies * pcs_per_ply
        cons_per_garment = mk_data['Marker_Length_m'] / mk_data['no_of_pieces']
        
        row = {
            "Lay #": str(lay_start_num + len(rows) + 1), 
            "Marker No": mk_id, "Size Combination": mk_data['Size_Combination'],
            "Ratio": mk_data['ratio'], "CAD Pcs/Ply": mk_data['no_of_pieces'],
            "Cons/Garment (m)": round(cons_per_garment, 3), 
            "Length (m)": mk_data['Marker_Length_m'], "Ply": actual_plies,
            "Efficiency (%)": mk_data['Efficiency_pct'], "Fabric Lot #": lot_id,
            "Fabric Available (m)": round(lot_avail - current_used, 2), 
            "Fabric Used (m)": round(fabric_used, 2), "Total Pcs Cut": pcs_cut,
            "End bit (m)": "" # Empty string for regular rows prevents PyArrow crash
        }
        for size in remaining_order.keys():
            yield_count = actual_plies * size_ratio_map.get(size, 0)
            row[f"Yield_{size}"] = yield_count; total_yield[size] += yield_count
            
        rows.append(row); current_used += fabric_used; total_pcs += pcs_cut
        total_eff_weighted += fabric_used * mk_data['Efficiency_pct']

    end_bit = round(lot_avail - current_used, 2)
    overall_eff = (total_eff_weighted / current_used) if current_used > 0 else 0
    
    summary_row = {
        "Lay #": "END", "Marker No": None, "Size Combination": None, "Ratio": None, "CAD Pcs/Ply": None,
        "Cons/Garment (m)": None, "Length (m)": None, "Ply": sum([r["Ply"] for r in rows]), 
        "Efficiency (%)": f"Avg: {overall_eff:.1f}%", "Fabric Lot #": lot_id, 
        "Fabric Available (m)": None, "Fabric Used (m)": round(current_used, 2), 
        "Total Pcs Cut": total_pcs, "End bit (m)": f"{end_bit}" # String value for PyArrow
    }
    for size in remaining_order.keys(): summary_row[f"Yield_{size}"] = total_yield[size]
    
    df_out = pd.DataFrame(rows + [summary_row])
    base_cols = ["Lay #", "Marker No", "Size Combination", "Ratio", "CAD Pcs/Ply", "Cons/Garment (m)", 
                 "Length (m)", "Ply", "Efficiency (%)", "Fabric Lot #", "Fabric Available (m)", 
                 "Fabric Used (m)", "Total Pcs Cut", "End bit (m)"]
    yield_cols = [f"Yield_{size}" for size in remaining_order.keys()]
    df_out = df_out[base_cols + yield_cols]
    
    return df_out, round(current_used, 2), end_bit, total_pcs, total_yield

# --- SESSION STATE ---
if 'results' not in st.session_state:
    st.session_state.results = None

# --- MAIN EXECUTION UI ---
if order_demand and marker_file and lot_dict:
    df_marker = pd.read_csv(marker_file)
    
    if st.button("🚀 Run Sequential LOT Engine", type="primary", use_container_width=True):
        with st.status("Executing 8 Algorithms Sequentially across LOTs...", expanded=True) as status:
            
            algo_functions = [
                ("DETERMINISTIC: Greedy", run_greedy), 
                ("DETERMINISTIC: Best-Fit", run_bestfit),
                ("DETERMINISTIC: PuLP", run_pulp), 
                ("DETERMINISTIC: OR-Tools", run_ortools),
                ("STOCHASTIC: Random (x3)", lambda o, m, l: run_stochastic_wrapper(_random_search, o, m, l)[0]), 
                ("STOCHASTIC: Annealing (x3)", lambda o, m, l: run_stochastic_wrapper(_simulated_annealing, o, m, l)[0]),
                ("STOCHASTIC: Genetic (x3)", lambda o, m, l: run_stochastic_wrapper(_genetic, o, m, l)[0]), 
                ("STOCHASTIC: Swarm (x3)", lambda o, m, l: run_stochastic_wrapper(_pso, o, m, l)[0])
            ]
            
            processed_algos = []
            
            for algo_name, algo_func in algo_functions:
                st.write(f"🧠 Running {algo_name}...")
                current_order = order_demand.copy()
                master_df_list = []
                total_global_cons = 0.0
                total_global_pcs = 0
                lay_counter = 0
                end_bits_summary = {}

                for lot_id, lot_length in lot_dict.items():
                    if not any(v > 0 for v in current_order.values()): break

                    res = algo_func(current_order, df_marker, lot_length)
                    if isinstance(res, tuple): res = res[0] # FIX: Tuple unpacking safety
                    
                    df_lot, cons, eb, pcs, yield_map = format_lot_plan(res, df_marker, current_order, lot_id, lot_length, lay_counter)
                    
                    master_df_list.append(df_lot)
                    total_global_cons += cons
                    total_global_pcs += pcs
                    lay_counter += len(df_lot) - 1 
                    end_bits_summary[lot_id] = eb
                    
                    for size, demand in current_order.items():
                        current_order[size] = max(0, demand - yield_map.get(size, 0))

                if master_df_list:
                    final_master_df = pd.concat(master_df_list, ignore_index=True)
                    overall_cons = (total_global_cons / total_global_pcs) if total_global_pcs > 0 else 0
                    
                    processed_algos.append({
                        "name": algo_name, "df": final_master_df, "total_cons": total_global_cons,
                        "total_pcs": total_global_pcs, "overall_cons": overall_cons,
                        "end_bits": end_bits_summary, "unfulfilled": {s: d for s, d in current_order.items() if d > 0}
                    })

            processed_algos.sort(key=lambda x: x['total_cons'])
            st.session_state.results = processed_algos
            status.update(label="✅ Optimization Complete!", state="complete", expanded=False)
        
        st.rerun()

    # --- DISPLAY RESULTS ---
    if st.session_state.results is not None:
        processed = st.session_state.results
        best = processed[0]
        others = sorted(processed[1:], key=lambda x: -x['overall_cons'])
        
        st.markdown("---")
        st.subheader(f"🏆 OPTIMAL LAY PLAN: {best['name']}")
        
        # Top Level Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Garments Produced", best['total_pcs'])
        m2.metric("Total Fabric Consumed", f"{best['total_cons']:.2f} m")
        m3.metric("Overall Consumption", f"{best['overall_cons']:.3f} m/garment")
        eb_str = ", ".join([f"{lot}: {eb}m" for lot, eb in best['end_bits'].items()])
        m4.metric("Total End-Bits Generated", eb_str)
        
        if best['unfulfilled']: st.error(f"⚠️ FABRIC EXHAUSTED. Unfulfilled sizes: {best['unfulfilled']}")
        else: st.success("✅ Order fulfilled perfectly across LOTs.")
        
        st.dataframe(best['df'], width="stretch", hide_index=True)
        
        # --- ZIP DOWNLOAD ---
        st.subheader("📥 Download Lay Plans")
        st.write("Select algorithms to export.")
        dl_cols = st.columns(4); selected_algos = []
        for i, algo in enumerate(processed):
            with dl_cols[i % 4]:
                if st.checkbox(algo['name'], value=(i==0)): selected_algos.append(algo)
                    
        if selected_algos:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for algo in selected_algos:
                    csv_data = algo['df'].to_csv(index=False).encode('utf-8')
                    safe_name = algo['name'].replace(" ", "_").replace(":", "").replace("/", "_")
                    zipf.writestr(f"{safe_name}.csv", csv_data)
            st.download_button(label=f"⬇️ Download {len(selected_algos)} Selected as ZIP", data=zip_buffer.getvalue(), file_name="Master_LayPlan.zip", mime="application/zip")

        # --- VIEW ALL ---
        with st.expander("🔍 View All 8 Algorithm Comparisons"):
            for algo in others:
                algo_eb = ", ".join([f"{lot}: {eb}m" for lot, eb in algo['end_bits'].items()])
                st.markdown(f"**{algo['name']}** | Pcs: {algo['total_pcs']} | Cons: {algo['total_cons']:.2f}m | Cons/Garment: {algo['overall_cons']:.3f}m | End-bits: {algo_eb}")
                st.dataframe(algo['df'], width="stretch", hide_index=True)
                st.markdown("---")