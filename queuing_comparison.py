import math, random, itertools
import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# Base Simulation Parameters (Defaults)
# -------------------------------
N = 10                    # Number of links/users
area_size = 100.0         # Square area side length in meters
path_loss_exp = 3.0       # Path loss exponent
P = 100.0                 # Transmit power in Watts (will be varied)
noise_power = 1e-9        # Noise power in Watts
SINR_threshold = 5        # SINR threshold (will be varied)
T = 100                  # Number of time slots in the simulation
ARRIVAL_RATE = 10         # Mean arrival per time slot for each link (will be varied)

# -------------------------------
# Helper Functions
# -------------------------------
def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def generate_channel_matrix():
    """
    Generate a random channel gain matrix for one time slot:
      - Transmitters are randomly placed in the area.
      - Each receiver is placed 5-15 meters away from its transmitter.
      - Channel gains follow a path loss model with Rayleigh fading.
    Returns:
      H: A numpy array (N x N) of channel gains.
    """
    tx_positions = [(random.uniform(0, area_size), random.uniform(0, area_size))
                    for _ in range(N)]
    rx_positions = []
    for tx in tx_positions:
        r_dist = random.uniform(5, 15)
        r_angle = random.uniform(0, 2 * math.pi)
        rx_positions.append((tx[0] + r_dist * math.cos(r_angle),
                             tx[1] + r_dist * math.sin(r_angle)))
    H = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            d = distance(rx_positions[i], tx_positions[j])
            if d < 1:  # Avoid extremely small distances
                d = 1
            path_loss = d ** (-path_loss_exp)
            # Rayleigh fading modeled as an exponential random variable
            rayleigh_factor = -math.log(random.random() + 1e-9)
            H[i][j] = path_loss * rayleigh_factor
    return np.array(H)

def compute_SINRs(active_set, H):
    """
    Compute the SINR for each link in the active set.
    For link i in active_set:
       SINR_i = (P * H[i,i]) / (noise_power + sum_{j in active_set, j != i} P * H[i,j])
    Returns:
      A dictionary mapping link index to its SINR.
    """
    sinr = {}
    for i in active_set:
        signal = P * H[i][i]
        interference = sum(P * H[i][j] for j in active_set if j != i)
        sinr[i] = signal / (noise_power + interference)
    return sinr

def build_conflict_edges(H):
    """
    Build pairwise conflict edges (conflict graph) based on SINR.
    Two links i and j conflict if, when transmitting together (active set {i,j}),
    at least one of them does not meet the SINR_threshold.
    Returns:
      List of tuples (i, j) with i < j.
    """
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            sinr = compute_SINRs({i, j}, H)
            if sinr[i] < SINR_threshold or sinr[j] < SINR_threshold:
                edges.append((i, j))
    return edges

def build_hyperedges(H, conflict_edges):
    """
    Build minimal hyperedge conflict sets.
    For every combination (of size 2 to N), if the set’s combined transmission causes
    at least one link's SINR to fall below threshold then it is conflicting.
    Only minimal (w.r.t. set inclusion) conflicting sets are returned.
    Returns:
      A list of sets, each representing a minimal hyperedge.
    """
    minimal_hyperedges = []
    for size in range(2, N+1):
        for combo in itertools.combinations(range(N), size):
            combo_set = set(combo)
            sinr = compute_SINRs(combo_set, H)
            if not any(sinr[u] < SINR_threshold for u in combo_set):
                continue
            if any(existing.issubset(combo_set) for existing in minimal_hyperedges):
                continue
            minimal_hyperedges = [e for e in minimal_hyperedges if not combo_set.issubset(e)]
            minimal_hyperedges.append(combo_set)
    return minimal_hyperedges

# -------------------------------
# Queue-aware Greedy Solvers
# -------------------------------

def queue_greedy_solver_conflict(H, conflict_edges, Q):
    """
    Greedy scheduler using conflict graph constraints.
    Links are ranked by a metric based solely on local (direct) channel information:
         metric_i = Q[i] * log2(1 + P*H[i,i]/noise_power)
    The schedule is built greedily (in sorted order) so that no pair in conflict_edges is scheduled together.
    After choosing S, the actual SINR (and thus rate) for each scheduled link is computed from the full H.
    Returns:
      schedule: numpy binary array (length N) with 1 if scheduled.
      rates: list of computed rates (0 for unscheduled links).
    """
    num_links = H.shape[0]
    # Compute local "direct" rates.
    direct_rates = [math.log2(1 + P * H[i,i] / noise_power) for i in range(num_links)]
    metrics = [(i, Q[i] * direct_rates[i]) for i in range(num_links)]
    sorted_links = sorted(metrics, key=lambda x: x[1], reverse=True)
    
    scheduled_set = set()
    schedule = [0] * num_links
    # Precompute conflict edges as a set for fast checking.
    conflict_edges_set = {(min(u, v), max(u, v)) for (u, v) in conflict_edges}
    
    for (i, m) in sorted_links:
        can_schedule = True
        for j in scheduled_set:
            if (min(i, j), max(i, j)) in conflict_edges_set:
                can_schedule = False
                break
        if can_schedule:
            scheduled_set.add(i)
            schedule[i] = 1
            
    # Compute actual SINRs (and hence rates) using the full interference information.
    if scheduled_set:
        sinr_final = compute_SINRs(scheduled_set, H)
    else:
        sinr_final = {}
    #rates = [math.log2(1 + sinr_final[i]) if i in scheduled_set else 0 for i in range(num_links)]
    rates = []
    for i in range(num_links):
        if i in scheduled_set:
            if sinr_final[i] > SINR_threshold:
                rates.append(math.log2(1 + sinr_final[i]))
            else:
                rates.append(0)
        else:
            rates.append(0)
    return np.array(schedule), rates

def queue_greedy_solver_hyper(H, hyperedges_list, Q):
    """
    Greedy scheduler using hyperedge constraints.
    Links are ranked by the local metric:
         metric_i = Q[i] * log2(1 + P*H[i,i]/noise_power)
    A link is added if its addition does not cause the candidate set to fully contain any hyperedge.
    Then, the actual SINRs (and thus rates) are computed for the scheduled set using the full H.
    Returns:
      schedule: numpy binary array (length N).
      rates: list of computed rates (0 for unscheduled links).
    """
    num_links = H.shape[0]
    direct_rates = [math.log2(1 + P * H[i,i] / noise_power) for i in range(num_links)]
    metrics = [(i, Q[i] * direct_rates[i]) for i in range(num_links)]
    sorted_links = sorted(metrics, key=lambda x: x[1], reverse=True)
    
    scheduled_set = set()
    schedule = [0] * num_links
    for (i, m) in sorted_links:
        candidate = scheduled_set.union({i})
        feasible = True
        for E in hyperedges_list:
            if set(E).issubset(candidate):
                feasible = False
                break
        if feasible:
            scheduled_set.add(i)
            schedule[i] = 1
            
    if scheduled_set:
        sinr_final = compute_SINRs(scheduled_set, H)
    else:
        sinr_final = {}
    #rates = [math.log2(1 + sinr_final[i]) if i in scheduled_set else 0 for i in range(num_links)]
    rates = []
    for i in range(num_links):
        if i in scheduled_set:
            if sinr_final[i] > SINR_threshold:
                rates.append(math.log2(1 + sinr_final[i]))
            else:
                rates.append(0)
        else:
            rates.append(0)
    return np.array(schedule), rates

# -------------------------------
# Simulation Function
# -------------------------------
def simulate_comparison():
    """
    Simulate T time slots with independently changing channel conditions.
    At every slot, the same arrivals (from a Poisson process) are added to two parallel systems:
      1. Conflict Graph Scheduler.
      2. Hyperedge Scheduler.
    Each scheduler makes its decision using only local (direct) channel information.
    After the schedule S is chosen, the actual served rates are computed from full H.
    The queues are updated accordingly.
    
    Returns:
      (Q_conflict_record, Q_hyper_record): Lists (time series) of queue vectors.
    """
    Q_conflict = np.zeros(N)  # Queue vector for conflict graph scheduler
    Q_hyper = np.zeros(N)     # Queue vector for hyperedge scheduler
    Q_conflict_record = []
    Q_hyper_record = []
    
    for t in range(T):
        # Identical arrivals for both systems.
        arrivals = np.random.poisson(lam=ARRIVAL_RATE, size=N)
        Q_conflict += arrivals
        Q_hyper += arrivals
        
        # Generate a new channel realization.
        H = generate_channel_matrix()
        
        # Build conflict and hyperedge constraints.
        conflict_edges = build_conflict_edges(H)
        hyperedges = build_hyperedges(H, conflict_edges)
        hyperedges_list = [list(e) for e in hyperedges]
        
        # Scheduling decisions.
        sched_conflict, rates_conflict = queue_greedy_solver_conflict(H, conflict_edges, Q_conflict)
        sched_hyper, rates_hyper = queue_greedy_solver_hyper(H, hyperedges_list, Q_hyper)
        
        # Update queues (each scheduled link transmits bits equal to its computed rate).
        for i in range(N):
            if sched_conflict[i] == 1:
                Q_conflict[i] = max(Q_conflict[i] - rates_conflict[i], 0)
            if sched_hyper[i] == 1:
                Q_hyper[i] = max(Q_hyper[i] - rates_hyper[i], 0)
        
        Q_conflict_record.append(Q_conflict.copy())
        Q_hyper_record.append(Q_hyper.copy())
        
    return Q_conflict_record, Q_hyper_record

# -------------------------------
# Experiment Wrapper Function
# -------------------------------
def run_simulation(P_val, arrival_rate, sinr_thresh, T_sim=T):
    """
    Run one simulation instance for T_sim slots using the given parameters.
    The global parameters (P, ARRIVAL_RATE, SINR_threshold) are updated accordingly.
    
    Returns:
       avg_conf: overall average queue size (across links and time) for the conflict graph scheduler.
       avg_hyper: overall average queue size for the hyperedge scheduler.
    """
    global P, ARRIVAL_RATE, SINR_threshold, T
    P = P_val
    ARRIVAL_RATE = arrival_rate
    SINR_threshold = sinr_thresh
    T = T_sim
    Q_conf_rec, Q_hyper_rec = simulate_comparison()
    avg_conf = np.mean(Q_conf_rec)
    avg_hyper = np.mean(Q_hyper_rec)
    return avg_conf, avg_hyper

# -------------------------------
# Experimental Studies & Plotting: Gain (Conflict Graph vs. Hyperedges)
# -------------------------------
num_runs = 5  # Number of independent simulation runs per parameter setting

# 1. Vary Transmit Power P while keeping ARRIVAL_RATE=10, SINR_threshold=1.
P_vals = [20, 34, 50, 70,100]
gain_P = []
gain_P_std = []
for p in P_vals:
    diff_runs = []
    for _ in range(num_runs):
        avg_conf, avg_hyper = run_simulation(p, arrival_rate=10, sinr_thresh=1, T_sim=T)
        diff_runs.append((avg_conf - avg_hyper)/avg_conf)
    gain_P.append(np.mean(diff_runs))
    gain_P_std.append(np.std(diff_runs))

plt.figure()
plt.errorbar(P_vals, gain_P, yerr=gain_P_std, marker='o', capsize=5)
plt.xlabel('Transmit Power P')
plt.ylabel('Queue Length Gain (Conflict - Hyper)')
plt.title('Performance Gain vs Transmit Power')
plt.grid(True)
plt.tight_layout()

plt.show()


# 3. Vary SINR Threshold while keeping P=100, ARRIVAL_RATE=10.
sinr_vals = [1, 2, 3, 4, 5]
gain_sinr = []
gain_sinr_std = []
for s in sinr_vals:
    diff_runs = []
    for _ in range(num_runs):
        avg_conf, avg_hyper = run_simulation(100, arrival_rate=10, sinr_thresh=s, T_sim=T)
        diff_runs.append((avg_conf - avg_hyper)/avg_conf)
    gain_sinr.append(np.mean(diff_runs))
    gain_sinr_std.append(np.std(diff_runs))

plt.figure()
plt.errorbar(sinr_vals, gain_sinr, yerr=gain_sinr_std, marker='o', capsize=5)
plt.xlabel('SINR Threshold')
plt.ylabel('Queue Length Gain (Conflict - Hyper)')
plt.title('Performance Gain vs SINR Threshold')
plt.grid(True)
plt.tight_layout()
plt.show()
