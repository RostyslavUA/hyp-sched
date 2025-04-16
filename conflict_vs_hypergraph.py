import math, random, itertools
import numpy as np

# -------------------------------
# Simulation parameters
# -------------------------------
N = 13                   # number of users/links
area_size = 100.0        # square area side length (meters)
path_loss_exp = 3.0      # path loss exponent
P = 50.0                 # transmit power in Watts
noise_power = 1e-8       # noise power in Watts
SINR_threshold = 6       # SINR threshold (linear scale, e.g., 10 ~ 10 dB)

# -------------------------------
# Helper functions
# -------------------------------
def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def generate_channel_matrix():
    """
    Generate a random network topology:
      - Random transmitter positions over a square area.
      - Each receiver is placed near its transmitter (between 5 and 15 meters away).
      - The channel gain matrix H is computed based on path loss and an independent
        Rayleigh (exponential) fading coefficient.
    Returns:
      H: a numpy array of shape (N, N)
    """
    # Random positions for transmitters
    tx_positions = [(random.uniform(0, area_size), random.uniform(0, area_size))
                    for _ in range(N)]
    # Position each receiver near its transmitter (within 5-15 meters)
    rx_positions = []
    for tx in tx_positions:
        r_dist = random.uniform(5, 10)
        r_angle = random.uniform(0, 2 * math.pi)
        rx_positions.append((tx[0] + r_dist * math.cos(r_angle),
                             tx[1] + r_dist * math.sin(r_angle)))
    # Compute channel gain matrix H
    H = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            d = distance(rx_positions[i], tx_positions[j])
            if d < 1:
                d = 1  # avoid extremely small distances
            path_loss = d ** (-path_loss_exp)
            # Rayleigh fading modeled as exponential (using inverse CDF of uniform)
            rayleigh_factor = -math.log(random.random() + 1e-9)
            H[i][j] = path_loss * rayleigh_factor
    return np.array(H)

def compute_SINRs(active_set, H):
    """
    Compute SINRs for a given set of active links.
      active_set: a set (or list) of indices that are active.
      H: channel gain matrix (can be list of lists or numpy array)
    Returns:
      Dictionary mapping link index to its SINR value.
    """
    sinr = {}
    for i in active_set:
        # Signal power
        signal = P * H[i][i]
        # Sum interference from the other active transmitters
        interference = sum(P * H[i][j] for j in active_set if j != i)
        sinr[i] = signal / (noise_power + interference)
    return sinr

def build_conflict_edges(H):
    """
    Build conflict graph edges (pairwise conflicts) based on the SINR threshold.
    Two links (i,j) form a conflict edge if, when transmitting together, either
    link does not meet the SINR threshold.
    """
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            sinr = compute_SINRs({i, j}, H)
            if sinr[i] < SINR_threshold or sinr[j] < SINR_threshold:
                edges.append((i, j))
    return edges

def build_hyperedges(H, conflict_edges):
    """
    Build hypergraph conflict sets (minimal hyperedges) for higher-order conflicts.
    For every combination of links (size 2 up to N), if the combined transmission makes
    at least one link's SINR fall below threshold then the set is considered conflicting.
    Minimal hyperedges are stored (i.e. a hyperedge is only included if none of its subsets
    already form a conflict).
    """
    minimal_hyperedges = []
    for size in range(2, N + 1):
        for combo in itertools.combinations(range(N), size):
            combo_set = set(combo)
            sinr = compute_SINRs(combo_set, H)
            # If at least one link violates the SINR threshold, we have a conflict.
            if not any(sinr[u] < SINR_threshold for u in combo_set):
                continue
            # Skip if any already-found minimal hyperedge is a subset of this combo
            if any(existing.issubset(combo_set) for existing in minimal_hyperedges):
                continue
            # Remove any existing hyperedge that is a superset of the new minimal conflict.
            minimal_hyperedges = [e for e in minimal_hyperedges if not combo_set.issubset(e)]
            minimal_hyperedges.append(combo_set)
    return minimal_hyperedges

def compute_throughput(schedule, H, noise_vec):
    """
    Given a binary schedule vector (1 = active, 0 = inactive) compute the total
    throughput (sum rate) using Shannon capacity.
    """
    num_links = len(schedule)
    throughput = 0.0
    for i in range(num_links):
        if schedule[i]:
            interference = sum(P * H[i, j] * schedule[j] for j in range(num_links) if j != i)
            sinr = (P * H[i, i]) / (noise_vec[i] + interference)
            if sinr < SINR_threshold:
                sinr = 0
            throughput += math.log2(1 + sinr)
    return throughput

# -------------------------------
# Greedy Solver
# -------------------------------
def greedy_solver(H, conflict_list):
    """
    Greedy solver to select a subset of links that does not fully contain any conflict set.
    The links are sorted in descending order by the direct channel gain (H[i][i]),
    and then links are added one at a time if the new link does not cause any conflict.
    Arguments:
      H: numpy array representing the channel gain matrix.
      conflict_list: list of conflict sets (each conflict set can be a pair [i,j] for a
                     conflict graph or larger for hypergraph conflicts)
    Returns:
      schedule: a numpy array of shape (N,) with 1 indicating an active link and 0 otherwise.
    """
    num_links = H.shape[0]
    # Evaluate channel condition using direct link gain (you could incorporate noise if desired)
    link_metrics = [(i, H[i, i]) for i in range(num_links)]
    sorted_links = sorted(link_metrics, key=lambda x: x[1], reverse=True)
    
    schedule = [0] * num_links  # binary schedule vector
    scheduled_set = set()
    
    for (i, metric) in sorted_links:
        # Candidate schedule if link i were added
        candidate = scheduled_set.union({i})
        violates_conflict = False
        
        # Check every conflict constraint: for every conflict set in conflict_list,
        # if candidate contains every link in that set, then adding link i would violate it.
        for conf in conflict_list:
            conf_set = set(conf)
            if conf_set.issubset(candidate):
                violates_conflict = True
                break
        
        if not violates_conflict:
            scheduled_set.add(i)
            schedule[i] = 1

    return np.array(schedule)


# -------------------------------
# Run Simulation & Compare Schedulers
# -------------------------------
num_trials = 50

# Lists to store throughput results over trials for both conflict representations
greedy_rate_conf = []   # for conflict graph (pairwise constraints)
greedy_rate_hyper = []  # for hypergraph constraints

for t in range(100):
    # Generate a random channel matrix
    H = generate_channel_matrix()

    # Build conflict constraints:
    # 1. Pairwise conflict edges (conflict graph)
    conflict_edges = build_conflict_edges(H)
    conflict_edges_list = [list(edge) for edge in conflict_edges]

    # 2. Higher order hyperedges (hypergraph conflicts)
    hyperedges = build_hyperedges(H, conflict_edges)
    hyperedges_list = [list(edge) for edge in hyperedges]

    # Noise vector for all links
    noise_vec = np.full(N, noise_power)

    # Greedy scheduling using the conflict graph constraints
    sched_conf = greedy_solver(H, conflict_edges_list)
    T_conf = compute_throughput(sched_conf, H, noise_vec)

    # Greedy scheduling using the hypergraph constraints
    sched_hyper = greedy_solver(H, hyperedges_list)
    T_hyper = compute_throughput(sched_hyper, H, noise_vec)

    #if T_hyper > T_conf:
    #    print(hyperedges)
    
    #if not np.array_equal(hyperedges_list, conflict_edges_list):
    #    print("rate conflict", T_conf)
    #    print("rate hyper", T_hyper)
        

    greedy_rate_conf.append(T_conf)
    greedy_rate_hyper.append(T_hyper)

avg_rate_conf = sum(greedy_rate_conf) / num_trials
avg_rate_hyper = sum(greedy_rate_hyper) / num_trials

print("Greedy Solver Results over {} trials:".format(num_trials))
print("  Average rate (Conflict Graph): {:.2f} bit/s/Hz".format(avg_rate_conf))
print("  Average rate (Hypergraph): {:.2f} bit/s/Hz".format(avg_rate_hyper))

# Uncomment the lines below if you wish to compare with the exhaustive solver (for small N only!)
# best_T_conf, best_sched_conf = exhaustive_solver(N, np.full(N, noise_power), H, conflict_edges_list)
# best_T_hyper, best_sched_hyper = exhaustive_solver(N, np.full(N, noise_power), H, hyperedges_list)
# print("Exhaustive Solver (Conflict Graph) rate:", best_T_conf)
# print("Exhaustive Solver (Hypergraph) rate:", best_T_hyper)
