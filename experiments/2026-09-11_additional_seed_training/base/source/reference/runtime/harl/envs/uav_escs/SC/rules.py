"""Rules use actor-observable information and fixed environment constants only."""
import numpy as np


def mode_choice(env, observation, kind):
    k, m = env.ds_per_uav, env.n_semantic_modes
    cache = observation[1:1+k] > 0
    cached_age = observation[1+k:1+2*k] * env.A_max
    aoi = observation[1+2*k:1+3*k] * env.A_max
    loads = observation[1+3*k:1+3*k+m] * env.Lambda_ref
    quality = observation[1+3*k+m:1+3*k+m+k*m].reshape(k, m) * env.Q_max
    # Last two real UAV observation entries are Q_req/Q_max and A_limit/ref.
    q_req = observation[env.uav_obs_dim-2] * env.Q_max
    a_limit = observation[env.uav_obs_dim-1] * env.A_limit_context_ref
    # float32 observations cannot reconstruct the float64 threshold exactly.
    feasible = cache[:, None] & (quality >= q_req - 1e-5)
    valid = np.flatnonzero(feasible.any(axis=0))
    if not valid.size:
        return 0  # Environment executes no transmission if all modes fail.
    if kind == "fixed":
        # Cheapest quality-feasible mode; ties prefer higher Q then row index.
        return min(valid, key=lambda j: (float(loads[j]), -float(quality[feasible[:, j], j].mean()), int(j)))
    if kind != "greedy":
        raise ValueError(kind)
    budget = float(observation[0] * env.Lambda_ref)  # previous allocation, observed
    scores = {}
    # Rules receive the explicit task one-hot through the same observation.
    start = env.uav_obs_dim - 2 - 4 - env.num_instructions
    gid = int(np.argmax(observation[start:start+env.num_instructions]))
    wq, wa, wl = env.reward_weights_by_instruction[gid]
    for j in valid:
        candidates = np.flatnonzero(feasible[:, j])
        ordered = candidates[np.lexsort((candidates, -aoi[candidates]))]
        count = min(len(ordered), int(max(0, budget) / max(loads[j], 1e-9)))
        chosen = ordered[:count]
        q_gain = np.maximum((quality[chosen, j] - env.Q_min_eval) / (env.Q_max-env.Q_min_eval), 0).sum()
        fresh_gain = np.maximum(aoi[chosen] - (cached_age[chosen]+1), 0).sum()
        next_local = aoi + 1
        next_local[chosen] = cached_age[chosen] + 1
        local_aoi = (env.aoi_mean_weight*next_local.sum()/env.n_ds
            + env.aoi_tail_weight*np.maximum(next_local-env.aoi_tail_threshold,0).sum()/env.n_ds)/env.aoi_reward_ref
        scores[j] = (-wa*local_aoi + wq*q_gain/env.n_ds
                     - wl*count*loads[j]/env.Lambda_ref/env.n_uav
                     + env.eta_recv_aoi_bonus*fresh_gain/env.n_ds/max(a_limit, 1e-9))
    # Myopic local surrogate, not an oracle for the max-AoI team objective.
    return max(valid, key=lambda j: (scores[j], -float(loads[j]), -int(j)))


def one_hot(mode, size):
    action = np.zeros(size, dtype=np.float32)
    action[int(mode)] = 1.0
    return action


def fixed_mode_actions(env):
    return [one_hot(mode_choice(env, env._build_uav_obs(n), "fixed"), env.n_mu_modes)
            for n in range(env.n_uav)]


def rule_actions(env, obs, available, kind, rng):
    if env.active_agent_ids != list(range(1+env.n_uav)):
        raise ValueError("Rule evaluation requires the full physical-agent interface")
    if kind not in ("R_fixed", "G_local_greedy", "Random", "RoundRobin"):
        raise ValueError(kind)
    sut = np.zeros(env.n_uav, dtype=np.float32)
    if kind == "Random":
        sut = (2*rng.dirichlet(np.ones(env.n_uav))-1).astype(np.float32)
    elif kind == "G_local_greedy":
        pending = obs[0, env.n_uav+1:2*env.n_uav+1]
        aoi = obs[0, 2*env.n_uav+1:3*env.n_uav+1]
        pressure = pending * (1 + aoi)
        if np.ptp(pressure) > 1e-12:
            sut = (2*pressure/pressure.sum()-1).astype(np.float32)
    actions = [sut]
    for n in range(env.n_uav):
        if kind == "Random":
            selected = rng.choice(np.flatnonzero(available[n+1, :env.n_mu_modes] > 0))
        else:
            selected = mode_choice(env, obs[n+1], "greedy" if kind == "G_local_greedy" else "fixed")
        actions.append(one_hot(selected, env.n_mu_modes))
    return actions
