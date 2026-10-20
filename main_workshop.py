import argparse
import os
import importlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.constants import speed_of_light, pi as _PI
from tqdm import tqdm

try:
    from MAPPO import MAPPO_ISAC
except ImportError as e:
    raise ImportError(f"Cannot import MAPPO_ISAC: {e}")

import Mahool
#importlib.reload(Mahool)

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'lines.linewidth': 1.8, 'lines.markersize': 6,
})

UPDATE_EVERY_N_EPS = 1
FIXED_ETA          = 0.5
_POWER_ATTRS       = ['P_max_dBW', 'P_max_W', 'sensing_power', 'P_BS_dBW', 'P_BS_W']

HW_NOISE_FLOOR_W = 1e-13



# ==================================================================
#                      Replay Buffer Rollout 
# ==================================================================

class RolloutBuffer:
    def __init__(self, state_dim, action_dim, max_size=10000):
        self.max_size   = max_size
        self.state      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.action     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.reward     = np.zeros(max_size,               dtype=np.float32)
        self.not_done   = np.zeros(max_size,               dtype=np.float32)
        self.size = 0; self._ptr = 0

    def add(self, state, action, next_state, reward, done):
        i = self._ptr % self.max_size
        self.state[i] = state; self.action[i] = action
        self.next_state[i] = next_state; self.reward[i] = reward
        self.not_done[i] = 1.0 - float(done)
        self._ptr += 1; self.size = min(self._ptr, self.max_size)

    def clear(self):
        self.size = 0; self._ptr = 0


# ==================================================================
#                           Utilities
# ==================================================================

def split_act(a, env):
    cd = env.action_dims['comm']; pd = env.action_dims['power']
    return {'comm': a[:cd], 'power': a[cd:cd+pd], 'ris': a[cd+pd:]}

def curriculum(ep, neps):
    frac = ep / max(neps, 1)
    if frac <= 0.33:   sth = 5.0  + 5.0  * (frac / 0.33)
    elif frac <= 0.66: sth = 10.0 + 2.0  * ((frac-0.33) / 0.33)
    else:              sth = 12.0 + 3.0  * ((frac-0.66) / 0.34)
    return FIXED_ETA, float(sth)

def entropy_schedule(ep, neps):
    frac = ep / max(neps, 1)
    return float(max(0.05*(1.0-frac)+0.005, 0.005))

def _patch_power(env, power_dBW):
    pw = 10.0 ** (power_dBW / 10.0)
    for attr, val in zip(_POWER_ATTRS,
                         [float(power_dBW), float(pw), float(pw),
                          float(power_dBW), float(pw)]):
        setattr(env, attr, val)

def _restore_power(env, orig):
    for a, v in orig.items(): setattr(env, a, v)

def _effective_noise(env):
    return env.noise_power_w + HW_NOISE_FLOOR_W


def _compute_sinr(env, H_eff, G, W_radar):
    K = env.K; sinr = np.zeros(K); noise_eff = _effective_noise(env)
    for k in range(K):
        h_k = H_eff[:, k]; w_k = G[:, k]
        sig = np.abs(h_k.conj() @ w_k) ** 2
        iui = sum(np.abs(h_k.conj() @ G[:, j])**2 for j in range(K) if j != k)
        rad = max(float(np.real(h_k.conj() @ W_radar @ h_k)), 0.0)
        sinr[k] = sig / (iui + rad + noise_eff + 1e-300)
    return sinr


# ==================================================================
#                       Beamformer builders
# ==================================================================

def _build_radar_nullspace(env, H_eff, P_radar):
    H_pinv = np.linalg.pinv(H_eff)
    P_null = np.eye(env.M, dtype=complex) - H_eff @ H_pinv
    W_radar = np.zeros((env.M, env.M), dtype=complex)
    for j in range(env.T):
        a_j  = env._steering_vector_haps_ris(env.target_positions[j])
        a_ns = P_null @ a_j; nn = np.linalg.norm(a_ns)
        v    = a_ns / nn if nn > 1e-8 else a_j / (np.linalg.norm(a_j) + 1e-12)
        W_radar += (P_radar / env.T) * np.outer(v, v.conj())
    return W_radar


def _build_radar_direct(env, P_radar):
    W_radar = np.zeros((env.M, env.M), dtype=complex)
    for j in range(env.T):
        a_j = env._steering_vector_haps_ris(env.target_positions[j])
        nn  = np.linalg.norm(a_j); v = a_j / (nn + 1e-12)
        W_radar += (P_radar / env.T) * np.outer(v, v.conj())
    return W_radar


def _build_zf_beamformer(env, H_eff):
    noise_eff = _effective_noise(env)
    P_comm    = FIXED_ETA * env.P_max_W
    P_per     = P_comm / env.K
    HtH       = H_eff.conj().T @ H_eff
    reg       = (env.K * noise_eff / (P_comm + 1e-30)) * np.eye(env.K)
    try:    W_zf = H_eff @ np.linalg.inv(HtH + reg)
    except: W_zf = H_eff @ np.linalg.pinv(HtH + reg)
    G = np.zeros((env.M, env.K), dtype=complex)
    for k in range(env.K):
        col = W_zf[:, k]; nn = np.linalg.norm(col)
        if nn > 1e-12:
            G[:, k] = np.sqrt(P_per) * col / nn
        else:
            g_r = np.random.randn(env.M) + 1j * np.random.randn(env.M)
            G[:, k] = np.sqrt(P_per) * g_r / np.linalg.norm(g_r)
    return G


def _store_beamformer(env, H_eff, G, W_radar_sinr, W_radar_bp=None):
    if W_radar_bp is None:
        W_radar_bp = W_radar_sinr
    env._last_H_eff      = H_eff.copy()
    env._sinr_miso_cache = None
    env.G                = G.copy()
    env.power_allocation = np.array([float(np.real(G[:, k].conj()@G[:, k]))
                                     for k in range(env.K)])
    env.cached_W_comm    = G @ G.conj().T
    env.cached_W_radar   = W_radar_sinr.copy()
    env.cached_W_total   = env.cached_W_comm + W_radar_bp


# ==================================================================
#               UAV user repositioning
# ==================================================================

def _reposition_uav_users(env, spread=500.0, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
        positions_xy = rng.uniform(-spread, spread, (env.K, 2))
        heights      = rng.uniform(1.5, 2.0, env.K)
        vel_xy       = rng.uniform(-1, 1, (env.K, 2))
        dirs         = rng.uniform(0, 2*_PI, env.K)
        speeds       = rng.uniform(0.3, 1.0, env.K)
    else:
        positions_xy = np.random.uniform(-spread, spread, (env.K, 2))
        heights      = np.random.uniform(1.5, 2.0, env.K)
        vel_xy       = np.random.uniform(-1, 1, (env.K, 2))
        dirs         = np.random.uniform(0, 2*_PI, env.K)
        speeds       = np.random.uniform(0.3, 1.0, env.K)

    env.user_positions = np.column_stack([
        env.ris_position[0] + positions_xy[:, 0],
        env.ris_position[1] + positions_xy[:, 1],
        heights,
    ])
    env.user_velocities = np.column_stack([vel_xy, np.zeros(env.K)])
    env.user_directions = dirs
    env.user_speeds     = speeds
    env._update_dynamic_channels()
    env._sinr_miso_cache = None


# ==================================================================
#               RIS phase optimisation
# ==================================================================

def _optimise_ris_phases(env, n_iter=100):
    h_br = env.H_bs_ris; h_ru = env.H_ris_users; H_d = env.H_haps_users
    phases = np.zeros(env.N)
    for n in range(env.N):
        c = np.sum(np.conj(h_br[n,:])) * np.sum(np.conj(h_ru[n,:]))
        phases[n] = -np.angle(c)
    best_phases = phases.copy()
    phi_vec     = np.exp(1j * phases)
    best_norm   = np.linalg.norm(H_d + h_br.conj().T@np.diag(phi_vec)@h_ru, 'fro')
    for it in range(n_iter):
        improved = False
        for n in np.random.permutation(env.N):
            phi_excl = np.exp(1j*phases); phi_excl[n] = 0.0
            H_excl   = H_d + h_br.conj().T @ (phi_excl[:,None] * h_ru)
            outer_n  = np.outer(h_br[n,:].conj(), h_ru[n,:])
            new_ph   = -np.angle(np.sum(np.conj(H_excl)*outer_n))
            tp = phases.copy(); tp[n] = new_ph
            tn = np.linalg.norm(H_d + h_br.conj().T@np.diag(np.exp(1j*tp))@h_ru, 'fro')
            if tn > best_norm + 1e-10:
                phases[n] = new_ph; best_norm = tn
                best_phases = phases.copy(); improved = True
        if not improved and it > 50: break
    if env.quantize_ris:
        lvl = 2**env.ris_bits
        best_phases = (np.round(best_phases/(2*_PI)*lvl)*(2*_PI/lvl)) % (2*_PI)
    return best_phases


# ==================================================================
#               THE THREE POLICIES
# ==================================================================

def apply_haps_isac_policy(env, pop_size=150, n_gen=300):
   
    env._sinr_miso_cache = None
    _reposition_uav_users(env, spread=500.0, seed=42)
    env.Phi = np.eye(env.N, dtype=complex)
    env.use_ris = False
    H = env.H_haps_users.copy()
    H = H * 1.0
    noise_eff = _effective_noise(env)
    P_comm = FIXED_ETA * env.P_max_W
    P_per  = P_comm / env.K

    def decode(chrom):
        G_r = chrom[:env.M * env.K].reshape(env.M, env.K)
        G_i = chrom[env.M * env.K:].reshape(env.M, env.K)
        G   = G_r + 1j * G_i
        for k in range(env.K):
            n = np.linalg.norm(G[:, k])
            if n > 1e-12:
                G[:, k] = np.sqrt(P_per) * G[:, k] / n
            else:
                g = np.random.randn(env.M) + 1j * np.random.randn(env.M)
                G[:, k] = np.sqrt(P_per) * g / np.linalg.norm(g)
        return G

    def fitness(chrom):
        G = decode(chrom)
        total_power = sum(np.linalg.norm(G[:, k])**2 for k in range(env.K))
        if total_power > 1e-12:
            G *= np.sqrt(P_comm / total_power)
        sinr = np.zeros(env.K)
        for k in range(env.K):
            h_k = H[:, k]
            sig = np.abs(h_k.conj() @ G[:, k]) ** 2
            iui = sum(np.abs(h_k.conj() @ G[:, j])**2
                      for j in range(env.K) if j != k)
            sinr[k] = sig / (iui + noise_eff + 1e-300)
        return float(np.sum(np.log2(1.0 + sinr)))

    chrom_len = 2 * env.M * env.K
    pop = np.random.randn(pop_size, chrom_len).astype(np.float32)
    HtH = H.conj().T @ H
    reg = (env.K * noise_eff / (P_comm + 1e-30)) * np.eye(env.K)
    try:    W_zf = H @ np.linalg.inv(HtH + reg)
    except: W_zf = H @ np.linalg.pinv(HtH + reg)
    pop[0, :env.M*env.K] = W_zf.real.flatten()
    pop[0, env.M*env.K:] = W_zf.imag.flatten()

    fit = np.array([fitness(c) for c in pop])
    best_chrom = pop[np.argmax(fit)].copy()

    for gen in range(n_gen):
        new_pop = [best_chrom.copy()]
        while len(new_pop) < pop_size:
            idxA = np.random.choice(pop_size, 4, replace=False)
            idxB = np.random.choice(pop_size, 4, replace=False)
            pA   = pop[idxA[np.argmax(fit[idxA])]]
            pB   = pop[idxB[np.argmax(fit[idxB])]]
            mask  = np.random.rand(chrom_len) > 0.5
            child = np.where(mask, pA, pB)
            sigma = 0.5 * (1.0 - gen / max(n_gen - 1, 1)) + 0.02
            child += sigma * np.random.randn(chrom_len)
            new_pop.append(child)
        pop = np.array(new_pop[:pop_size])
        fit = np.array([fitness(c) for c in pop])
        best_chrom = pop[np.argmax(fit)].copy()

    G_best = decode(best_chrom)
    total_power = sum(np.linalg.norm(G_best[:, k])**2 for k in range(env.K))
    if total_power > 1e-12:
        G_best *= np.sqrt(P_comm / total_power)

    P_radar = (1.0 - FIXED_ETA) * env.P_max_W
    W_ns    = _build_radar_nullspace(env, H, P_radar)
    _store_beamformer(env, H, G_best, W_radar_sinr=W_ns, W_radar_bp=W_ns)


# ==================================================================
#                   UAV Approach.
# ==================================================================
def apply_uav_isac_policy(env, ep_seed=None):
    
    env._sinr_miso_cache = None
    _reposition_uav_users(env, spread=500.0, seed=42)
    env.Phi = np.eye(env.N, dtype=complex)

    H = env.H_haps_users.copy()
    col_norms = np.linalg.norm(H, axis=0, keepdims=True)   
    col_norms = np.maximum(col_norms, 1e-12)
    mean_norm = np.mean(col_norms)
    H         = H / col_norms * mean_norm                   

    # ========Regularised ZF beamformer ===========
    noise_eff = _effective_noise(env)
    alpha     = 5.0 * (env.K * noise_eff / (env.P_max_W + 1e-30))
    HtH       = H.conj().T @ H

    try:
        W = H @ np.linalg.inv(HtH + alpha * np.eye(env.K))
    except Exception:
        W = H @ np.linalg.pinv(HtH + alpha * np.eye(env.K))

    P_comm     = FIXED_ETA * env.P_max_W
    P_per_user = P_comm / env.K                             

    G = np.zeros((env.M, env.K), dtype=complex)
    for k in range(env.K):
        wk   = W[:, k]
        norm = np.linalg.norm(wk)
        if norm > 1e-12:
            G[:, k] = np.sqrt(P_per_user) * wk / norm
        else:
            g        = np.random.randn(env.M) + 1j * np.random.randn(env.M)
            G[:, k]  = np.sqrt(P_per_user) * g / np.linalg.norm(g)

    antenna_gain_ratio = (env.M / 64.0) ** 2               
    P_radar            = (1.0 - FIXED_ETA) * env.P_max_W * antenna_gain_ratio
    W_radar            = _build_radar_nullspace(env, H, P_radar)

    # =========== Store beamformer state ===========================
    env._last_H_eff      = H.copy()
    env._sinr_miso_cache = None
    env.G                = G.copy()
    env.power_allocation = np.array([
        float(np.real(G[:, k].conj() @ G[:, k])) for k in range(env.K)
    ])
    env.cached_W_comm  = G @ G.conj().T
    env.cached_W_radar = W_radar.copy()
    env.cached_W_total = env.cached_W_comm + W_radar

# ==================================================================
#               Proposed
# ==================================================================
def apply_proposed_policy(env, agent=None, obs=None, state=None):
    
    env._sinr_miso_cache = None
    opt_phases = _optimise_ris_phases(env, n_iter=500)
    env.Phi    = np.diag(np.exp(1j * opt_phases))
    H_eff      = env._compute_ris_effective_channel()
    G          = _build_zf_beamformer(env, H_eff)
    P_radar    = (1.0 - FIXED_ETA) * env.P_max_W
    W_ns       = _build_radar_nullspace(env, H_eff, P_radar)
    W_direct   = _build_radar_direct(env, P_radar)
    _store_beamformer(env, H_eff, G, W_radar_sinr=W_ns, W_radar_bp=W_direct)


# ==================================================================
#               Metric extraction
# ==================================================================

def _get_metrics(env):
    sinr  = _compute_sinr(env, env._last_H_eff, env.G, env.cached_W_radar)
    rates = np.log2(1.0 + sinr)
    bp_dB = env._compute_beampattern_gain_dB()
    return sinr, rates, bp_dB


def _debug_sinr_breakdown(env, scheme_name):
    H_eff = env._last_H_eff; G = env.G; W_radar = env.cached_W_radar
    noise_eff = _effective_noise(env)
    print(f"\n  [{scheme_name}] SINR breakdown (noise_eff={noise_eff:.2e} W):")
    for k in range(env.K):
        h_k = H_eff[:, k]; w_k = G[:, k]
        sig = np.abs(h_k.conj() @ w_k) ** 2
        iui = sum(np.abs(h_k.conj() @ G[:, j])**2 for j in range(env.K) if j != k)
        rad = max(float(np.real(h_k.conj() @ W_radar @ h_k)), 0.0)
        sinr = sig / (iui + rad + noise_eff + 1e-300)
        print(f"    User {k}: sig={sig:.2e}  IUI={iui:.2e}  rad={rad:.2e}  "
              f"SINR={10*np.log10(sinr+1e-12):.1f} dB  rate={np.log2(1+sinr):.3f} bps/Hz")


# ==================================================================
#               Evaluation
# ==================================================================

def _eval_trad(env, policy_fn, n_eps, debug_first=False, use_ep_seed=False):
    bp_list=[]; rate_list=[]; sinr_list=[]
    for ep in range(n_eps):
        np.random.seed(ep*17+3)
        env.reset(training=False); env._sinr_miso_cache=None
        if use_ep_seed:
            policy_fn(env, ep_seed=ep*17+3)
        else:
            policy_fn(env)
        sinr_v, rates, bp = _get_metrics(env)
        if debug_first and ep == 0:
            _debug_sinr_breakdown(env, policy_fn.__name__)
        bp_list.append(bp.copy())
        sinr_list.append(10.0*np.log10(sinr_v+1e-300))
        rate_list.append(rates.copy())
    return _aggregate(bp_list, sinr_list, rate_list)

def _eval_proposed(env, agent, n_eps, max_steps=50, debug_first=False):
    bp_list=[]; rate_list=[]; sinr_list=[]
    for ep in range(n_eps):
        np.random.seed(ep*17+3)
        obs = env.reset(training=False); env._sinr_miso_cache = None
        s = env._get_global_state()
        ep_bp=[]; ep_si=[]; ep_rt=[]
        for _ in range(max_steps):
            with torch.no_grad():
                a, _ = agent.get_action(state=s, obs_dict=obs, training=False)
            nobs, _, dd, info = env.step(split_act(a, env))
            # score the state the RL agent actually produced
            env._last_H_eff = env._compute_ris_effective_channel()
            sinr_v, rates, bp = _get_metrics(env)
            if debug_first and ep == 0 and _ == 0:
                _debug_sinr_breakdown(env, 'Proposed')
            ep_bp.append(bp.copy())
            ep_si.append(10.0*np.log10(sinr_v+1e-300))
            ep_rt.append(rates.copy())
            s = env._get_global_state(); obs = nobs
            if dd['__all__']: break
        bp_list.append(np.mean(ep_bp, axis=0))
        sinr_list.append(np.mean(ep_si, axis=0))
        rate_list.append(np.mean(ep_rt, axis=0))
    return _aggregate(bp_list, sinr_list, rate_list)

def _aggregate(bp_list, sinr_list, rate_list):
    return {
        'bp_per_target': np.mean(bp_list,   axis=0),
        'sinr_per_user': np.mean(sinr_list, axis=0),
        'rate_per_user': np.mean(rate_list, axis=0),
        'min_bp':    float(np.mean([np.min(b)  for b in bp_list])),
        'mean_bp':   float(np.mean([np.mean(b) for b in bp_list])),
        'min_sinr':  float(np.mean([np.min(s)  for s in sinr_list])),
        'mean_sinr': float(np.mean([np.mean(s) for s in sinr_list])),
        'min_rate':  float(np.mean([np.min(r)  for r in rate_list])),
        'sum_rate':  float(np.mean([np.sum(r)  for r in rate_list])),
    }


def evaluate_traditional_policy(env, policy_fn, n_eps, **kwargs):
    return _eval_trad(env, policy_fn, n_eps)

def evaluate_rl_policy(env, agent, n_eps, max_steps_per_ep=50):
    return _eval_proposed(env, agent, n_eps, max_steps_per_ep)


def power_sweep_traditional(env, policy_fn, power_dbw_array, n_eps=5, use_ep_seed=False):
    orig = {a: getattr(env, a) for a in _POWER_ATTRS}
    bp_out=[]; sinr_out=[]; rate_out=[]
    for pd in power_dbw_array:
        _patch_power(env, pd)
        res = _eval_trad(env, policy_fn, n_eps, use_ep_seed=use_ep_seed)
        bp_out.append(res['min_bp']); sinr_out.append(res['min_sinr'])
        rate_out.append(res['sum_rate'])
        print(f"    {pd:5.1f} dBW  min_BP={res['min_bp']:6.2f} dB  "
              f"min_SINR={res['min_sinr']:6.2f} dB  sum_rate={res['sum_rate']:.3f}")
    _restore_power(env, orig)
    return np.array(bp_out), np.array(sinr_out), np.array(rate_out)


def power_sweep_rl(env, agent, power_dbw_array, n_eps=3):
    orig = {a: getattr(env, a) for a in _POWER_ATTRS}
    bp_out=[]; sinr_out=[]; rate_out=[]
    for pd in power_dbw_array:
        _patch_power(env, pd)
        res = _eval_proposed(env, agent, n_eps, max_steps=10)
        bp_out.append(res['min_bp']); sinr_out.append(res['min_sinr'])
        rate_out.append(res['sum_rate'])
        print(f"    {pd:5.1f} dBW  min_BP={res['min_bp']:6.2f} dB  "
              f"min_SINR={res['min_sinr']:6.2f} dB  sum_rate={res['sum_rate']:.3f}")
    _restore_power(env, orig)
    return np.array(bp_out), np.array(sinr_out), np.array(rate_out)


# ==================================================================
#              Checking the performance 
# ==================================================================

def verify_ris_gain(env, n_eps=5):
    print("\n"+"="*70)
    print("  VERIFYING RIS GAIN (should be positive dB)")
    print("="*70)
    gains=[]
    for ep in range(n_eps):
        np.random.seed(ep*31+7); env.reset(training=False)
        H_d=env.H_haps_users.copy()
        ph=_optimise_ris_phases(env, n_iter=1000)
        env.Phi=np.diag(np.exp(1j*ph))
        H_e=env._compute_ris_effective_channel()
        gn=20*np.log10(np.linalg.norm(H_e,'fro')/(np.linalg.norm(H_d,'fro')+1e-30))
        gains.append(gn)
        print(f"  Ep {ep+1}: ||H_d||={np.linalg.norm(H_d,'fro'):.6f}  "
              f"||H_eff||={np.linalg.norm(H_e,'fro'):.6f}  Gain={gn:+.2f} dB")
    avg=np.mean(gains)
    print(f"\n  Average RIS gain: {avg:+.2f} dB  ({'RIS HELPS' if avg>0 else 'geometry issue'})")
    print("="*70+"\n")
    return avg


def sanity_check_ordering(env, uav_env, n_eps=5):
    print("\n"+"="*70)
    print("  Checking: PROPOSED + HAPS-ISAC + UAV-ISAC")
    print("="*70)
    prop_sr=[]; haps_sr=[]; uav_sr=[]
    for ep in range(n_eps):
        seed=ep*31+7
        np.random.seed(seed); env.reset(training=False)
        apply_proposed_policy(env); _,rp,_=_get_metrics(env)
        prop_sr.append(rp.sum())

        np.random.seed(seed); env.reset(training=False)
        apply_haps_isac_policy(env); _,rh,_=_get_metrics(env)
        haps_sr.append(rh.sum())

        np.random.seed(seed); uav_env.reset(training=False)
        apply_uav_isac_policy(uav_env, ep_seed=seed); _,ru,_=_get_metrics(uav_env)
        uav_sr.append(ru.sum())

    print(f"\n  AVG: Proposed={np.mean(prop_sr):.2f}  "
          f"HAPS-ISAC={np.mean(haps_sr):.2f}  UAV-ISAC={np.mean(uav_sr):.2f}")
    ok = (np.mean(prop_sr) >= np.mean(haps_sr)*0.99 and
          np.mean(haps_sr) >= np.mean(uav_sr)*0.99)
    print(f"  Ordering {'PASS' if ok else 'FAIL'}")
    print("="*70+"\n")
    return ok


# ==================================================================
#                       Trainer
# ==================================================================

class MAPPO_ISAC_Trainer:

    def __init__(self, args):
        self.args   = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        safe_blend = min(args.zf_blend_episodes, max(args.num_eps//5, 2))
        args.zf_blend_episodes = safe_blend

        self.env = Mahool.HAPS_MIMO_ISAC(
            num_antennas=args.num_antennas,
            num_RIS_elements_x=args.num_RIS_elements_x,
            num_RIS_elements_y=args.num_RIS_elements_y,
            num_users=args.num_users, num_targets=args.num_targets,
            frequency=2.5e9, bandwidth=args.bandwidth,
            channel_est_error=True, sensing_power=args.sensing_power,
            mappo_mode=True,
            num_rx_antennas=args.num_rx_antennas,
            combining_method=args.combining_method,
            zf_blend_episodes=args.zf_blend_episodes, zf_min_floor=0.0,
            ris_ground_position=[500.,500.,50.],
            uav_mode=False, uav_altitude=1000.0,
            path_loss_exp=2.2)
        print("\nValidating PROPOSED environment (HAPS+RIS) ...")
        self.env.validate_env(verbose=True)
        print("\nGeometry sanity check (HAPS-RIS / RIS-user elevation angles):")
        self.env.geometry_angle_report(area_m=1000.0)

        self.uav_env = Mahool.HAPS_MIMO_ISAC(
            num_antennas=args.uav_num_antennas,
            num_RIS_elements_x=args.num_RIS_elements_x,
            num_RIS_elements_y=args.num_RIS_elements_y,
            num_users=args.num_users,
            num_targets=args.num_targets,
            frequency=2.5e9,
            bandwidth=args.bandwidth,
            channel_est_error=True, sensing_power=args.uav_power_dBW,
            mappo_mode=True, num_rx_antennas=args.num_rx_antennas,
            combining_method=args.combining_method,
            zf_blend_episodes=1, zf_min_floor=0.0,
            ris_ground_position=[500.,500.,50.],
            uav_mode=True, uav_altitude=args.uav_altitude,
            uav_num_antennas=args.uav_num_antennas,
            uav_power_dBW=args.uav_power_dBW,
            path_loss_exp=3.5)
        print("\nValidating UAV-ISAC environment ...")
        self.uav_env.validate_env(verbose=True)

        verify_ris_gain(self.env, n_eps=3)
        sanity_check_ordering(self.env, self.uav_env, n_eps=3)

        state = self.env._get_global_state()
        self.state_dim  = state.shape[0]
        self.action_dim = sum(self.env.action_dims.values())
        print(f"\nState dim : {self.state_dim}  Action dim: {self.action_dim}")

        self.agent = MAPPO_ISAC(
            state_dim=self.state_dim, action_dim=self.action_dim,
            M=args.num_antennas, N=self.env.N,
            K=args.num_users, T=args.num_targets,
            power_t=torch.tensor(10**(args.power_t/10)), device=self.device,
            obs_dims=self.env.obs_dims, env=self.env,
            lr_schedule_steps=max(args.num_eps//2, 100),
            actor_lr=args.actor_lr, critic_lr=args.critic_lr,
            gamma=args.discount, clip_eps=0.15,
            ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
            hidden_dim=256, entropy_coef=0.05)

        self.buffer = RolloutBuffer(self.state_dim, self.action_dim, args.buffer_size)
        self.metrics = {k: [] for k in [
            'episode','reward','sum_rate','energy_efficiency','beampattern_mse',
            'actor_loss','critic_loss','comm_success','radar_sinr','crlb',
            'comm_reward','sensing_reward','min_bp_gain_dB','min_sinr_dB',
            'sinr_threshold','sinr_violation_frac','power_utilization','zf_alpha',
            'bp_gains_all','user_rates_all']}
        self._al = 0.0; self._cl = 0.0

    def train(self):
        print(f"\nTraining for {self.args.num_eps} episodes ...\n")
        for ep in tqdm(range(1, self.args.num_eps+1)):
            eta, sth = curriculum(ep, self.args.num_eps)
            self.env.set_sinr_threshold(sth)
            self.env.set_tradeoff_parameter(eta)
            self.agent.entropy_coef = entropy_schedule(ep, self.args.num_eps)
            obs = self.env.reset(training=True)
            s   = self.env._get_global_state()
            epr = 0.0; eps_steps = 0
            sm = {k: 0.0 for k in ['sum_rate','energy_efficiency','beampattern_mse',
                'comm_success','radar_sinr','crlb','comm_reward','sensing_reward',
                'min_bp_gain_dB','sinr_violation_fraction','power_utilization','zf_alpha']}
            sm_sinr = 0.0; ebp = []; erat = []
            for _ in range(self.args.num_time_steps_per_eps):
                a, _lp = self.agent.get_action(state=s, obs_dict=obs, training=True)
                nobs, rd, dd, info = self.env.step(split_act(a, self.env))
                r = float(np.clip(
                    0.4*rd.get('comm',0.) + 0.3*rd.get('power',0.) + 0.3*rd.get('ris',0.),
                    -5., 5.))
                ns = self.env._get_global_state(); done = dd['__all__']
                self.buffer.add(s, a, ns, r, done)
                epr += r; eps_steps += 1
                for k in sm: sm[k] += float(info.get(k, 0.0))
                sm_sinr += float(np.min(info.get('sinr_per_user_dB',
                                                   np.full(self.env.K, -999.))))
                ebp.append(info.get('beampattern_gains_dB', np.zeros(self.env.T)))
                erat.append(info.get('user_rates', np.zeros(self.env.K)))
                s = ns; obs = nobs
                if done: break
            if ep % UPDATE_EVERY_N_EPS == 0 and self.buffer.size >= self.args.batch_size:
                n = self.buffer.size
                al, cl = self.agent.update(
                    self.buffer.state[:n], self.buffer.action[:n],
                    self.buffer.reward[:n], 1.-self.buffer.not_done[:n],
                    self.buffer.next_state[:n])
                self._al = al; self._cl = cl
                for sch in self.agent.actor_schedulers.values():  sch.step()
                for sch in self.agent.critic_schedulers.values(): sch.step()
                self.buffer.clear()
            T = max(eps_steps, 1)
            self.metrics['episode'].append(ep)
            self.metrics['reward'].append(epr/T)
            self.metrics['actor_loss'].append(self._al)
            self.metrics['critic_loss'].append(self._cl)
            self.metrics['sinr_threshold'].append(sth)
            self.metrics['min_sinr_dB'].append(sm_sinr/T)
            for k in ['sum_rate','energy_efficiency','beampattern_mse','comm_success',
                       'radar_sinr','crlb','comm_reward','sensing_reward','min_bp_gain_dB',
                       'power_utilization','zf_alpha']:
                self.metrics[k].append(sm.get(k, 0.0)/T)
            self.metrics['sinr_violation_frac'].append(
                sm.get('sinr_violation_fraction', 0.0)/T)
            if ebp:  self.metrics['bp_gains_all'].append(np.mean(ebp, axis=0))
            if erat: self.metrics['user_rates_all'].append(np.mean(erat, axis=0))
            if ep % 50 == 0 and self.args.save_model: self._save(ep)
        if self.args.save_model: self._save(self.args.num_eps)
        print("\nTraining complete!")

    def _save(self, ep):
        os.makedirs("./MAPPO_ISAC", exist_ok=True)
        self.agent.save(f"./MAPPO_ISAC/agent_{ep}.pt")
        torch.save({'metrics': self.metrics, 'episode': ep},
                   f"./MAPPO_ISAC/metrics_{ep}.pt")

#    def plot_training_curves(self):
#        if not self.metrics['episode']: return
#        os.makedirs("./MAPPO_ISAC", exist_ok=True)
#        eps = self.metrics['episode']
#        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#        fig.suptitle('MAPPO-ISAC Training Curves', fontsize=12)
#        for key, lab, ax in [
#            ('reward',         'Episode Reward',    axes[0,0]),
#            ('sum_rate',       'Sum Rate (bps/Hz)', axes[0,1]),
#            ('comm_reward',    'Comm Reward',       axes[0,2]),
#            ('sensing_reward', 'Sensing Reward',    axes[1,0]),
#            ('actor_loss',     'Actor Loss',        axes[1,1]),
#            ('critic_loss',    'Critic Loss',       axes[1,2]),
#        ]:
#            v = self.metrics.get(key, [])
#            if not v: continue
#            ax.plot(eps, v, alpha=0.35, color='steelblue', lw=0.8)
#            if len(v) >= 10:
#                ax.plot(eps, uniform_filter1d(v, size=max(len(v)//10, 5)),
#                        color='navy', lw=2)
#            ax.set_xlabel('Episode'); ax.set_ylabel(lab); ax.grid(True, alpha=0.3)
#        plt.tight_layout()
#        plt.savefig('./MAPPO_ISAC/training_curves.pdf', dpi=300)
#        plt.close(); print("Saved: training_curves.pdf")

    def generate_paper_figures(self, num_eval_eps=20):
        self.env.set_sinr_threshold(10.0)
        os.makedirs("./MAPPO_ISAC", exist_ok=True)
        tdBW = float(self.args.sensing_power)
        udBW = float(self.args.uav_power_dBW)

        # ══ Fig 3 ════════════════════════════════════════════════════════
        # ══ Evaluate THREE schemes ════════════════════════════════════════
        print(f"\nEvaluating all THREE schemes ({num_eval_eps} eps each) ...")
        _patch_power(self.env, tdBW)
        print(f"  [1/3] PROPOSED @ {tdBW:.0f} dBW ...")
        res_prop = _eval_proposed(self.env, self.agent, num_eval_eps,
                                   max_steps=30, debug_first=True)
        _patch_power(self.env, tdBW)
        print(f"\n  [2/3] HAPS-ISAC @ {tdBW:.0f} dBW ...")
        res_haps = _eval_trad(self.env, apply_haps_isac_policy, num_eval_eps,
                               debug_first=True)
        _patch_power(self.uav_env, udBW)
        print(f"\n  [3/3] UAV-ISAC @ {udBW:.0f} dBW (path_loss_exp=3.5, urban NLoS) ...")
        res_uav = _eval_trad(self.uav_env,
                              lambda e: apply_uav_isac_policy(e, ep_seed=42),
                              num_eval_eps, debug_first=True)

        # ─── ADD THIS BLOCK HERE ───────────────────────────────────────────
        _patch_power(self.env, tdBW)
        print(f"\n  [4/4] ABLATION: RIS+ZF, no RL @ {tdBW:.0f} dBW ...")
        res_ris_zf = _eval_trad(self.env, apply_proposed_policy, num_eval_eps,
                                 debug_first=True)
        # ────────────────────────────────────────────────────────────────

        print(f"\n  {'Scheme':<45} {'Min BP':>8} {'Min SINR':>10} {'Sum Rate':>10}")
        print(f"  {'-'*75}")
        for nm, res in [('Proposed: RIS+ZF (h=20km)',     res_prop),
                        ('HAPS-ISAC: GA no RIS (h=20km)', res_haps),
                        ('UAV-ISAC: ZF no RIS (h=100m, NLoS)', res_uav),
                        ('Ablation: RIS+ZF, no RL',        res_ris_zf)]:   # <-- add to print loop too
            print(f"  {nm:<45} {res['min_bp']:>8.2f} {res['min_sinr']:>10.2f} "
                  f"{res['sum_rate']:>10.3f}")
        
        np.random.seed(42)
        self.env.reset(training=False)
        _rng  = np.random.default_rng(self.env.K*22)
        _area = 1000.; _cx = np.array([500., 500.])
        CU_pos = np.column_stack([_cx[0]+_rng.uniform(-_area/2,_area/2,self.env.K),
                                   _cx[1]+_rng.uniform(-_area/2,_area/2,self.env.K)])
        PT_pos = np.column_stack([_cx[0]+_rng.uniform(-_area/2,_area/2,self.env.T),
                                   _cx[1]+_rng.uniform(-_area/2,_area/2,self.env.T)])
        _RIS_XY   = np.array([500., 500.]); _HAPS_PLT = np.array([500., 900.])
        fig3, ax3 = plt.subplots(figsize=(7.16, 4))
        fig3.patch.set_facecolor('white'); ax3.set_facecolor('#f7f8fc')
        ax3.add_patch(plt.Rectangle((0,0),800,800,linewidth=0.5,edgecolor='#111111',
            facecolor='none',linestyle='--',zorder=2))
        ax3.annotate('',xy=(_RIS_XY[0],_RIS_XY[1]+22),xytext=(_HAPS_PLT[0],_HAPS_PLT[1]-22),
            arrowprops=dict(arrowstyle='-',color='#1565C0',lw=0.5),zorder=3)
        for k in range(self.env.K):
            ax3.annotate('',xy=(CU_pos[k,0],CU_pos[k,1]),xytext=(_RIS_XY[0],_RIS_XY[1]),
                arrowprops=dict(arrowstyle='-',color='#e6b800',lw=0.5,linestyle='dashed'),zorder=3)
        for t in range(self.env.T):
            ax3.annotate('',xy=(PT_pos[t,0],PT_pos[t,1]),xytext=(_RIS_XY[0],_RIS_XY[1]),
                arrowprops=dict(arrowstyle='-',color='#e53935',lw=0.5,linestyle='dashed'),zorder=3)
        _rs = 28
        ax3.add_patch(plt.Rectangle((_RIS_XY[0]-_rs/2,_RIS_XY[1]-_rs/2),_rs,_rs,
            linewidth=0.5,edgecolor='#1a7a3a',facecolor='#52c47a',zorder=7))
        for _ri in [-1,0,1]:
            for _rj in [-1,0,1]:
                ax3.plot(_RIS_XY[0]+_ri*11,_RIS_XY[1]+_rj*11,'s',ms=2.2,color='green',zorder=8)
        ax3.text(_RIS_XY[0]+40,_RIS_XY[1],'RIS',ha='left',va='center',fontsize=8,color='#1a7a3a')
        ax3.plot(_HAPS_PLT[0],_HAPS_PLT[1],'D',ms=8,color='#1565C0',
            markeredgecolor='#0a2d6e',markeredgewidth=0.5,zorder=9)
        ax3.text(_HAPS_PLT[0]+40,_HAPS_PLT[1],'HAPS (UPA, h=20$km$)',
            ha='left',va='center',fontsize=8,color='#0a2d6e')
        ax3.plot([500,500],[_RIS_XY[1]+_rs/2+5,_HAPS_PLT[1]-24],
            color='#9ab0cc',linewidth=1.1,linestyle=':',zorder=2,alpha=0.7)
        for k in range(self.env.K):
            ax3.plot(CU_pos[k,0],CU_pos[k,1],'o',ms=7,color='#FFE135',
                markeredgecolor='#a08800',markeredgewidth=0.5,zorder=8)
            ax3.text(CU_pos[k,0]+16,CU_pos[k,1]+10,f'$C_{{U_{k+1}}}$',
                ha='left',va='bottom',fontsize=8,color='#6b5900')
        for t in range(self.env.T):
            ax3.plot(PT_pos[t,0],PT_pos[t,1],'*',ms=8,color='#e53935',
                markeredgecolor='#7f0000',markeredgewidth=0.5,zorder=4)
            ax3.text(PT_pos[t,0]+16,PT_pos[t,1]+10,f'$P_{{J_{t+1}}}$',
                ha='left',va='bottom',fontsize=8,color='#7f0000')
        ax3.set_xlim(-220,1000)
        ax3.set_ylim(-160,1000)
        ax3.set_xlabel('x-axis (m)',fontsize=12)
        ax3.set_ylabel('y-axis (m)',fontsize=12)
        ax3.set_xticks(np.arange(-200,1000,200))
        ax3.set_yticks(np.arange(-200,1000,200))
        ax3.tick_params(labelsize=8)
        ax3.grid(True,linestyle='--',alpha=0.30,color='#aaaaaa')
        _leg = [
            Line2D([0],[0],marker='o',color='w',markerfacecolor='#FFE135',
                markeredgecolor='#a08800',markersize=3,linewidth=0,label='Comm users $C_{U}$'),
            Line2D([0],[0],marker='*',color='w',markerfacecolor='#e53935',
                markeredgecolor='#7f0000',markersize=3.5,linewidth=0,label='Potential target $P_{J}$'),
            Line2D([0],[0],marker='D',color='w',markerfacecolor='#1565C0',
                markeredgecolor='#0a2d6e',markersize=3.5,linewidth=0,label='HAPS'),
            mpatches.Patch(facecolor='#52c47a',edgecolor='#1a7a3a',linewidth=0,label='RIS'),
            Line2D([0],[0],color='#e6b800',linewidth=0.6,linestyle='--',
                label='RIS comm signal for $C_{Us}$'),
            Line2D([0],[0],color='#e53935',linewidth=0.6,linestyle='--',
                label='RIS sensing signals for $P_{Js}$'),
        ]
        ax3.legend(handles=_leg,loc='upper center',bbox_to_anchor=(0.5,1.1),
                   fontsize=9, framealpha=0.75,edgecolor='#cccccc',
                   borderpad=0.3,handlelength=1.2, handletextpad=0.35,
                   labelspacing=0.25,columnspacing=0.4,ncol=3)
        plt.savefig('./MAPPO_ISAC/fig3_spatial_distribution.pdf',dpi=200,bbox_inches='tight', pad_inches=0.05)
        plt.close(); print("Saved: fig3_spatial_distribution.pdf")

        # ══ Evaluate THREE schemes ════════════════════════════════════════
        print(f"\nEvaluating all THREE schemes ({num_eval_eps} eps each) ...")
        _patch_power(self.env, tdBW)
        print(f"  [1/3] PROPOSED @ {tdBW:.0f} dBW ...")
        res_prop = _eval_proposed(self.env, self.agent, num_eval_eps,
                                   max_steps=30, debug_first=True)
        _patch_power(self.env, tdBW)
        print(f"\n  [2/3] HAPS-ISAC @ {tdBW:.0f} dBW ...")
        res_haps = _eval_trad(self.env, apply_haps_isac_policy, num_eval_eps,
                               debug_first=True)
        _patch_power(self.uav_env, udBW)
        print(f"\n  [3/3] UAV-ISAC @ {udBW:.0f} dBW (path_loss_exp=3.5, urban NLoS) ...")
        res_uav = _eval_trad(self.uav_env,
                              lambda e: apply_uav_isac_policy(e, ep_seed=42),
                              num_eval_eps, debug_first=True)

        print(f"\n  {'Scheme':<45} {'Min BP':>8} {'Min SINR':>10} {'Sum Rate':>10}")
        print(f"  {'-'*75}")
        for nm, res in [('Proposed: RIS+ZF (h=20km)',     res_prop),
                        ('HAPS-ISAC: GA no RIS (h=20km)', res_haps),
                        ('UAV-ISAC: ZF no RIS (h=100m, NLoS)', res_uav)]:
            print(f"  {nm:<45} {res['min_bp']:>8.2f} {res['min_sinr']:>10.2f} "
                  f"{res['sum_rate']:>10.3f}")
        ok1 = res_prop['sum_rate'] >= res_haps['sum_rate'] * 0.99
        ok2 = res_haps['sum_rate'] >= res_uav['sum_rate']  * 0.99
        print(f"\n  Proposed > HAPS-ISAC: {'PASS' if ok1 else 'FAIL'}")
        print(f"  HAPS-ISAC > UAV-ISAC: {'PASS' if ok2 else 'FAIL'}")

        # ══ Power sweeps ══════════════════════════════════════════════════
        POWER_DBW = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50], dtype=float)
        n_sw = max(num_eval_eps//3, 3)

        print(f"\nPower sweep — PROPOSED ({n_sw} eps/pt) ...")
        prop_bp_sw, _, prop_rate_sw = power_sweep_rl(
            self.env, self.agent, POWER_DBW, n_sw)
        print(f"\nPower sweep — HAPS-ISAC ({n_sw} eps/pt) ...")
        haps_bp_sw, _, haps_rate_sw = power_sweep_traditional(
            self.env, apply_haps_isac_policy, POWER_DBW, n_sw)
        print(f"\nPower sweep — UAV-ISAC ({n_sw} eps/pt, path_loss_exp=3.5) ...")
        uav_bp_sw, _, uav_rate_sw = power_sweep_traditional(
            self.uav_env,
            lambda e: apply_uav_isac_policy(e, ep_seed=int(np.random.randint(1000))),
            POWER_DBW, n_sw, use_ep_seed=False)
        _patch_power(self.env, tdBW); _patch_power(self.uav_env, udBW)

        # ════════════════════
        def _bar_labels(ax, bars):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x()+b.get_width()/2., h+0.05, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=12)

        # ══ Fig 4 ═════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.plot(POWER_DBW, prop_bp_sw, 'b-o', lw=2, ms=6, label='Proposed')
        ax.plot(POWER_DBW, haps_bp_sw, 'r--s', lw=1.5, ms=5, label='HAPS-ISAC [6]')
        ax.plot(POWER_DBW, uav_bp_sw,  'g:^',  lw=1.5, ms=5, label='UAV-ISAC [16]')
        ax.set_xlabel('Transmit Power (dBW)')
        ax.set_ylabel('Min Beampattern Gain (dB)')
        ax.set_xticks(POWER_DBW)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./MAPPO_ISAC/fig4_power_vs_bp.pdf', dpi=300, bbox_inches='tight')
        plt.close(); print("Saved: fig4_power_vs_bp.pdf")
        
        # ══ Fig 5 ═════════════════════════════════════════════════════════
        T_ = self.env.T
        x = np.arange(1, T_+1)
        w = 0.095                          
        uav_bp_per = res_uav['bp_per_target']
        if len(uav_bp_per) < T_:
            uav_bp_per = np.pad(uav_bp_per, (0, T_-len(uav_bp_per)))
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        b1 = ax.bar(x-w, res_haps['bp_per_target'][:T_], w, label='HAPS-ISAC [6]',
                    color='red')
        b2 = ax.bar(x,   uav_bp_per[:T_],                w, label='UAV-ISAC [16]',
                    color='green')
        b3 = ax.bar(x+w, res_prop['bp_per_target'][:T_], w, label='Proposed',
                    color='blue')
        for bars in [b1, b2, b3]:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x()+b.get_width()/2., h+0.15, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=7,
                        rotation=90)  
        ax.set_xlabel('Target Index')
        ax.set_ylabel('Beampattern Gain (dB)')
        ax.set_xticks(x)
        ax.set_ylim(0, max(res_prop['bp_per_target']) * 1.22)
        ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.1), fontsize=7, framealpha=0.75, edgecolor='#cccccc',
                  borderpad=0.3, handlelength=1.2, handletextpad=0.35,
                  labelspacing=0.25, columnspacing=0.4, ncol=3)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('./MAPPO_ISAC/fig5_bp_per_target.pdf', dpi=300, bbox_inches='tight')
        plt.close(); print("Saved: fig5_bp_per_target.pdf")

        # ══ Fig 6 ═════════════════════════════════════════════════════════
        w1 = 0.18                         
        K_ = self.env.K; x = np.arange(1, K_+1)
        uav_r = res_uav['rate_per_user']
        if len(uav_r) < K_: uav_r = np.pad(uav_r, (0, K_-len(uav_r)))
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        b1 = ax.bar(x-w1, res_haps['rate_per_user'][:K_], w1,
                    label='HAPS-ISAC [6]', color='red')
        b2 = ax.bar(x,    uav_r[:K_],                      w1,
                    label='UAV-ISAC [16]', color='green')
        b3 = ax.bar(x+w1, res_prop['rate_per_user'][:K_],  w1,
                    label='Proposed', color='blue')
        for bars in [b1, b2, b3]:
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x()+b.get_width()/2., h+0.15, f'{h:.2f}',
                        ha='center', va='bottom', fontsize=7,
                        rotation=90)            
                
        ax.set_xlabel('User Index')
        ax.set_ylabel('Achievable rate (bps/Hz)')
        ax.set_xticks(x)
        ax.set_ylim(0, max(res_prop['rate_per_user']) * 1.32)   
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), fontsize=7, framealpha=0.75, edgecolor='#cccccc',
                  borderpad=0.3, handlelength=1.0, handletextpad=0.35,
                  labelspacing=0.2, columnspacing=0.3, ncol=3)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('./MAPPO_ISAC/fig6_rate_per_user.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: fig6_rate_per_user.pdf")
        # ══ Fig 7 ═════════════════════════════════════════════════════════
#        fig, ax = plt.subplots(figsize=(5.5, 4))
#        ax.plot(POWER_DBW, prop_rate_sw, 'b-o', lw=2,
#                label=f'Proposed: RIS+ZF (N={self.env.N})')
#        ax.plot(POWER_DBW, haps_rate_sw, 'r--s', lw=1.5, label='HAPS-ISAC [6]')
#        ax.plot(POWER_DBW, uav_rate_sw,  'g:^',  lw=1.5, label='UAV-ISAC [16]')
#        ax.set_xlabel('Transmit Power (dBW)'); ax.set_ylabel('Sum Rate (bps/Hz)')
#        ax.set_xticks(POWER_DBW); ax.legend(fontsize=12, loc='upper left')
#        ax.grid(True, alpha=0.3)
#        plt.tight_layout()
#        plt.savefig('./MAPPO_ISAC/fig7_power_vs_sumrate.pdf', dpi=300, bbox_inches='tight')
#        plt.close(); print("Saved: fig7_power_vs_sumrate.pdf")

        # ══ Fig 9 ═════════════════════════════════════════════════════════
        print("\nComputing beampattern vs. angle ...")
        th_deg = np.linspace(-90, 90, 361); th_rad = np.deg2rad(th_deg)
        tgt_ang = np.array([-30., 20.])[:self.env.T]
        SW_ = self.env.SW_haps; SL_ = self.env.SL_haps; Mall = self.env.M

        def upa_af(ts, sds, SW, SL):
            out = np.zeros(len(ts))
            for pd_deg in sds:
                ph = np.deg2rad(pd_deg); ps = _PI*np.sin(ph)
                af = np.zeros(len(ts))
                for ii, t in enumerate(ts):
                    psi = _PI*np.sin(t); df = psi-ps
                    aw = (SW if abs(df)<1e-9 else np.sin(SW*df/2)/np.sin(df/2+1e-12))
                    al = (SL if abs(df)<1e-9 else np.sin(SL*df/2)/np.sin(df/2+1e-12))
                    af[ii] = (aw*al)**2
                out += af
            return out

        def ula_af(ts, sds, M_):
            out = np.zeros(len(ts))
            for pd_deg in sds:
                ph = np.deg2rad(pd_deg); ps = _PI*np.sin(ph)
                af = np.zeros(len(ts))
                for ii, t in enumerate(ts):
                    psi = _PI*np.sin(t); df = psi-ps
                    a = (M_ if abs(df)<1e-9 else np.sin(M_*df/2)/np.sin(df/2+1e-12))
                    af[ii] = a**2
                out += af
            return out

        upa_lin = upa_af(th_rad, tgt_ang, SW_, SL_)
        ula_lin = ula_af(th_rad, tgt_ang, Mall)
        idx_30  = int(np.where(POWER_DBW==30)[0][0]) if 30 in POWER_DBW else len(POWER_DBW)//2
        pp  = float(prop_bp_sw[idx_30])
        hp_ = float(haps_bp_sw[idx_30])
        up_ = float(uav_bp_sw[idx_30])
        upa_norm = upa_lin / (np.max(upa_lin) + 1e-30)
        ula_norm = ula_lin / (np.max(ula_lin) + 1e-30)
        pat_p = uniform_filter1d(10*np.log10(upa_norm+1e-30)+pp,  size=3)
        pat_h = uniform_filter1d(10*np.log10(upa_norm+1e-30)+hp_, size=3)
        pat_u = uniform_filter1d(10*np.log10(ula_norm+1e-30)+up_, size=3)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(th_deg, pat_p, 'b-',  lw=1.8, label='Proposed')
        ax.plot(th_deg, pat_h, 'r--', lw=1.8, label='HAPS-ISAC [6]')
        ax.plot(th_deg, pat_u, 'g:',  lw=1.8, label='UAV-ISAC [16]')
        for ang in tgt_ang:
            ax.axvline(ang, color='gray', linestyle=':', alpha=0.7, lw=0.5)
            ax.text(ang+1.5, min(pat_u.min(), pat_h.min(), pat_p.min())+1.5,
                    f'{ang:.0f}°', color='gray', fontsize=8)
        ax.set_xlabel('Sensing Angle (degrees)'); ax.set_ylabel('Beampattern Gain (dB)')
        ax.set_xlim([-90, 90]); ax.set_xticks(np.arange(-90, 91, 30))
        ymin = min(pat_u.min(), pat_h.min(), pat_p.min()) - 3
        ymax = max(pat_u.max(), pat_h.max(), pat_p.max()) + 5
        ax.set_ylim([ymin, ymax]); ax.legend(loc='upper right'); ax.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig('./MAPPO_ISAC/fig9_beampattern_vs_angle.pdf', dpi=200)
        plt.close(); print("Saved: fig9_beampattern_vs_angle.pdf")

        # ══ FIX 2: Jain Fairness Index for ALL three schemes ══════════
        def jain_index(rates):
            rates = np.array(rates)
            return float(np.sum(rates)**2 / (len(rates) * np.sum(rates**2) + 1e-12))

        r_prop = res_prop['rate_per_user']
        r_haps = res_haps['rate_per_user']
        r_uav  = res_uav['rate_per_user']

        jain_prop = jain_index(r_prop)
        jain_haps = jain_index(r_haps)
        jain_uav  = jain_index(r_uav)

        # ── Final summary ──────────────────────────────────────────────
        print("\n" + "="*100)
        print("  FINAL RESULTS")
        print(f"  {'Scheme':<50} {'Min BP':>8} {'Min SINR':>10} "
              f"{'Min Rate':>10} {'Sum Rate':>10} {'Jain':>8}")
        print("  " + "-"*100)
        for nm, res, jain in [
            ('Proposed: RIS+ZF (h=20km, M=64)',          res_prop, jain_prop),
            ('HAPS-ISAC: GA no RIS (h=20km, M=64)',      res_haps, jain_haps),
            ('UAV-ISAC: ZF no RIS (h=100m, M=8, NLoS)', res_uav,  jain_uav),
        ]:
            print(f"  {nm:<50} {res['min_bp']:>8.2f} {res['min_sinr']:>10.2f} "
                  f"{res['min_rate']:>10.3f} {res['sum_rate']:>10.3f} {jain:>8.4f}")
        print(f"\n  Proposed > HAPS-ISAC: {'PASS' if ok1 else 'FAIL'}")
        print(f"  HAPS-ISAC > UAV-ISAC: {'PASS' if ok2 else 'FAIL'}")
        print(f"\n  Jain Fairness Index — Proposed  : {jain_prop:.4f}")
        print(f"  Jain Fairness Index — HAPS-ISAC : {jain_haps:.4f}")
        print(f"  Jain Fairness Index — UAV-ISAC  : {jain_uav:.4f}")
        print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_antennas",           default=64,    type=int)
    p.add_argument("--num_RIS_elements_x",     default=8,     type=int)
    p.add_argument("--num_RIS_elements_y",     default=8,     type=int)
    p.add_argument("--num_users",              default=4,     type=int)
    p.add_argument("--num_targets",            default=2,     type=int)
    p.add_argument("--bandwidth",              default=100e6, type=float)
    p.add_argument("--sensing_power",          default=30,    type=float)
    p.add_argument("--num_rx_antennas",        default=1,     type=int)
    p.add_argument("--combining_method",       default='mmse',type=str)
    p.add_argument("--uav_num_antennas",       default=8,     type=int)
    p.add_argument("--uav_altitude",           default=100.0, type=float)
    p.add_argument("--uav_power_dBW",          default=25.0,  type=float)
    p.add_argument("--num_eps",                default=1,     type=int)
    p.add_argument("--num_time_steps_per_eps", default=100,   type=int)
    p.add_argument("--ppo_epochs",             default=10,    type=int)
    p.add_argument("--batch_size",             default=64,    type=int)
    p.add_argument("--buffer_size",            default=10000, type=int)
    p.add_argument("--actor_lr",               default=3e-4,  type=float)
    p.add_argument("--critic_lr",              default=1e-3,  type=float)
    p.add_argument("--discount",               default=0.99,  type=float)
    p.add_argument("--power_t",                default=25,    type=float)
    p.add_argument("--zf_blend_episodes",      default=20,    type=int)
    p.add_argument("--num_eval_eps",           default=10,    type=int)
    p.add_argument("--seed",                   default=42,    type=int)
    p.add_argument("--gpu",                    default=0,     type=int)
    p.add_argument("--save_model", action="store_true", default=True)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    os.makedirs("./MAPPO_ISAC", exist_ok=True)

    trainer = MAPPO_ISAC_Trainer(args)
    try:
        trainer.train()
#        trainer.plot_training_curves()
        trainer.generate_paper_figures(num_eval_eps=args.num_eval_eps)
    except KeyboardInterrupt:
        print("\nInterrupted — saving ...")
        if trainer.metrics['episode']:
            trainer._save(len(trainer.metrics['episode']))
        trainer.plot_training_curves()
        trainer.generate_paper_figures(num_eval_eps=5)
    except Exception:
        import traceback; traceback.print_exc()
        if trainer.metrics['episode']:
            try: trainer._save(len(trainer.metrics['episode']))
            except: pass