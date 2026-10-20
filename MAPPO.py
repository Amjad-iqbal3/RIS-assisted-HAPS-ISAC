import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR


# ==================================================================
#  Shared building blocks
# ==================================================================

def _mlp(in_dim, out_dim, hidden, n_hidden=2, activation=nn.ReLU):
    layers = [nn.Linear(in_dim, hidden), activation(), nn.LayerNorm(hidden)]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), activation(), nn.LayerNorm(hidden)]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module, gain=1.0):
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0)


# ==================================================================
#  Actor  (one per agent)
# ==================================================================

class ActorNetwork(nn.Module):
    """
    Gaussian policy:  Mean produced by a tanh-activated MLP so outputs
    are bounded to (-1, 1) before the environment rescales them.
    log_std is a learnable parameter vector.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.net = _mlp(obs_dim, action_dim, hidden_dim, n_hidden=2)
        
        for layer in reversed(list(self.net)):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)
                break

        
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    # ================================================================== 
    #                   forward
    # ==================================================================
    def forward(self, obs):
        mean = self.net(obs)              
        std  = torch.clamp(self.log_std.exp(), 0.01, 1.0)
        std  = std.expand_as(mean)
        return mean, std

    # ================================================================== 
    #                   Sample                
    # ================================================================== 
    def get_action(self, obs, training=True):
        mean, std = self.forward(obs)
        if training:
            dist     = torch.distributions.Normal(mean, std)
            raw      = dist.rsample()
            action   = torch.tanh(raw)                       
            
            log_prob = dist.log_prob(raw).sum(-1)
            log_prob = log_prob - torch.log(
                    1 - action.pow(2) + 1e-6).sum(-1)
            log_prob = torch.clamp(log_prob, -50.0, 0.0)
        else:
            action   = mean
            log_prob = None
        return action, log_prob

    # ================================================================== 
    #                    evaluate    
    # ================================================================== 
    def evaluate(self, obs, action):
        mean, std = self.forward(obs)
        dist      = torch.distributions.Normal(mean, std)
        action_clamped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        raw       = torch.atanh(action_clamped)
        log_probs = dist.log_prob(raw).sum(-1)
        log_probs = log_probs - torch.log(
                1 - action.pow(2) + 1e-6).sum(-1)
        log_probs = torch.clamp(log_probs, -50.0, 0.0)
        entropy   = dist.entropy().sum(-1).mean()
        entropy   = torch.clamp(entropy, 0.0, 20.0)
        return log_probs, entropy


     # ================================================================== 
    #   Per-agent Critic  (conditioned on global state for CTDE)        
    # ==================================================================


class AgentCritic(nn.Module):
    """
Value function V(s) for agent i uses the global state within the CTDE paradigm.
Each agent has its own separate critic to prevent gradients from interfering.
    """

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        return self.net(state).squeeze(-1)


    #  ================================================================== 
    #          Single-agent wrapper              
    # ==================================================================


class MAPPO_Agent:
    def __init__(self, agent_id, obs_dim, action_dim, state_dim,
                 device, actor_lr=3e-4, critic_lr=1e-3, hidden_dim=256):
        self.agent_id   = agent_id
        self.device     = device

        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = AgentCritic(state_dim, hidden_dim).to(device)

        self.actor_optim  = torch.optim.Adam(
            self.actor.parameters(),  lr=actor_lr,  eps=1e-5, weight_decay=1e-5)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=1e-5, weight_decay=1e-5)
        
     # ================================================================== 
    #                     Action
    # ==================================================================

    
    def get_action(self, obs, training=True):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            obs_t = torch.clamp(obs_t, -10.0, 10.0)
            action, log_prob = self.actor.get_action(obs_t, training)

            if torch.isnan(action).any() or torch.isinf(action).any():
                print(f"[WARN] NaN/Inf in {self.agent_id} action approaching to zero")
                action = torch.zeros_like(action)
            action   = torch.clamp(action, -1.0, 1.0)
            action_np   = action.cpu().numpy()[0]
            log_prob_np = (log_prob.cpu().numpy().item()
                           if log_prob is not None else None)
        return action_np, log_prob_np

     # ================================================================== 
    #                        value
    # ================================================================== 
    def get_value(self, state):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            s = torch.clamp(s, -10.0, 10.0)
            return self.critic(s).cpu().numpy().item()



    # ================================================================== 
    #                MAPPO_ISAC  —  top-level controller        
    # ==================================================================


class MAPPO_ISAC:
    def __init__(self, state_dim, action_dim, M, N, K, T, power_t, device,
                 obs_dims,
                 env=None,
                 lr_schedule_steps=500,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, clip_eps=0.2,
                 ppo_epochs=10, batch_size=128, gae_lambda=0.95,
                 entropy_coef=0.02,
                 hidden_dim=256):

        self.device       = device
        self.gamma        = gamma
        self.clip_eps     = clip_eps
        self.ppo_epochs   = ppo_epochs
        self.batch_size   = batch_size
        self.gae_lambda   = gae_lambda
        self.entropy_coef = entropy_coef
        self.update_count = 0
        self.hidden_dim   = hidden_dim
        self.env          = env

        self.M = M; self.N = N; self.K = K; self.T = T
        self.power_t = torch.as_tensor(power_t, device=device, dtype=torch.float32)

        self.comm_action_dim  = 2 * M * K
        self.power_action_dim = K
        self.ris_action_dim   = N
        total_action_dim = self.comm_action_dim + self.power_action_dim + self.ris_action_dim
        assert total_action_dim == action_dim, \
            f"Action dim mismatch: {total_action_dim} vs {action_dim}"

        self.obs_dims         = obs_dims
        self.global_state_dim = state_dim
        
        
        # ================================================================== 
    #           One agent (actor + critic) per sub-problem             
    # ==================================================================
        
        self.agents = {}
        for aid, obs_d, act_d in [
            ('comm',  obs_dims['comm'],  self.comm_action_dim),
            ('power', obs_dims['power'], self.power_action_dim),
            ('ris',   obs_dims['ris'],   self.ris_action_dim),
        ]:
            self.agents[aid] = MAPPO_Agent(
                agent_id=aid, obs_dim=obs_d, action_dim=act_d,
                state_dim=state_dim, device=device,
                actor_lr=actor_lr, critic_lr=critic_lr, hidden_dim=hidden_dim)
            
      # ================================================================== 
    #          LR schedulers              
    # ==================================================================    
            
       
        self.actor_schedulers = {
            aid: CosineAnnealingLR(ag.actor_optim,
                                   T_max=lr_schedule_steps, eta_min=1e-6)
            for aid, ag in self.agents.items()
        }
        self.critic_schedulers = {
            aid: CosineAnnealingLR(ag.critic_optim,
                                   T_max=lr_schedule_steps, eta_min=1e-6)
            for aid, ag in self.agents.items()
        }


      # ================================================================== 
    #          Running reward normalisation (EMA-based)         
    # ==================================================================
        
        self._rew_mean  = 0.0
        self._rew_var   = 1.0
        self._rew_count = 0
        self._rew_alpha = 0.01

        self.max_grad_norm = 0.5

    # ================================================================== 
    #        State decomposition                
    # ==================================================================
    

    def decompose_state(self, state):
        if self.env is not None:
            seg = self.env.decompose_state(state)
            return {
                'comm':  np.concatenate([seg['shared'], seg['ucq'],
                                         seg['rates'], seg['ris_summary']]).astype(np.float32),
                'power': np.concatenate([seg['shared'], seg['ucq'],
                                         seg['rates'], seg['bf_pwr']]).astype(np.float32),
                'ris':   np.concatenate([seg['shared'], seg['tgt_dists'],
                                         seg['udists_bs'], seg['udists_ris'],
                                         seg['tgt_rcs'], seg['elev_gs']]).astype(np.float32),
            }
        raise ValueError("env must be provided for state decomposition")

    # ==================================================================
    #  Action selection
    # ==================================================================

    def get_action(self, state, obs_dict=None, training=True):
        obs = obs_dict if obs_dict is not None else self.decompose_state(state)

        actions, log_probs = {}, {}
        for aid, agent in self.agents.items():
            a, lp = agent.get_action(obs[aid], training)
            actions[aid]   = a
            log_probs[aid] = lp

        combined = np.concatenate([actions['comm'], actions['power'], actions['ris']])
        combined_lp = (sum(lp for lp in log_probs.values() if lp is not None)
                       if training else None)
        return combined, combined_lp

    # ==================================================================
    #                        Update
    # ==================================================================
    def update(self, states, actions, rewards, dones, next_states):
        states      = torch.clamp(
            torch.as_tensor(np.array(states),      dtype=torch.float32, device=self.device),
            -10.0, 10.0)
        next_states = torch.clamp(
            torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            -10.0, 10.0)
        actions     = torch.clamp(
            torch.as_tensor(np.array(actions),     dtype=torch.float32, device=self.device),
            -1.0, 1.0)
        dones       = torch.as_tensor(np.array(dones),   dtype=torch.float32, device=self.device)

       # ==================================================================
    #              Running reward normalisation
       # ==================================================================
           
        rewards_np = np.array(rewards, dtype=np.float32)
        batch_mean = float(rewards_np.mean())
        batch_var  = float(rewards_np.var()) + 1e-8
        self._rew_mean = (1 - self._rew_alpha) * self._rew_mean + self._rew_alpha * batch_mean
        self._rew_var  = (1 - self._rew_alpha) * self._rew_var  + self._rew_alpha * batch_var
        rewards_norm = (rewards_np - self._rew_mean) / (np.sqrt(self._rew_var) + 1e-8)
        rewards_norm = np.clip(rewards_norm, -10.0, 10.0)
        rewards_t    = torch.as_tensor(rewards_norm, dtype=torch.float32, device=self.device)
        
        #==================================================================
        # Split combined actions
        # ==================================================================
        
        ce = self.comm_action_dim
        pe = ce + self.power_action_dim
        agent_actions = {
            'comm':  actions[:, :ce],
            'power': actions[:, ce:pe],
            'ris':   actions[:, pe:],
        }
        
        # ==================================================================
        #       Pre-compute agent observations from states
        # ==================================================================
        
        states_np = states.cpu().numpy()
        n         = len(states_np)
        agent_obs = {aid: [] for aid in self.agents}
        for i in range(n):
            obs = self.decompose_state(states_np[i])
            for aid in self.agents:
                agent_obs[aid].append(obs[aid])
        for aid in self.agents:
            agent_obs[aid] = torch.as_tensor(
                np.array(agent_obs[aid]), dtype=torch.float32, device=self.device)
            
        # ==================================================================
        #               Per-agent GAE + returns
        # ================================================================== 
        agent_advantages = {}
        agent_returns    = {}
        with torch.no_grad():
            for aid, agent in self.agents.items():
                values      = agent.critic(states)
                next_values = agent.critic(next_states)
                adv, ret    = self._calc_gae(rewards_t, dones, values, next_values)
                # Whiten advantages over the whole buffer
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                adv = torch.clamp(adv, -10.0, 10.0)
                agent_advantages[aid] = adv
                agent_returns[aid]    = ret
        
        # ==================================================================
        #  Old log-probs (before any parameter update) 
        # ==================================================================
        with torch.no_grad():
            agent_old_lp = {}
            for aid, agent in self.agents.items():
                old_lp, _ = agent.actor.evaluate(agent_obs[aid], agent_actions[aid])
                agent_old_lp[aid] = old_lp

        # ==================================================================
        #   PPO epochs
        # ==================================================================
        actor_losses  = {aid: [] for aid in self.agents}
        critic_losses = {aid: [] for aid in self.agents}

        for _ in range(self.ppo_epochs):
            for batch_idx in self._make_batches(n):

                for aid, agent in self.agents.items():
                    b_obs    = agent_obs[aid][batch_idx]
                    b_act    = agent_actions[aid][batch_idx]
                    b_adv    = agent_advantages[aid][batch_idx]
                    b_old_lp = agent_old_lp[aid][batch_idx]
                    b_ret    = agent_returns[aid][batch_idx]
                    
                    # ==================================================================
                    # Actor loss (clipped surrogate) 
                    # ==================================================================
                    curr_lp, entropy = agent.actor.evaluate(b_obs, b_act)
                    if torch.isnan(curr_lp).any() or torch.isinf(curr_lp).any():
                        continue

                    ratio  = torch.exp(curr_lp - b_old_lp)
                    ratio  = torch.clamp(ratio, 0.01, 20.0)
                    surr1  = ratio * b_adv
                    surr2  = torch.clamp(ratio,
                                          1.0 - self.clip_eps,
                                          1.0 + self.clip_eps) * b_adv

                    actor_loss = (-torch.min(surr1, surr2).mean()
                                  - self.entropy_coef * entropy)

                    if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                        continue

                    agent.actor_optim.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.actor.parameters(), self.max_grad_norm)
                    agent.actor_optim.step()
                    actor_losses[aid].append(actor_loss.item())
                    
                    # ==================================================================
                    # Critic loss (MSE to GAE returns)
                    # ==================================================================
                    b_states = states[batch_idx]
                    critic_val  = agent.critic(b_states)
                    critic_loss = F.mse_loss(critic_val, b_ret)

                    if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                        continue

                    agent.critic_optim.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.critic.parameters(), self.max_grad_norm)
                    agent.critic_optim.step()
                    critic_losses[aid].append(critic_loss.item())

        self.update_count += 1

        avg_actor  = {aid: (float(np.mean(v)) if v else 0.0)
                      for aid, v in actor_losses.items()}
        avg_critic = {aid: (float(np.mean(v)) if v else 0.0)
                      for aid, v in critic_losses.items()}

        mean_actor_loss  = float(np.mean(list(avg_actor.values())))
        mean_critic_loss = float(np.mean(list(avg_critic.values())))
        return mean_actor_loss, mean_critic_loss

    # ==================================================================
    #                       Helpers
    # ==================================================================

    def _calc_gae(self, rewards, dones, values, next_values):
        T   = len(rewards)
        adv = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nv    = values[t + 1] if t < T - 1 else next_values[t]
            delta = rewards[t] + self.gamma * nv * (1.0 - dones[t]) - values[t]
            delta = torch.clamp(delta, -10.0, 10.0)
            gae   = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = torch.clamp(adv + values, -200.0, 200.0)
        return adv, returns

    def _make_batches(self, n):
        idx = torch.randperm(n, device=self.device)
        for i in range(0, n, self.batch_size):
            yield idx[i: i + self.batch_size]

    # ==================================================================
    #                Persistence
    # ==================================================================

    def save(self, filename):
        d = {
            'update_count': self.update_count,
            'hidden_dim':   self.hidden_dim,
            'rew_mean':     self._rew_mean,
            'rew_var':      self._rew_var,
            'entropy_coef': self.entropy_coef,
        }
        for aid, ag in self.agents.items():
            d[f'actor_{aid}']        = ag.actor.state_dict()
            d[f'actor_optim_{aid}']  = ag.actor_optim.state_dict()
            d[f'critic_{aid}']       = ag.critic.state_dict()
            d[f'critic_optim_{aid}'] = ag.critic_optim.state_dict()
        torch.save(d, filename)

    def load(self, filename, map_location=None):
        ckpt = torch.load(filename, map_location=map_location)
        for aid, ag in self.agents.items():
            ag.actor.load_state_dict(ckpt[f'actor_{aid}'])
            ag.actor_optim.load_state_dict(ckpt[f'actor_optim_{aid}'])
            ag.critic.load_state_dict(ckpt[f'critic_{aid}'])
            ag.critic_optim.load_state_dict(ckpt[f'critic_optim_{aid}'])
        self.update_count = ckpt.get('update_count', 0)
        self._rew_mean    = ckpt.get('rew_mean', 0.0)
        self._rew_var     = ckpt.get('rew_var',  1.0)
        self.entropy_coef = ckpt.get('entropy_coef', self.entropy_coef)