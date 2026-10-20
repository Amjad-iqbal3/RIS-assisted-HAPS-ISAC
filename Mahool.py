import numpy as np
from scipy.constants import speed_of_light, pi
from collections import deque

FIXED_ETA = 0.5


class HAPS_RIS_MIMO_ISAC:

    def __init__(self,
                 num_antennas=64,
                 num_RIS_elements_x=8,
                 num_RIS_elements_y=8,
                 num_users=4,
                 num_targets=2,
                 frequency=2.5e9,
                 bandwidth=100e6,
                 channel_est_error=True,
                 sensing_power=30,
                 max_steps=200,
                 clutter_power=1e-10,
                 channel_noise_var=1e-5,
                 quantize_ris=True,
                 visualize=False,
                 rcs_model='swerling3',
                 target_types=None,
                 multipath=True,
                 mappo_mode=True,
                 sinr_threshold_dB=10.0,
                 rician_K_factor=10.0,
                 num_rx_antennas=1,
                 combining_method='mmse',
                 zf_blend_episodes=50,
                 zf_min_floor=0.0,
                 ris_ground_position=None,
                 uav_mode=False,
                 uav_altitude=100.0,
                 uav_num_antennas=8,
                 uav_power_dBW=30.0,
                 path_loss_exp=2.2):

        self.uav_mode = uav_mode

        if uav_mode:
            num_antennas  = uav_num_antennas
            sensing_power = uav_power_dBW

        self.M  = num_antennas
        self.Nx = num_RIS_elements_x
        self.Ny = num_RIS_elements_y
        self.N  = self.Nx * self.Ny
        self.K  = num_users
        self.T  = num_targets

        # UPA factorisation
        sw = int(np.floor(np.sqrt(self.M)))
        while self.M % sw != 0 and sw > 1:
            sw -= 1
        self.SW_haps = sw
        self.SL_haps = self.M // sw
        self.SW = self.SW_haps
        self.SL = self.SL_haps

        # RIS position
        if ris_ground_position is not None:
            self._ris_ground_position = np.array(ris_ground_position, dtype=float)
        else:
            self._ris_ground_position = np.array([500.0, 500.0, 50.0])

        # RF / waveform
        self.frequency   = frequency
        self.bandwidth   = bandwidth
        self.wavelength  = speed_of_light / frequency
        self.k_wave      = 2 * pi / self.wavelength
        self.d           = self.wavelength / 2
        self.subcarriers = 16
        self.fs          = self.subcarriers * bandwidth
        self.time_step   = 1e-3
        self.max_steps   = max_steps
        self.episode_t   = 0
        self.total_episodes = 0
        self.mappo_mode  = mappo_mode

        # Path loss model
        self.path_loss_exp = path_loss_exp
        self.d0            = 1.0          # reference distance (m)

        # Altitudes
        self.haps_altitude = uav_altitude if uav_mode else 20000.0
        self.haps_drift_speed        = 1.0
        self.haps_coverage_radius    = 5000.0
        self.haps_coverage_boundary  = 2000.0
        self.haps_tx_power_dbw       = sensing_power
        self.haps_antenna_gain_dbi   = 15.0
        self.haps_noise_figure_db    = 5.0
        self.haps_solar_power_w      = 0.0
        self.haps_propulsion_power_w = 500.0
        self.atm_absorption_db_per_km = 0.022

        # Training helpers
        self._train_episode_count = 0
        self._zf_blend_episodes   = zf_blend_episodes
        self._zf_min_floor        = zf_min_floor

        # Agents
        self.agent_names = ['comm', 'power', 'ris']
        self.num_agents  = 3

        # SINR threshold
        self.sinr_threshold_dB = sinr_threshold_dB
        self.sinr_threshold    = 10 ** (sinr_threshold_dB / 10.0)

        # Noise
        self.K_rician         = rician_K_factor
        self.NR               = num_rx_antennas
        self.combining_method = combining_method.lower()
        self.clutter_power    = clutter_power
        self.noise_figure_dB  = 7
        k = 1.38e-23
        T = 290
        NF = 10 ** (self.noise_figure_dB / 10)

        self.noise_power_w = k * T * self.bandwidth * NF
        self.noise_power      = self.noise_power_w
        self.channel_noise_var = channel_noise_var
        self.sensing_noise_var = 1e-12
        self.channel_est_error = channel_est_error
        self.est_error_std     = 0.05

        # RIS quantisation
        self.quantize_ris    = quantize_ris
        self.ris_bits        = 4 if quantize_ris else 32
        self.mutual_coupling = True
        self.multipath       = multipath

        # Sensing / power
        self.rcs_model    = rcs_model
        self.target_types = (target_types or (['drone', 'car'] * self.T))[:self.T]
        self.P_max_dBW    = float(sensing_power)
        self.P_max_W      = 10 ** (self.P_max_dBW / 10.0)
        self.sensing_power  = self.P_max_W
        self.P_BS_dBW       = self.P_max_dBW
        self.P_BS_W         = self.P_max_W
        self.pulse_duration = 5e-6

        # Near-field
        aperture = max(self.Nx * self.d, self.Ny * self.d)
        self.near_field_threshold = 2 * aperture ** 2 / self.wavelength
        self.wavelength_4pi       = self.wavelength / (4 * pi)

        # QoS
        self.qos_threshold = 0.4
        self.R_min         = 2.0
        self.lambda_qos    = 0.5

        # Performance tracking
        self.recent_comm_performance    = 0.0
        self.recent_sensing_performance = 0.0
        self.performance_history = {
            'comm_reward': [], 'sense_reward': [], 'target_eta': FIXED_ETA}
        self._prev_bp_gain_min_dB = 30.0
        self._prev_sinr_min_dB    = -30.0
        self._prev_sum_rate       = 0.0
        self.reward_count         = 0
        self.positions_changed    = True
        self.visualize            = visualize

        self._initialize_geometry()
        self._initialize_channels()
        self._setup_mappo_structure()
        self.reset()

    # ==================================================================
    #  Path loss helpers  (CORRECT model — no wavelength penalty)
    # ==================================================================

    def _path_loss_amplitude(self, distance):
        distance = np.maximum(distance, self.d0)
        return (self.d0 / distance) ** (self.path_loss_exp / 2.0)

    def _rician_ss_channel(self, tx_pos, rx_pos):
        delta = rx_pos - tx_pos if tx_pos.ndim == 1 else tx_pos - rx_pos
        r     = max(np.linalg.norm(delta if tx_pos.ndim == 1 else rx_pos - tx_pos), 1.0)
        K     = self.K_rician
        mu    = np.sqrt(K / (K + 1.0))
        sig   = np.sqrt(1.0 / (2.0 * (K + 1.0)))
        h_los = mu * np.exp(-1j * self.k_wave * r)
        h_nlos = sig * (np.random.randn() + 1j * np.random.randn())
        return h_los + h_nlos

    def _build_channel_matrix(self, tx_positions, rx_positions):
        Ntx = tx_positions.shape[0]
        Nrx = rx_positions.shape[0]
        H   = np.zeros((Ntx, Nrx), dtype=complex)

        K_r   = self.K_rician
        mu    = np.sqrt(K_r / (K_r + 1.0))
        sig   = np.sqrt(1.0 / (2.0 * (K_r + 1.0)))

        diff      = tx_positions[:, None, :] - rx_positions[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        distances = np.maximum(distances, self.d0)

        amp   = self._path_loss_amplitude(distances)
        phase = np.exp(-1j * self.k_wave * distances)

        H_los  = mu * phase
        H_nlos = sig * (np.random.randn(Ntx, Nrx) +
                        1j * np.random.randn(Ntx, Nrx))
        return amp * (H_los + H_nlos)

    # =================================================================
    #  CRITICAL: Effective channel
    # =================================================================

    def _compute_ris_effective_channel(self):
         eff_Phi = self.coupling_matrix @ self.Phi if self.mutual_coupling else self.Phi
         H_reflected = self.H_bs_ris.conj().T @ eff_Phi @ self.H_ris_users
         norm_factor = np.linalg.norm(H_reflected, 'fro') + 7e-2
         H_reflected = H_reflected / norm_factor * np.sqrt(self.M)
         H_eff = self.H_haps_users + H_reflected

         return H_eff

    # =================================================================
    #  Geometry
    # =================================================================

    def _initialize_geometry(self):
        self.ris_position  = self._ris_ground_position.copy()
        self.haps_position = np.array([
            self.ris_position[0],
            self.ris_position[1],
            self.haps_altitude,
        ])
        self.bs_position = self.haps_position.copy()
        self.bs_to_ris_distance   = np.linalg.norm(self.bs_position - self.ris_position)
        self.haps_drift_direction = np.random.uniform(0, 2 * pi, 2)
        self._rebuild_ris_elements()
        self._rebuild_bs_ant_pos()
        self._initialize_users()
        self._initialize_targets()
        self._compute_coupling_matrix()
        self._initialize_clutter()

    def _rebuild_ris_elements(self):
        x = np.linspace(-(self.Nx-1)*self.d/2, (self.Nx-1)*self.d/2, self.Nx)
        y = np.linspace(-(self.Ny-1)*self.d/2, (self.Ny-1)*self.d/2, self.Ny)
        xx, yy = np.meshgrid(x, y)
        self.ris_elements = np.column_stack([
            xx.flatten() + self.ris_position[0],
            yy.flatten() + self.ris_position[1],
            np.full(self.N, self.ris_position[2]),
        ])

    def _rebuild_bs_ant_pos(self):
        xs = np.linspace(-(self.SW_haps-1)*self.d/2, (self.SW_haps-1)*self.d/2, self.SW_haps)
        ys = np.linspace(-(self.SL_haps-1)*self.d/2, (self.SL_haps-1)*self.d/2, self.SL_haps)
        xx_a, yy_a = np.meshgrid(xs, ys)
        self.bs_ant_pos = np.column_stack([
            xx_a.flatten() + self.haps_position[0],
            yy_a.flatten() + self.haps_position[1],
            np.full(self.M, self.haps_position[2]),
        ])

    def _compute_coupling_matrix(self):
        dist_matrix    = np.linalg.norm(
            self.ris_elements[:, None] - self.ris_elements[None, :], axis=2)
        coupling_coeff = 0.3 * np.exp(-dist_matrix / (2 * self.d))
        np.fill_diagonal(coupling_coeff, 1.0)
        self.coupling_matrix = coupling_coeff
    
#    def _compute_coupling_matrix(self):
#    dist_matrix    = np.linalg.norm(
#        self.ris_elements[:, None] - self.ris_elements[None, :], axis=2)
#    coupling_coeff = 0.3 * np.exp(-dist_matrix / (2 * self.d))
#    np.fill_diagonal(coupling_coeff, 1.0)
#    self.coupling_matrix = coupling_coeff

    def geometry_angle_report(self, area_m=1000.0, verbose=True):
        """
        Diagnostic: reports the actual HAPS-RIS and RIS-user elevation
        angles implied by the current geometry (self.ris_position,
                                                self.haps_position, self.bs_ant_pos, self.ris_elements).
                                                Does not alter env state.
                                                """
  
        diff  = self.bs_ant_pos[:, None, :] - self.ris_elements[None, :, :]
        horiz = np.linalg.norm(diff[..., :2], axis=-1)
        vert  = np.abs(diff[..., 2])
        elev_haps_ris = np.degrees(np.arctan2(vert, horiz))

        d3     = self.ris_position - self.user_positions
        uhoriz = np.linalg.norm(d3[:, :2], axis=1)
        uvert  = self.ris_position[2] - self.user_positions[:, 2]
        elev_ris_users = np.degrees(np.arctan2(uvert, uhoriz))

        center = self.ris_position[:2]
        edge_pts = np.array([[center[0] + area_m/2, center[1]],
                          [center[0] - area_m/2, center[1]],
                          [center[0], center[1] + area_m/2],
                          [center[0], center[1] - area_m/2]])
        corner_pts = np.array([[center[0] + area_m/2, center[1] + area_m/2],
                            [center[0] + area_m/2, center[1] - area_m/2],
                            [center[0] - area_m/2, center[1] + area_m/2],
                            [center[0] - area_m/2, center[1] - area_m/2]])
        uh = np.mean(self.user_positions[:, 2]) if len(self.user_positions) else 1.75
        for label, pts in [('edge', edge_pts), ('corner', corner_pts)]:
            p3 = np.column_stack([pts, np.full(len(pts), uh)])
            dd = self.ris_position - p3
            h  = np.linalg.norm(dd[:, :2], axis=1)
            v  = self.ris_position[2] - p3[:, 2]
            e  = np.degrees(np.arctan2(v, h))
            if verbose:
                print(f"RIS->{label} of {area_m:.0f}x{area_m:.0f} m area "
                  f"(user_h={uh:.2f} m): elevation={e[0]:.3f} deg "
                  f"(horiz={h[0]:.1f} m)")

        if verbose:
            print(f"HAPS-RIS elevation: min={elev_haps_ris.min():.5f} deg, "
              f"max={elev_haps_ris.max():.5f} deg")
            print(f"RIS-user elevation (current users): "
              f"min={elev_ris_users.min():.3f} deg, "
              f"max={elev_ris_users.max():.3f} deg, "
              f"mean={elev_ris_users.mean():.3f} deg")

        return {
                   'haps_ris_elev_deg': elev_haps_ris,
                   'ris_user_elev_deg': elev_ris_users,
                   }

    def _initialize_clutter(self):
        self.num_clutter       = 20
        self.clutter_positions = np.column_stack([
            np.random.uniform(-200, 200, self.num_clutter) + self.ris_position[0],
            np.random.uniform(-200, 200, self.num_clutter) + self.ris_position[1],
            np.random.uniform(0, 30, self.num_clutter),
        ])
        self.clutter_rcs = np.random.exponential(1, self.num_clutter)

    def _initialize_users(self):
      
        spread = 600.0   # users initialized 
        self.user_positions = np.column_stack([
            self.ris_position[0] + np.random.uniform(-spread, spread, self.K),
            self.ris_position[1] + np.random.uniform(-spread, spread, self.K),
            np.random.uniform(1.5, 2.0, self.K),
        ])
        self.user_velocities = np.column_stack([
            np.random.uniform(-1, 1, self.K),
            np.random.uniform(-1, 1, self.K),
            np.zeros(self.K),
        ])
        self.user_directions = np.random.uniform(0, 2 * pi, self.K)
        self.user_speeds     = np.random.uniform(0.3, 1.0, self.K)

    def _initialize_targets(self):
        spread = 50.0
        self.target_positions = np.column_stack([
            self.ris_position[0] + np.random.uniform(-spread, spread, self.T),
            self.ris_position[1] + np.random.uniform(-spread, spread, self.T),
            np.array([np.random.uniform(20, 50) if t == 'drone'
                      else np.random.uniform(1.5, 2.0)
                      for t in self.target_types]),
        ])
        self.target_velocities = np.zeros((self.T, 3))
        for i, t_type in enumerate(self.target_types):
            if t_type == 'drone':
                self.target_velocities[i] = np.random.uniform(-5, 5, 3)
            else:
                self.target_velocities[i, :2] = np.random.uniform(-5, 5, 2)

    # =================================================================
    #  initialisation channel  and update
    # =================================================================

    def _initialize_channels(self):
        self.H_haps_users = self._build_channel_matrix(
            self.bs_ant_pos, self.user_positions)
        self._build_mimo_channels()
        
        self.H_bs_ris = self._build_channel_matrix(
            self.bs_ant_pos, self.ris_elements).T
        self.H_bs_ris_static   = self.H_bs_ris.copy()
        self.cached_target_rcs = np.array([self._calculate_rcs(t) for t in range(self.T)])
        self._update_dynamic_channels()
        self.Phi              = np.eye(self.N, dtype=complex)
        self.G                = np.zeros((self.M, self.K), dtype=complex)
        self.R_sensing        = np.zeros((self.M, self.T), dtype=complex)
        self.power_allocation = np.ones(self.K) * self.P_max_W / self.K
        self.tradeoff_param   = FIXED_ETA
        self.multipath_components = deque(maxlen=3)
        self._update_multipath()
        self.cached_W_radar = np.zeros((self.M, self.M), dtype=complex)
        self.cached_W_comm  = np.zeros((self.M, self.M), dtype=complex)
        self.cached_W_total = np.zeros((self.M, self.M), dtype=complex)

    def _build_mimo_channels(self):
        K_f = self.K_rician
        self.H_mimo = []
        for k in range(self.K):
            r_k = max(np.linalg.norm(self.haps_position - self.user_positions[k]), 1.0)
            pl  = self._path_loss_amplitude(r_k)
            theta_k, phi_k = self._get_aod(self.haps_position, self.user_positions[k])
            sv_los = self._upa_steering_vector(theta_k, phi_k)
            H_los  = np.zeros((self.NR, self.M), dtype=complex)
            for r in range(self.NR):
                rx_phase = np.exp(-1j * 2 * pi * r * self.d *
                                  np.sin(theta_k) / self.wavelength)
                H_los[r, :] = rx_phase * sv_los.conj()
            H_nlos = (np.random.randn(self.NR, self.M) +
                      1j * np.random.randn(self.NR, self.M)) / np.sqrt(2)
            mu  = np.sqrt(K_f / (K_f + 1))
            sig = np.sqrt(1.0 / (K_f + 1))
            H_k = pl * (mu * H_los + sig * H_nlos)
            self.H_mimo.append(H_k)

    def _update_dynamic_channels(self):
        self.H_haps_users = self._build_channel_matrix(
            self.bs_ant_pos, self.user_positions)
        self._build_mimo_channels()
        
        self.H_ris_users       = self._build_channel_matrix(
            self.ris_elements, self.user_positions)
        self.H_ris_users *= np.sqrt(self.N)
        self.H_ris_users_clean = self.H_ris_users.copy()
        
        self.H_bs_ris = self._build_channel_matrix(
            self.bs_ant_pos, self.ris_elements).T
        self.H_direct       = self._build_channel_matrix(
            self.bs_ant_pos, self.user_positions)
        self.H_direct_clean = self.H_direct.copy()

        # Target channels
        self.target_distances        = np.zeros((self.N, self.T))
        self.bs_to_target_distances  = np.zeros(self.T)
        self.ris_to_target_distances = np.zeros(self.T)
        self.H_ris_target            = np.zeros((self.N, self.T), dtype=complex)
        for t in range(self.T):
            delta = self.ris_elements - self.target_positions[t]
            self.target_distances[:, t]     = np.linalg.norm(delta, axis=1)
            self.bs_to_target_distances[t]  = np.linalg.norm(
                self.bs_position - self.target_positions[t])
            self.ris_to_target_distances[t] = np.linalg.norm(
                self.ris_position - self.target_positions[t])
            rcs  = self.cached_target_rcs[t]
            d_rt = np.maximum(self.target_distances[:, t], 1.0)
            d_rb = max(self.ris_to_target_distances[t], 1.0)
           
            pl = (self._path_loss_amplitude(d_rt) *
                  self._path_loss_amplitude(d_rb) *
                  np.sqrt(self.N))
            self.H_ris_target[:, t] = np.sqrt(rcs) * pl

        self.user_haps_coverage_flag = np.ones(self.K, dtype=bool)
        self.user_nf_flag            = self.user_haps_coverage_flag
        self._update_multipath()

    def _update_multipath(self):
        if not self.multipath:
            self.multipath_components = []
            return
        self.multipath_components = [
            {'position':    np.array([np.random.uniform(-50,50),
                                       np.random.uniform(-50,50),
                                       np.random.uniform(5,30)]),
             'delay':       np.random.exponential(100e-9),
             'attenuation': np.random.uniform(0.05, 0.3),
             'phase':       np.random.uniform(0, 2*pi)}
            for _ in range(3)]

    # =================================================================
    #  Users and Target Mobility
    # =================================================================

    def _update_users(self):
        if np.random.rand() < 0.05:
            self.user_directions += np.random.uniform(-pi/4, pi/4, self.K)
        self.user_velocities = np.column_stack([
            self.user_speeds * np.cos(self.user_directions),
            self.user_speeds * np.sin(self.user_directions),
            np.zeros(self.K),
        ])
        self.user_positions += self.user_velocities * self.time_step
        max_dist = 300.0   
        for k in range(self.K):
            dist = np.linalg.norm(self.user_positions[k, :2] - self.ris_position[:2])
            if dist > max_dist:
                direction = (self.user_positions[k, :2] - self.ris_position[:2]) / dist
                self.user_positions[k, :2] = self.ris_position[:2] + direction * max_dist * 0.9
                self.user_velocities[k, :2] *= -0.5
                self.user_directions[k] += pi
        self.user_positions[:, 2] = np.clip(self.user_positions[:, 2], 1, 3)
        self.positions_changed = True

    def _update_targets(self):
        for t in range(self.T):
            self.target_positions[t] += self.target_velocities[t] * self.time_step
            dist = np.linalg.norm(self.target_positions[t, :2] - self.ris_position[:2])
            if dist > 60.0:
                self.target_positions[t, :2] = (
                    self.ris_position[:2] +
                    (self.target_positions[t, :2] - self.ris_position[:2]) * 60.0 / dist)
                self.target_velocities[t, :2] *= -1
        self.positions_changed = True

    def _update_haps(self):
        drift = self.haps_drift_speed * np.array([
            np.cos(self.haps_drift_direction[0]),
            np.sin(self.haps_drift_direction[1]),
            0.0,
        ])
        self.haps_position[:2] += drift[:2] * self.time_step
        lat_dist = np.linalg.norm(self.haps_position[:2] - self.ris_position[:2])
        if lat_dist > 100.0:
            d_ = (self.haps_position[:2] - self.ris_position[:2]) / lat_dist
            self.haps_position[:2] = self.ris_position[:2] + d_ * 100.0
        self.bs_position = self.haps_position.copy()
        self.bs_to_ris_distance = np.linalg.norm(self.bs_position - self.ris_position)
        self._rebuild_ris_elements()
        self._rebuild_bs_ant_pos()
        self.H_bs_ris_static = self._build_channel_matrix(
            self.bs_ant_pos, self.ris_elements).T
        self.positions_changed = True

    # =================================================================
    #  Basic compnents 
    # =================================================================
    def _compute_fspl(self, distance):
        return self.wavelength / (4.0 * pi * np.maximum(distance, 1.0))

    def _upa_steering_vector(self, theta, phi):
        alpha = np.exp(-1j * 2*pi * np.arange(self.SW_haps) * self.d *
                       np.sin(theta) * np.cos(phi) / self.wavelength)
        b_vec = np.exp(-1j * 2*pi * np.arange(self.SL_haps) * self.d *
                       np.sin(theta) * np.sin(phi) / self.wavelength)
        return np.kron(alpha, b_vec)

    def _get_aod(self, tx_pos, rx_pos):
        delta = rx_pos - tx_pos
        r     = np.linalg.norm(delta) + 1e-9
        elev  = np.arcsin(np.clip(delta[2] / r, -1, 1))
        theta = pi / 2.0 - elev
        phi   = np.arctan2(delta[1], delta[0])
        return theta, phi

    def _steering_vector_haps_ris(self, target_pos):
        theta, phi = self._get_aod(self.haps_position, target_pos)
        return self._upa_steering_vector(theta, phi)

    def _calculate_rcs(self, target_idx):
        t_type   = self.target_types[target_idx]
        RCS_BASE = {'car': 10, 'drone': 0.1, 'person': 1, 'default': 5}
        base_rcs = RCS_BASE.get(t_type, RCS_BASE['default'])
        if self.rcs_model == 'swerling3':
            return base_rcs * np.random.chisquare(4) / 4
        return base_rcs * (1 + 0.1 * np.random.randn())

    def _compute_null_space_projector(self, H_eff):
        H_pinv = np.linalg.pinv(H_eff)
        return np.eye(self.M, dtype=complex) - H_eff @ H_pinv

    def _compute_joint_tx_covariance(self, H_eff=None):
        if H_eff is None:
            H_eff = self._compute_ris_effective_channel()
        W_comm  = self.G @ self.G.conj().T
        P_radar = (1 - FIXED_ETA) * self.P_max_W
        W_radar = np.zeros((self.M, self.M), dtype=complex)
        P_null  = self._compute_null_space_projector(H_eff)
        for j in range(self.T):
            a_j  = self._steering_vector_haps_ris(self.target_positions[j])
            a_ns = P_null @ a_j
            nn   = np.linalg.norm(a_ns)
            v    = a_ns / nn if nn > 1e-8 else a_j / (np.linalg.norm(a_j) + 1e-12)
            W_radar += (P_radar / self.T) * np.outer(v, v.conj())
        return W_comm + W_radar, W_comm, W_radar

    # =================================================================
    #  SINR / Data rate / Beam Pattern
    # =================================================================

    def _compute_sinr_miso(self):
        if hasattr(self, '_sinr_miso_cache') and self._sinr_miso_cache is not None:
            return self._sinr_miso_cache
        W_radar = getattr(self, 'cached_W_radar',
                          np.zeros((self.M, self.M), dtype=complex))
        H_eff   = self._compute_ris_effective_channel()
        if self.G.shape != (self.M, self.K):
            self._init_beamformer()
        sinr = np.zeros(self.K)
        for k in range(self.K):
            h_k = H_eff[:, k]; wk = self.G[:, k]
            sig = np.abs(h_k.conj() @ wk) ** 2
            iui = sum(np.abs(h_k.conj() @ self.G[:, i])**2
                      for i in range(self.K) if i != k)
            si  = max(np.real(h_k.conj() @ W_radar @ h_k), 0.0)
            sinr[k] = np.maximum(sig / (iui + si + self.noise_power_w + 1e-12), 1e-10)
        self._sinr_miso_cache = sinr
        return sinr

    def _compute_sinr_mimo(self):
        return self._compute_sinr_miso()

    def _get_combining_vector(self, H_k):
        sigma2 = self.noise_power_w
        HHH    = H_k @ H_k.conj().T + sigma2 * np.eye(self.NR)
        try:    G_mmse = H_k.conj().T @ np.linalg.inv(HHH)
        except: G_mmse = H_k.conj().T @ np.linalg.pinv(HHH)
        v = H_k @ G_mmse[:, 0]
        return v / (np.linalg.norm(v) + 1e-12)

    def _compute_spectral_efficiency(self):
        sinr  = self._compute_sinr_miso()
        rates = np.log2(1.0 + sinr)
        return rates.sum(), rates

    def _compute_beampattern_gain(self):
        W_total = getattr(self, 'cached_W_total',
                          np.zeros((self.M, self.M), dtype=complex))
        bp = np.zeros(self.T)
        for j in range(self.T):
            a_j = self._steering_vector_haps_ris(self.target_positions[j]).reshape(-1, 1)
            bp[j] = max(np.real(a_j.conj().T @ W_total @ a_j)[0, 0], 1e-12)
        return bp

    def _compute_beampattern_gain_dB(self):
        return 10 * np.log10(self._compute_beampattern_gain() + 1e-12)

    def _compute_beampattern_mse(self):
        bp_dB = self._compute_beampattern_gain_dB()
        return float(np.mean((bp_dB - np.max(bp_dB)) ** 2))

    def _compute_sensing_sinr(self):
        W_comm  = self.cached_W_comm
        W_radar = self.cached_W_radar
        eff_Phi = self.coupling_matrix @ self.Phi if self.mutual_coupling else self.Phi
        sinr    = np.zeros(self.T)
        for t in range(self.T):
            h_rt      = self.H_ris_target[:, t]
            diag_h    = np.diag(h_rt.conj())
            rt_ris    = self.H_bs_ris.conj().T @ diag_h @ eff_Phi @ self.H_bs_ris
            rcs_t     = self.cached_target_rcs[t]
            d_bt      = max(np.linalg.norm(self.haps_position - self.target_positions[t]), 1.0)
            pl_bt     = self._path_loss_amplitude(d_bt)
            theta_t, phi_t = self._get_aod(self.haps_position, self.target_positions[t])
            a_t            = self._upa_steering_vector(theta_t, phi_t)
            rt_direct = np.sqrt(rcs_t) * pl_bt**2 * np.outer(a_t.conj(), a_t)
            rt        = rt_ris + rt_direct
            signal    = np.real(np.trace(rt @ W_radar @ rt.conj().T))
            interf    = np.real(np.trace(rt @ W_comm  @ rt.conj().T))
            sinr[t]   = signal / (interf + self.sensing_noise_var * self.bandwidth + 1e-12)
        return sinr

    def _compute_sensing_crlb(self):
        c          = speed_of_light
        beta_rms   = self.bandwidth / np.sqrt(12)   
        W_total    = getattr(self, 'cached_W_total',
                         np.zeros((self.M, self.M), dtype=complex))
        crlb_list  = []
        for t in range(self.T):
            d_bt   = max(self.bs_to_target_distances[t], 1.0)
            rcs_t  = self.cached_target_rcs[t]   
            a_t    = self._steering_vector_haps_ris(self.target_positions[t])
            P_sig  = float(np.real(a_t.conj() @ W_total @ a_t)) * rcs_t
            P_sig *= self._path_loss_amplitude(d_bt) ** 2
            snr_t  = P_sig / (self.sensing_noise_var * self.bandwidth + 1e-30)
            J_tau  = (8 * pi**2 * beta_rms**2 * snr_t) / (c**2 + 1e-30)
            if J_tau > 1e-30:
                crlb_range = c**2 / (4.0 * J_tau)
            else:
                crlb_range = 1e6
            crlb_list.append(float(np.clip(crlb_range, 1e-3, 1e6)))
        return float(np.mean(crlb_list))

    def _compute_energy_efficiency(self, sum_rate):
        W = getattr(self, 'cached_W_total', self._compute_joint_tx_covariance()[0])
        P_total = np.real(np.trace(W)) / 0.38 + self.M * 0.1 + 15.0 + self.N * 0.005
        return sum_rate / (P_total + 1e-9)

    def _compute_radar_sinr(self, echoes):
        return float(10 * np.log10(np.mean(self._compute_sensing_sinr()) + 1e-12))

    def _check_qos_constraints(self):
        sinr = self._compute_sinr_miso()
        _, rates = self._compute_spectral_efficiency()
        return bool(np.all(sinr >= self.sinr_threshold) and np.all(rates >= self.R_min))

    def _generate_clutter(self):
        Ts = max(int(self.pulse_duration * self.fs), 8)
        bs_c  = np.linalg.norm(self.bs_position - self.clutter_positions, axis=1)
        rc    = np.linalg.norm(self.ris_position - self.clutter_positions, axis=1)
        total = bs_c + rc
        delay = (total / speed_of_light * self.fs).astype(int)
        resp  = (np.sqrt(self.clutter_rcs) *
                 self._path_loss_amplitude(bs_c) *
                 self._path_loss_amplitude(rc))
        phases = np.random.uniform(0, 2*pi, self.num_clutter)
        phasors = resp * np.exp(1j * phases)
        echo    = np.zeros((self.M, Ts), dtype=complex)
        for d_, p in zip(delay, phasors):
            if d_ < Ts:
                echo[:, d_] += p
        noise = np.sqrt(self.clutter_power / 2) * (
            np.random.randn(self.M, Ts) + 1j * np.random.randn(self.M, Ts))
        return echo + noise

    def _receive_echoes(self, tx_signal):
        M, Ts = tx_signal.shape; Ts = max(Ts, 8)
        echoes  = np.zeros((M, Ts), dtype=complex)
        eff_Phi = self.coupling_matrix @ self.Phi if self.mutual_coupling else self.Phi
        ris_rad = eff_Phi @ (self.H_bs_ris @ tx_signal)
        for t_idx in range(self.T):
            h_rt   = self.H_ris_target[:, t_idx]
            refld  = h_rt[:, None].conj() * ris_rad
            field  = self.H_bs_ris.conj().T @ refld
            dist   = 2 * (self.bs_to_ris_distance + self.ris_to_target_distances[t_idx])
            ds     = int(np.round(dist / speed_of_light * self.fs))
            if ds >= Ts: ds = 0
            delayed = np.zeros_like(field)
            delayed[:, ds:] = field[:, :Ts - ds]
            echoes += delayed
        clutter = self._generate_clutter()
        if clutter.shape == echoes.shape:
            echoes += clutter
        echoes += np.sqrt(self.sensing_noise_var / 2) * (
            np.random.randn(*echoes.shape) + 1j * np.random.randn(*echoes.shape))
        return echoes

    # =================================================================
    #  Beamformer initialisation
    # =================================================================
    def _compute_zf_beamformer(self, H_eff=None):
        if H_eff is None:
            H_eff = self._compute_ris_effective_channel()
        HtH     = H_eff.conj().T @ H_eff
        snr_est = (FIXED_ETA * self.P_max_W) / (self.K * self.noise_power_w + 1e-30)
        reg     = (1.0 / (snr_est + 1e-9)) * np.eye(self.K)
        try:    return H_eff @ np.linalg.inv(HtH + reg)
        except: return H_eff @ np.linalg.pinv(HtH + reg)
    
    def _normalize_precoder(self, G):
        power = np.real(np.trace(G @ G.conj().T)) + 1e-12
        return G / np.sqrt(power)
    def _get_zf_alpha(self):
        blend = self._zf_blend_episodes
        count = self._train_episode_count
        if count >= 2 * blend:   return 0.0
        elif count >= blend:
            return float(0.3 * (1.0 - (count - blend) / max(blend, 1)))
        else:
            return float(1.0 - 0.7 * count / max(blend, 1))

    def _init_beamformer(self):
        H_eff  = self._compute_ris_effective_channel()
        W_zf   = self._compute_zf_beamformer(H_eff)
        p_init = np.ones(self.K) * FIXED_ETA * self.P_max_W / self.K
        self.G = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            g_k = W_zf[:, k]; nn = np.linalg.norm(g_k)
            if nn > 1e-12:
                self.G[:, k] = np.sqrt(p_init[k]) * g_k / nn
            else:
                g_r = np.random.randn(self.M) + 1j * np.random.randn(self.M)
                self.G[:, k] = np.sqrt(p_init[k]) * g_r / np.linalg.norm(g_r)

    # =================================================================
    #  Action application
    # =================================================================

    def _apply_actions_multi_agent(self, actions_dict):
        comm_action  = actions_dict['comm']
        power_action = actions_dict['power']
        ris_action   = actions_dict['ris']

        # RIS phases
        theta_scaled = 2 * pi * (ris_action + 1) / 2
        if self.quantize_ris:
            levels       = 2 ** self.ris_bits
            theta_scaled = (np.round(theta_scaled / (2*pi) * levels) *
                            (2*pi / levels)) % (2*pi)
        self.Phi = np.diag(np.exp(1j * theta_scaled))

        # Beamformer
        G_real    = comm_action[:self.M * self.K].reshape(self.M, self.K)
        G_imag    = comm_action[self.M * self.K:].reshape(self.M, self.K)
        G_learned = G_real + 1j * G_imag
        alpha_zf  = self._get_zf_alpha()
        if alpha_zf > 0:
            H_eff_curr = self._compute_ris_effective_channel()
            W_zf       = self._compute_zf_beamformer(H_eff_curr)
            W_zf = self._normalize_precoder(W_zf)
            G_learned = self._normalize_precoder(G_learned)

            G_blended = alpha_zf * W_zf + (1.0 - alpha_zf) * G_learned
        else:
            G_blended = G_learned

        # Power allocation
        p_exp = np.exp(power_action - np.max(power_action))
        p     = p_exp / (np.sum(p_exp) + 1e-12)
        p     = np.maximum(p, 0.05); p /= np.sum(p)
        P_total_comm          = FIXED_ETA * self.P_max_W
        self.power_allocation = p * P_total_comm
        self.G                = np.zeros((self.M, self.K), dtype=complex)
        for k in range(self.K):
            g_k = G_blended[:, k]; nn = np.linalg.norm(g_k)
            if nn > 1e-12:
                self.G[:, k] = np.sqrt(self.power_allocation[k]) * g_k / (nn + 1e-9)
            else:
                g_r = np.random.randn(self.M) + 1j * np.random.randn(self.M)
                self.G[:, k] = np.sqrt(self.power_allocation[k]) * g_r / np.linalg.norm(g_r)
        total_power = np.real(np.trace(self.G @ self.G.conj().T))
        if total_power > 1e-9:
            self.G = self.G * np.sqrt((FIXED_ETA * self.P_max_W) / total_power)
        H_eff_curr = self._compute_ris_effective_channel()
        P_radar    = (1 - FIXED_ETA) * self.P_max_W
        P_null     = self._compute_null_space_projector(H_eff_curr)
        self.R_sensing = np.zeros((self.M, self.T), dtype=complex)
        for j in range(self.T):
            a_j  = self._steering_vector_haps_ris(self.target_positions[j])
            a_ns = P_null @ a_j; nn_j = np.linalg.norm(a_ns)
            v    = a_ns / nn_j if nn_j > 1e-8 else a_j / (np.linalg.norm(a_j) + 1e-12)
            self.R_sensing[:, j] = np.sqrt(P_radar / self.T) * v

    # =================================================================
    #  Reward
    # =================================================================

    def _compute_rewards(self, SE, user_rates, EE, sensing_sinr_vec,
                         BP_MSE, crlb, eta=FIXED_ETA):
        bp_gains_dB = self._compute_beampattern_gain_dB()
        min_bp_dB   = float(np.min(bp_gains_dB))
        bp_score    = np.clip((min_bp_dB - 20.0) / 30.0, 0.0, 1.0)
        bp_delta    = min_bp_dB - self._prev_bp_gain_min_dB
        bp_shaping  = 0.5 * np.tanh(bp_delta / 3.0)
        self._prev_bp_gain_min_dB = min_bp_dB
        sense_reward = float(np.clip(bp_score + bp_shaping, -0.5, 1.5))

        sinr_vals   = self._compute_sinr_miso()
        sinr_dB     = 10 * np.log10(sinr_vals + 1e-12)
        min_sinr_dB = float(np.min(sinr_dB))
        sinr_score  = np.clip((min_sinr_dB + 10.0) / 35.0, 0.0, 1.0)
        cur_sum_rate = float(np.sum(user_rates))
        rate_score   = np.clip(cur_sum_rate / 30.0, 0.0, 1.0)
        sinr_delta   = min_sinr_dB - self._prev_sinr_min_dB
        sinr_shaping = 0.5 * np.tanh(sinr_delta / 2.0)
        self._prev_sinr_min_dB = min_sinr_dB
        rate_delta   = cur_sum_rate - self._prev_sum_rate
        rate_shaping = 0.5 * np.tanh(rate_delta / 0.5)
        self._prev_sum_rate = cur_sum_rate
        sinr_margin  = sinr_dB - self.sinr_threshold_dB
        qos_penalty  = float(np.mean(np.maximum(0.0, -sinr_margin) / 10.0))
        jain_num     = np.sum(user_rates) ** 2
        jain_den     = self.K * np.sum(user_rates ** 2) + 1e-12
        fairness     = float(np.clip(jain_num / jain_den, 0.0, 1.0))
        comm_reward  = float(np.clip(
            0.5*sinr_score + 0.3*rate_score + 0.2*fairness +
            sinr_shaping + rate_shaping - 0.5*qos_penalty, -2.0, 3.0))
        alpha_ema = 0.1
        self.recent_comm_performance    = (1-alpha_ema)*self.recent_comm_performance    + alpha_ema*max(comm_reward, 0.)
        self.recent_sensing_performance = (1-alpha_ema)*self.recent_sensing_performance + alpha_ema*max(sense_reward, 0.)
        total_reward = float(np.clip(0.5*comm_reward + 0.5*sense_reward, -2.0, 3.0))
        return comm_reward, sense_reward, total_reward

    def _decompose_rewards(self, comm_reward, sense_reward, total_reward):
        sinr_vals   = self._compute_sinr_miso()
        sinr_dB     = 10 * np.log10(sinr_vals + 1e-12)
        sinr_margin = sinr_dB - self.sinr_threshold_dB
        sinr_term   = float(np.mean(1.0 / (1.0 + np.exp(-0.5 * sinr_margin))))
        bp_gains_dB = self._compute_beampattern_gain_dB()
        cur_min_bp  = float(np.min(bp_gains_dB))
        delta_bp    = np.clip((cur_min_bp - self._prev_bp_gain_min_dB) / 5.0, -1.0, 1.0)
        ris_reward  = float(np.clip(
            0.35*sinr_term + 0.30*float(sense_reward) +
            0.20*float(comm_reward) + 0.15*(delta_bp*0.5+0.5), 0.0, 1.5))
        return {'comm': float(comm_reward), 'power': float(total_reward), 'ris': float(ris_reward)}

    def set_tradeoff_parameter(self, eta):  self.tradeoff_param = FIXED_ETA
    def set_target_tradeoff(self, t_eta):   self.performance_history['target_eta'] = FIXED_ETA
    def set_sinr_threshold(self, th):
        self.sinr_threshold_dB = float(th)
        self.sinr_threshold    = 10 ** (th / 10.0)

    # =================================================================
    #  Reset / Step
    # =================================================================

    def reset(self, training=True):
        self.episode_t       = 0
        self.total_episodes += 1
        if training:
            self._train_episode_count += 1
        self._initialize_users(); self._initialize_targets(); self._initialize_clutter()
        self.H_bs_ris_static = self._build_channel_matrix(self.bs_ant_pos, self.ris_elements).T
        self.H_bs_ris        = self.H_bs_ris_static.copy()
        self.cached_target_rcs = np.array([self._calculate_rcs(t) for t in range(self.T)])
        self._update_dynamic_channels()
        self.user_haps_coverage_flag = np.ones(self.K, dtype=bool)
        self.user_nf_flag            = self.user_haps_coverage_flag
        self.Phi              = np.eye(self.N, dtype=complex)
        self.power_allocation = np.ones(self.K) * FIXED_ETA * self.P_max_W / self.K
        self.tradeoff_param   = FIXED_ETA
        self.last_user_rates  = np.zeros(self.K)
        self._sinr_miso_cache = None
        self._init_beamformer()
        _sinr0 = self._compute_sinr_miso()
        self._prev_sinr_min_dB = float(np.min(10*np.log10(_sinr0 + 1e-12)))
        self._prev_sum_rate    = float(np.sum(np.log2(1.0 + _sinr0)))
        try:
            _bp0 = self._compute_beampattern_gain_dB()
            self._prev_bp_gain_min_dB = float(np.min(_bp0))
        except Exception:
            self._prev_bp_gain_min_dB = 30.0
        H_eff_init   = self._compute_ris_effective_channel()
        P_null_init  = self._compute_null_space_projector(H_eff_init)
        P_radar      = (1 - FIXED_ETA) * self.P_max_W
        W_radar_init = np.zeros((self.M, self.M), dtype=complex)
        self.R_sensing = np.zeros((self.M, self.T), dtype=complex)
        for j in range(self.T):
            a_j  = self._steering_vector_haps_ris(self.target_positions[j])
            a_ns = P_null_init @ a_j; nn_ns = np.linalg.norm(a_ns)
            if nn_ns > 1e-8:
                a_ns_n = a_ns / nn_ns
                W_radar_init += (P_radar/self.T) * np.outer(a_ns_n, a_ns_n.conj())
                self.R_sensing[:, j] = np.sqrt(P_radar/self.T) * a_ns_n
            else:
                nn_j = np.linalg.norm(a_j)
                if nn_j > 1e-8:
                    a_j_n = a_j / nn_j
                    W_radar_init += (P_radar/self.T) * np.outer(a_j_n, a_j_n.conj())
                    self.R_sensing[:, j] = np.sqrt(P_radar/self.T) * a_j_n
        self.cached_W_radar      = W_radar_init
        self.cached_W_comm       = self.G @ self.G.conj().T
        self.cached_W_total      = self.cached_W_comm + W_radar_init
        self.cached_sensing_sinr = np.ones(self.T) * 1e-6
        self.cached_crlb         = 500.0
        if self.mappo_mode:
            return self.get_all_agent_obs()
        return self._get_global_state()

    def step(self, actions):
        self.episode_t        += 1
        self.positions_changed = False
        self._sinr_miso_cache  = None
        user_dists = np.linalg.norm(self.user_positions - self.ris_position, axis=1)
        if self.channel_est_error:
            err = self.est_error_std
            for attr, clean in [('H_bs_ris','H_bs_ris_static'),
                                 ('H_ris_users','H_ris_users_clean'),
                                 ('H_direct','H_direct_clean')]:
                c = getattr(self, clean); sc = np.abs(c).mean()
                setattr(self, attr, c + sc*err*(np.random.randn(*c.shape)+1j*np.random.randn(*c.shape)))
            sc_hu = np.abs(self.H_haps_users).mean()
            self.H_haps_users += sc_hu*err*(np.random.randn(*self.H_haps_users.shape)+1j*np.random.randn(*self.H_haps_users.shape))
        self.cached_target_rcs = np.array([self._calculate_rcs(t) for t in range(self.T)])
        if self.mappo_mode:
            self._apply_actions_multi_agent(actions)
        self._sinr_miso_cache = None
        Ts = int(self.pulse_duration * self.fs)
        H_eff_step = self._compute_ris_effective_channel()
        W, W_comm, W_radar = self._compute_joint_tx_covariance(H_eff=H_eff_step)
        self.cached_W_radar = W_radar.copy()
        self.cached_W_comm  = W_comm.copy()
        self.cached_W_total = W.copy()
        s_comm    = (np.random.randn(self.K, Ts) + 1j*np.random.randn(self.K, Ts))/np.sqrt(2)
        tx_signal = self.G @ s_comm
        echoes    = self._receive_echoes(tx_signal)
        SE, user_rates   = self._compute_spectral_efficiency()
        self.last_user_rates = user_rates.copy()
        radar_sinr       = self._compute_radar_sinr(echoes)
        crlb             = self._compute_sensing_crlb()
        EE               = self._compute_energy_efficiency(SE)
        BP_MSE           = self._compute_beampattern_mse()
        sensing_sinr_vec = self._compute_sensing_sinr()
        qos_satisfied    = self._check_qos_constraints()
        self.cached_sensing_sinr = sensing_sinr_vec
        self.cached_crlb         = crlb
        bp_gains_dB   = self._compute_beampattern_gain_dB()
        min_bp_gain   = float(np.min(bp_gains_dB))
        sinr_per_user = self._compute_sinr_miso()
        min_sinr      = float(np.min(10*np.log10(sinr_per_user + 1e-12)))
        avg_snr = 10*np.log10(np.mean([np.abs(self.H_haps_users[:,k]@self.G[:,k])**2
                                        for k in range(self.K)])/(self.noise_power_w+1e-12))
        outage_prob = float(np.sum(user_rates < self.R_min)/self.K)
        sinr_viol   = float(np.sum(sinr_per_user < self.sinr_threshold)/self.K)
        comm_reward, sense_reward, total_reward = self._compute_rewards(
            SE, user_rates, EE, sensing_sinr_vec, BP_MSE, crlb, FIXED_ETA)
        done = self.episode_t >= self.max_steps
        elev_angles = np.array([
            np.degrees(np.arctan2(
                self.haps_position[2]-self.user_positions[k,2],
                np.linalg.norm(self.user_positions[k,:2]-self.haps_position[:2])+1e-9))
            for k in range(self.K)])
        self._update_users(); self._update_targets()
        if self.episode_t % 10 == 0: self._update_haps()
        self._update_dynamic_channels()
        info = {
            'beampattern_gains_dB':      bp_gains_dB,
            'min_beampattern_gain_dB':   min_bp_gain,
            'min_bp_gain_dB':            min_bp_gain,
            'sinr_per_user_dB':          10*np.log10(sinr_per_user+1e-12),
            'min_sinr_dB':               min_sinr,
            'sinr_threshold_dB':         self.sinr_threshold_dB,
            'sinr_constraint_satisfied': bool(np.all(sinr_per_user >= self.sinr_threshold)),
            'sinr_violation_fraction':   sinr_viol,
            'user_rates':                user_rates,
            'sum_rate':                  SE,
            'energy_efficiency':         EE,
            'outage_probability':        outage_prob,
            'min_user_rate':             float(np.min(user_rates)),
            'radar_sinr':                radar_sinr,
            'beampattern_mse':           BP_MSE,
            'crlb':                      crlb,
            'comm_success':              qos_satisfied,
            'sensing_reward':            sense_reward,
            'comm_reward':               comm_reward,
            'tradeoff_param':            FIXED_ETA,
            'haps_covered_users':        int(np.sum(self.user_haps_coverage_flag)),
            'avg_user_distance':         float(np.mean(user_dists)),
            'avg_snr':                   float(avg_snr),
            'user_positions':            self.user_positions.copy(),
            'target_positions':          self.target_positions.copy(),
            'haps_position':             self.haps_position.copy(),
            'elevation_angles_deg':      elev_angles,
            'total_tx_power_W':          float(np.real(np.trace(W))),
            'power_budget_W':            self.P_max_W,
            'power_utilization':         float(np.real(np.trace(W))/self.P_max_W),
            'zf_alpha':                  self._get_zf_alpha(),
            'train_episode':             self._train_episode_count,
        }
        if self.mappo_mode:
            obs_dict    = self.get_all_agent_obs()
            reward_dict = self._decompose_rewards(comm_reward, sense_reward, total_reward)
            done_dict   = {a: done for a in self.agent_names}
            done_dict['__all__'] = done
            return obs_dict, reward_dict, done_dict, info
        return self._get_global_state(), total_reward, done, info

    # =================================================================
    #  MAPPO structure
    # =================================================================
    def _setup_mappo_structure(self):
        self.action_dims = {'comm': 2*self.M*self.K, 'power': self.K, 'ris': self.N}
        self.shared_obs_components = {
            'chan_stats': 8, 'system_params': 4, 'multipath_info': 2,
            'performance_info': 3, 'haps_state': 3, 'sinr_margin_info': 2}
        self.shared_obs_dim = sum(self.shared_obs_components.values())
        self.agent_obs_components = {
            'comm':  {'user_channel_quality': self.K, 'user_rates': self.K, 'ris_summary': 3},
            'power': {'user_channel_quality': self.K, 'user_rates': self.K, 'beamformer_power': self.K},
            'ris':   {'target_distances': self.T, 'user_distances_bs': self.K,
                      'user_distances_ris': self.K, 'target_rcs_norm': self.T, 'elevation_angles': self.K},
        }
        self.obs_dims = {}
        for agent in self.agent_names:
            specific_dim = sum(self.agent_obs_components[agent].values())
            self.obs_dims[agent] = self.shared_obs_dim + specific_dim
        self.global_state_dim = (self.shared_obs_dim +
                                  self.K + self.K + self.T +
                                  self.K + self.K + self.T +
                                  self.K + 3 + self.K)

    def get_shared_obs(self):
        sensing_sinr_mean = float(np.mean(self.cached_sensing_sinr)/20.) if hasattr(self,'cached_sensing_sinr') else 0.
        crlb_norm = float(np.clip(self.cached_crlb/1000.,0,1)) if hasattr(self,'cached_crlb') else .5
        sinr_vals = self._compute_sinr_miso()
        sinr_dB   = 10*np.log10(sinr_vals+1e-12)
        msm = float(np.clip(np.mean(sinr_dB-self.sinr_threshold_dB)/20.,-1,1))
        msmin = float(np.clip(np.min(sinr_dB-self.sinr_threshold_dB)/20.,-1,1))
        chan_stats = np.array([
            np.mean(np.abs(self.H_haps_users)), np.std(np.abs(self.H_haps_users)),
            np.mean(np.abs(self.H_ris_users)),  np.std(np.abs(self.H_ris_users)),
            np.mean(np.abs(self.H_direct)),      np.std(np.abs(self.H_direct)),
            sensing_sinr_mean, crlb_norm], dtype=np.float32)
        system_params = np.array([FIXED_ETA, self.P_max_W/1000.,
                                   float(self.episode_t)/self.max_steps,
                                   float(np.sum(self.user_haps_coverage_flag))/self.K],
                                  dtype=np.float32)
        multipath_info = np.zeros(2, dtype=np.float32)
        if self.multipath_components:
            multipath_info[0] = float(np.mean([p['attenuation'] for p in self.multipath_components]))
            multipath_info[1] = len(self.multipath_components)/3.
        performance_info = np.array([self.recent_comm_performance,
                                      self.recent_sensing_performance, FIXED_ETA], dtype=np.float32)
        haps_state = np.array([self.haps_position[0]/1000., self.haps_position[1]/1000.,
                                self.bs_to_ris_distance/1000.], dtype=np.float32)
        sinr_margin_info = np.array([msm, msmin], dtype=np.float32)
        shared = np.concatenate([chan_stats, system_params, multipath_info,
                                   performance_info, haps_state, sinr_margin_info]).astype(np.float32)
        assert len(shared) == self.shared_obs_dim
        return shared

    def get_comm_agent_obs(self):
        shared = self.get_shared_obs()
        ucq    = np.array([np.linalg.norm(self.H_haps_users[:,k]) for k in range(self.K)], dtype=np.float32)
        ucq   /= (np.max(ucq)+1e-6)
        rates  = np.clip(self.last_user_rates/10.,0,1).astype(np.float32)
        eff_Phi = self.coupling_matrix@self.Phi if self.mutual_coupling else self.Phi
        ris_ch  = self.H_ris_users.conj().T @ eff_Phi @ self.H_bs_ris
        ris_gains = np.linalg.norm(ris_ch, axis=1)
        ris_summary = np.array([np.mean(ris_gains),np.std(ris_gains),np.max(ris_gains)], dtype=np.float32)
        return np.concatenate([shared,ucq,rates,ris_summary]).astype(np.float32)

    def get_power_agent_obs(self):
        shared = self.get_shared_obs()
        ucq    = np.array([np.linalg.norm(self.H_haps_users[:,k]) for k in range(self.K)], dtype=np.float32)
        ucq   /= (np.max(ucq)+1e-6)
        rates  = np.clip(self.last_user_rates/10.,0,1).astype(np.float32)
        bf_pwr = (np.linalg.norm(self.G,axis=0)/np.sqrt(self.P_max_W)).astype(np.float32)
        return np.concatenate([shared,ucq,rates,bf_pwr]).astype(np.float32)

    def get_ris_agent_obs(self):
        shared     = self.get_shared_obs()
        tgt_dists  = (self.ris_to_target_distances/max(self.haps_altitude,1.)).astype(np.float32)
        udists_bs  = (np.linalg.norm(self.user_positions-self.haps_position,axis=1)/max(self.haps_coverage_boundary,1.)).astype(np.float32)
        udists_ris = (np.linalg.norm(self.user_positions-self.ris_position,axis=1)/max(self.haps_altitude,1.)).astype(np.float32)
        tgt_rcs    = (self.cached_target_rcs/10.).astype(np.float32)
        elev       = np.array([np.degrees(np.arctan2(
            self.haps_position[2]-self.user_positions[k,2],
            np.linalg.norm(self.user_positions[k,:2]-self.haps_position[:2])+1e-9))/90.
            for k in range(self.K)], dtype=np.float32)
        return np.concatenate([shared,tgt_dists,udists_bs,udists_ris,tgt_rcs,elev]).astype(np.float32)

    def get_all_agent_obs(self):
        return {'comm': self.get_comm_agent_obs(),
                'power': self.get_power_agent_obs(),
                'ris': self.get_ris_agent_obs()}

    def _get_global_state(self):
        shared     = self.get_shared_obs()
        ucq        = np.array([np.linalg.norm(self.H_haps_users[:,k]) for k in range(self.K)], dtype=np.float32)
        ucq       /= (np.max(ucq)+1e-6)
        rates      = np.clip(self.last_user_rates/10.,0,1).astype(np.float32)
        tgt_dists  = (self.ris_to_target_distances/max(self.haps_altitude,1.)).astype(np.float32)
        udists_bs  = (np.linalg.norm(self.user_positions-self.haps_position,axis=1)/max(self.haps_coverage_boundary,1.)).astype(np.float32)
        udists_ris = (np.linalg.norm(self.user_positions-self.ris_position,axis=1)/max(self.haps_altitude,1.)).astype(np.float32)
        tgt_rcs    = (self.cached_target_rcs/10.).astype(np.float32)
        bf_pwr     = (np.linalg.norm(self.G,axis=0)/np.sqrt(self.P_max_W)).astype(np.float32)
        eff_Phi    = self.coupling_matrix@self.Phi if self.mutual_coupling else self.Phi
        ris_ch     = self.H_ris_users.conj().T @ eff_Phi @ self.H_bs_ris
        ris_gains  = np.linalg.norm(ris_ch, axis=1)
        ris_summary = np.array([np.mean(ris_gains),np.std(ris_gains),np.max(ris_gains)], dtype=np.float32)
        elev_gs    = np.array([np.degrees(np.arctan2(
            self.haps_position[2]-self.user_positions[k,2],
            np.linalg.norm(self.user_positions[k,:2]-self.haps_position[:2])+1e-9))/90.
            for k in range(self.K)], dtype=np.float32)
        state = np.concatenate([shared,ucq,rates,tgt_dists,udists_bs,
                                  udists_ris,tgt_rcs,bf_pwr,ris_summary,elev_gs]).astype(np.float32)
        assert len(state) == self.global_state_dim
        return state

    def decompose_state(self, state):
        assert len(state) == self.global_state_dim
        K=self.K; T=self.T; S=self.shared_obs_dim; idx=0
        shared=state[idx:idx+S]; idx+=S
        ucq=state[idx:idx+K]; idx+=K
        rates=state[idx:idx+K]; idx+=K
        tgt_dists=state[idx:idx+T]; idx+=T
        udists_bs=state[idx:idx+K]; idx+=K
        udists_ris=state[idx:idx+K]; idx+=K
        tgt_rcs=state[idx:idx+T]; idx+=T
        bf_pwr=state[idx:idx+K]; idx+=K
        ris_summary=state[idx:idx+3]; idx+=3
        elev_gs=state[idx:idx+K]; idx+=K
        return {'shared':shared,'ucq':ucq,'rates':rates,'tgt_dists':tgt_dists,
                'udists_bs':udists_bs,'udists_ris':udists_ris,'tgt_rcs':tgt_rcs,
                'bf_pwr':bf_pwr,'ris_summary':ris_summary,'elev_gs':elev_gs}

    @property
    def observation_space(self):
        try:   from gymnasium import spaces
        except: from gym import spaces
        return {a: spaces.Box(low=-np.inf,high=np.inf,shape=(self.obs_dims[a],),dtype=np.float32)
                for a in self.agent_names}

    @property
    def action_space(self):
        try:   from gymnasium import spaces
        except: from gym import spaces
        return {a: spaces.Box(low=-1.,high=1.,shape=(self.action_dims[a],),dtype=np.float32)
                for a in self.agent_names}

    def validate_env(self, verbose=True):
        if verbose:
            print(f"=== validate_env ({'UAV' if self.uav_mode else 'HAPS'} mode) ===")
        obs = self.reset(training=False)
        assert isinstance(obs, dict)
        for agent in self.agent_names:
            assert agent in obs
            assert obs[agent].shape[0] == self.obs_dims[agent]
            assert not np.any(np.isnan(obs[agent]))
        gs = self._get_global_state(); assert gs.shape[0] == self.global_state_dim
        rng_actions = {a: np.random.uniform(-1,1,self.action_dims[a]).astype(np.float32)
                       for a in self.agent_names}
        obs2, reward_dict, done_dict, info = self.step(rng_actions)
        for agent in self.agent_names:
            assert obs2[agent].shape[0] == self.obs_dims[agent]
            assert not np.any(np.isnan(obs2[agent]))
            assert np.isfinite(reward_dict[agent])

            P_direct = np.mean(np.abs(self.H_haps_users)**2)
            H_refl   = self.H_bs_ris.conj().T @ self.H_ris_users
            P_refl   = np.mean(np.abs(H_refl)**2)

            d_HR = float(np.mean(np.linalg.norm(
                self.bs_ant_pos[:,None,:] - self.ris_elements[None,:,:], axis=2)))
            d_RU = float(np.mean(np.linalg.norm(
                self.ris_elements[:,None,:] - self.user_positions[None,:,:], axis=2)))
            d_HU = float(np.mean(np.linalg.norm(
                self.bs_ant_pos[:,None,:] - self.user_positions[None,:,:], axis=2)))
            beta_HR = (self.d0/max(d_HR,1.))**self.path_loss_exp
            beta_RU = (self.d0/max(d_RU,1.))**self.path_loss_exp
            beta_HU = (self.d0/max(d_HU,1.))**self.path_loss_exp
            ris_gain_dB = 10*np.log10(self.N**2 * beta_HR * beta_RU / (beta_HU + 1e-30))
        return True

    def get_benchmark_metrics(self):
        _, user_rates = self._compute_spectral_efficiency()
        sinr = self._compute_sinr_miso()
        sinr_dB = 10*np.log10(sinr+1e-12)
        bp_gains_dB = self._compute_beampattern_gain_dB()
        return {
            'per_target_bp_gain_dB': bp_gains_dB,
            'min_bp_gain_dB':        float(np.min(bp_gains_dB)),
            'mean_bp_gain_dB':       float(np.mean(bp_gains_dB)),
            'per_user_rate_bps_hz':  user_rates,
            'min_user_rate':         float(np.min(user_rates)),
            'sum_rate':              float(np.sum(user_rates)),
            'per_user_sinr_dB':      sinr_dB,
            'min_sinr_dB':           float(np.min(sinr_dB)),
            'mean_sinr_dB':          float(np.mean(sinr_dB)),
            'sinr_threshold_dB':     self.sinr_threshold_dB,
            'ris_active':            not self.uav_mode,
            'N_ris':                 self.N,
        }


HAPS_MIMO_ISAC = HAPS_RIS_MIMO_ISAC