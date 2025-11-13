"""
simulationPPO.py
-----------------
Collection of small vehicle simulation tools and a Gym environment used to train
and evaluate a PPO agent for energy-efficient driving on a recorded track.

Principales composantes :
- Dataclasses physiques : Corp, Roue, Moteur, Condition, Voiture
- Préparation de piste à partir d'un CSV (`Piste`)
- Environnement Gym `VehicleEnv` (observations: [v, pos, énergie, temps])
- Boucle d'entraînement PPO + évaluation et utilitaires de visualisation

Le fichier est principalement orienté vers la recherche/visualisation. Les
fonctions et classes possèdent des docstrings donnant le contrat d'entrée/sortie.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from cma import fmin2
from dataclasses import dataclass
from typing import Dict, Union
import gymnasium as gym
from gymnasium import spaces
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import tempfile
from stable_baselines3.common.monitor import Monitor
import pandas as pd

# --- gear ratio from sprocket teeth (motor=20, wheel=100) ---
motor_teeth = 20
wheel_teeth = 100
gear_ratio_from_teeth = float(wheel_teeth) / float(motor_teeth)  # = 5.0

# --- Training defaults (modifiable) ---
GENERATIONS = 10               # nombre de générations / checkpoints à entraîner
TIMESTEPS_PER_GEN = 1000       # timesteps par génération (augmente pour vrai entraînement)
EVAL_EPISODES = 5              # épisodes d'évaluation par génération


@dataclass
class Corp:
    """Representation du carénage (corp) + pilote.

    Args:
        car_mass: masse du véhicule sans le pilote (kg).
        pilot_mass: masse du pilote (kg).
        CdA_coeff: soit un float constant pour CdA, soit un dict angle_deg->CdA
            pour interpolation en fonction de l'angle de braquage.

    Attributs calculés:
        mass: masse totale (car + pilot) en kg
        _angles_deg, _cdas: si CdA_coeff est dict, tableaux pour interpolation
    """
    car_mass: float
    pilot_mass: float
    CdA_coeff: Union[float, Dict[float, float]]

    def __post_init__(self):
        self.mass: float = float(self.car_mass + self.pilot_mass)
        if isinstance(self.CdA_coeff, dict):
            if len(self.CdA_coeff) < 2:
                raise ValueError("CdA_coeff dict doit contenir au moins 2 points pour interpoler.")
            items = sorted(self.CdA_coeff.items())
            self._angles_deg = np.array([k for k, _ in items], dtype=float)
            self._cdas = np.array([v for _, v in items], dtype=float)
        else:
            self._angles_deg = None
            self._cdas = None

    def CdA(self, turning_angle_deg: Union[float, np.ndarray]) -> np.ndarray:
        """Retourne la valeur(s) de CdA pour l'angle donné.

        turning_angle_deg peut être un scalaire ou un tableau. Si `CdA_coeff`
        était fourni comme dict, on interpole linéairement entre les points.
        """
        ang = np.asarray(turning_angle_deg, dtype=float)
        if self._angles_deg is None:
            return np.full_like(ang, fill_value=float(self.CdA_coeff), dtype=float)
        return np.interp(ang, self._angles_deg, self._cdas,
                         left=self._cdas[0], right=self._cdas[-1])


@dataclass
class Roue:
    """Paramètres d'une roue.

    Args:
        rayon: rayon en mètres
        mass: masse de la roue (kg)
        coeff_rr: coefficient de roulement (dimensionné)
        coeff_linear, coeff_alpha: coefficients additionnels si besoin
        T_b: moment de frottement au palier (Nm)
        inertia: inertie rotorique si connue (kg·m²); calculée si None
    """
    rayon: float
    mass: float
    coeff_rr: float
    coeff_linear: float
    coeff_alpha: float
    T_b: float
    inertia: float = None

    def __post_init__(self):
        # calcul d'inertie approximative si non fournie (disque plein)
        if self.inertia is None:
            self.inertia = 0.5 * self.mass * (self.rayon ** 2)
        if self.rayon <= 0:
            raise ValueError("rayon doit être > 0")
        if self.mass <= 0:
            raise ValueError("mass doit être > 0")
        if self.coeff_rr < 0:
            raise ValueError("coeff_rr doit être >= 0")
        if self.T_b < 0:
            raise ValueError("T_b (bearing frictional moment) doit être >= 0")


@dataclass
class Moteur:
    """Paramètres électriques et mécaniques du moteur.

    Args:
        gear_ratio: rapport de démultiplication (wheel_rpm * gear_ratio = motor_rpm)
        inertia: inertie du rotor (kg·m²)
        torque_constant: constante de couple Kt (Nm/A)
        resistance: résistance statorique (Ohm)
        engine_cutoff_rpm: limite mécanique/électronique du moteur (rpm)
        transmission_efficiency: rendement mécanique entre moteur et roue (0..1)
        max_motor_torque: (optionnel) couple maximal moteur (Nm)
    """
    gear_ratio: float
    inertia: float
    torque_constant: float
    resistance: float
    engine_cutoff_rpm: int
    transmission_efficiency: float
    max_motor_torque: float = None 

    def __post_init__(self):
        if self.gear_ratio <= 0:
            raise ValueError("gear_ratio doit être > 0 (motor_rpm / wheel_rpm).")
        if not (0 < self.transmission_efficiency <= 1.0):
            raise ValueError("transmission_efficiency doit être dans (0, 1].")
        if self.max_motor_torque is not None and self.max_motor_torque < 0:
            raise ValueError("max_motor_torque doit être >= 0 ou None.")

    def wheel_torque_from_motor(self, motor_torque_nm: float) -> float:
        """Retourne le couple appliqué à la roue (Nm) compte tenu du rendement et du rapport."""
        return motor_torque_nm * self.transmission_efficiency * self.gear_ratio

    def motor_rpm_from_wheel(self, wheel_rpm: float) -> float:
        """Convertit wheel_rpm -> motor_rpm via le gear_ratio."""
        return wheel_rpm * self.gear_ratio

    def limited_motor_rpm(self, requested_motor_rpm: float) -> float:
        """Applique la limite mécanique maximum (cutoff) au régime demandé."""
        return min(requested_motor_rpm, float(self.engine_cutoff_rpm))

    def max_vehicle_speed_mps(self, wheel_radius_m: float) -> float:
        """Estime la vitesse max du véhicule (m/s) liée à la coupure moteur.

        Formule: motor_rpm_max -> wheel angular speed -> lin. speed
        """
        return (self.engine_cutoff_rpm / 60.0) * (2.0 * np.pi * wheel_radius_m) / self.gear_ratio


@dataclass
class Condition:
    """Conditions atmosphériques utiles pour calcul des résistances.

    Args:
        air_pressure: pression (kPa par défaut, ou Pa si pressure_unit='Pa')
        air_temperature_C: température en °C
        wind_speed: vitesse du vent (m/s)
        wind_direction_deg: direction du vent en degrés
        pressure_unit: 'kPa' ou 'Pa'

    Attributs calculés:
        rho: densité de l'air (kg/m³)
        wind_vector: vecteur unité du vent
        wind_velocity: vecteur vitesse du vent (m/s)
    """
    air_pressure: float
    air_temperature_C: float
    wind_speed: float
    wind_direction_deg: float
    pressure_unit: str = "kPa"

    def __post_init__(self):
        if self.pressure_unit.lower() == "kpa":
            p_pa = float(self.air_pressure) * 1e3
        elif self.pressure_unit.lower() == "pa":
            p_pa = float(self.air_pressure)
        else:
            raise ValueError("pressure_unit doit être 'kPa' ou 'Pa'.")

        T_K = float(self.air_temperature_C) + 273.15
        R = 287.058
        self.rho = p_pa / (R * T_K)
        self.wind_direction_rad = np.deg2rad(self.wind_direction_deg)
        phi_from = self.wind_direction_rad
        phi_to = (phi_from + np.pi) % (2*np.pi)
        self.wind_vector = np.array([
            np.sin(phi_to),
            np.cos(phi_to)
        ], dtype=float)
        self.wind_velocity = self.wind_vector * float(self.wind_speed)


@dataclass
class Voiture:
    """Objet véhicule agrégant corp, roue et moteur.

    Fournit des propriétés dérivées utiles pour la simulation (masse totale,
    inertie équivalente, vitesse angulaire des roues, etc.).
    """
    corp: "Corp"
    roue: "Roue"
    moteur: "Moteur"
    n_wheels: int = 3
    position_init: float = 0.0
    speed_init: float = 0.0
    acceleration_init: float = 0.0

    def __post_init__(self):
        if self.n_wheels <= 0:
            raise ValueError("n_wheels doit être ≥ 1")
        m_total = self.corp.mass + self.n_wheels * self.roue.mass
        I_trans = m_total * (self.roue.rayon ** 2)
        self.mass = m_total
        self.inertia = I_trans + self.n_wheels * self.roue.inertia
        self.adjusted_mass = self.inertia / (self.roue.rayon ** 2)
        self.position = float(self.position_init)
        self.speed = float(self.speed_init)
        self.acceleration = float(self.acceleration_init)
        self.value_position = [self.position]
        self.value_speed = [self.speed]
        self.value_acceleration = [self.acceleration]

    def wheel_angular_speed_rad_s(self) -> float:
        """Vitesse angulaire des roues (rad/s) d'après la vitesse linéaire du véhicule."""
        return self.speed / self.roue.rayon

    def wheel_rpm(self) -> float:
        """Régime roue en tours par minute (RPM)."""
        return self.wheel_angular_speed_rad_s() * 60.0 / (2*np.pi)


class Piste:
    def __init__(self, filepath: str):
        """Charge une piste depuis un CSV et calcule des propriétés géométriques.

        Le CSV attendu doit contenir au minimum les colonnes suivantes (noms
        insensibles à la casse) : 'latitude', 'longitude', 'slope (%)',
        'distance (m)'. La classe calcule la position cartésienne (x,y) autour du
        premier point, la pente, l'abscisse curviligne cumulée `s_m` et le rayon
        de courbure estimé le long de la piste (`rayonCourbure`).

        Args:
            filepath: chemin vers le fichier CSV de piste.
        """
        self.filepath = filepath
        df = pd.read_csv(filepath)
        df.columns = [str(c).strip().lower() for c in df.columns]
        for c in list(df.columns):
            if c.startswith("unnamed"):
                df = df.drop(columns=[c])
        df = df[['latitude', 'longitude', 'slope (%)', 'distance (m)']].dropna()
        lat = df['latitude'].to_numpy(float)
        lon = df['longitude'].to_numpy(float)
        slope = df['slope (%)'].to_numpy(float)
        dcol = df['distance (m)'].to_numpy(float)
        if np.all(np.diff(dcol) >= -1e-12):
            s_cum = dcol.copy()
        else:
            s_cum = np.cumsum(dcol)
        ds = np.diff(s_cum, prepend=s_cum[0])
        keep = np.ones_like(s_cum, dtype=bool)
        keep[1:] = ds[1:] > 0.0
        if not np.all(keep):
            s_cum = s_cum[keep]
            lat   = lat[keep]
            lon   = lon[keep]
            slope = slope[keep]
        ds = np.diff(s_cum, prepend=s_cum[0])
        eps_s = max(1e-6, np.nanmedian(ds[1:]) * 1e-9 if ds.size > 1 else 1e-6)
        for i in range(1, len(s_cum)):
            if s_cum[i] <= s_cum[i-1]:
                s_cum[i] = s_cum[i-1] + eps_s
        self.s_m = s_cum.astype(float)
        self.slope = slope.astype(float)
        N = self.s_m.size
        if N < 5:
            raise ValueError("La piste doit contenir au moins 5 points pour calculer la courbure.")
        lat0 = np.deg2rad(lat[0])
        lon0 = np.deg2rad(lon[0])
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        R_earth = 6_371_000.0
        dlat = lat_rad - lat0
        dlon = lon_rad - lon0
        self.x = (R_earth * np.cos(lat0) * dlon).astype(float)
        self.y = (R_earth * dlat).astype(float)
        t = np.arange(N, dtype=float)
        dx_dt = np.gradient(self.x, t, edge_order=2)
        dy_dt = np.gradient(self.y, t, edge_order=2)
        ds_dt = np.gradient(self.s_m, t, edge_order=2)
        small = 1e-12
        inv_ds_dt = np.divide(1.0, ds_dt, out=np.zeros_like(ds_dt), where=np.abs(ds_dt) > small)
        dx_ds = dx_dt * inv_ds_dt
        dy_ds = dy_dt * inv_ds_dt
        ddx_dt = np.gradient(dx_ds, t, edge_order=2)
        ddy_dt = np.gradient(dy_ds, t, edge_order=2)
        ddx_ds = ddx_dt * inv_ds_dt
        ddy_ds = ddy_dt * inv_ds_dt
        num = np.abs(dx_ds * ddy_ds - dy_ds * ddx_ds)
        den = (dx_ds*dx_ds + dy_ds*dy_ds)
        den_pow = np.power(den, 1.5, where=den>small)
        kappa = np.divide(num, den_pow, out=np.zeros_like(num), where=den > small)
        with np.errstate(divide='ignore', invalid='ignore'):
            rayon = np.divide(1.0, kappa, out=np.full_like(kappa, np.inf), where=kappa > small)
        R_CAP = 1e9
        self.rayonCourbure = np.clip(rayon, 0.0, R_CAP).astype(float)
    # rayonCourbure contient des valeurs finies plafonnées. Une valeur
    # très grande (R_CAP) signifie courbure proche de 0 (segment presque
    # rectiligne).


# ==============================================================
#  ENVIRONNEMENT GYM : VehicleEnv
# ==============================================================

class VehicleEnv(gym.Env):
    """
    Environnement Gym pour un véhicule électrique sur piste.
    Observations: [vitesse, position, énergie, temps]
    Action: throttle ∈ [0,1]
    """
    def __init__(self, voiture: "Voiture", track_length_m=3832.0, laps=4, dt=0.1):
        super().__init__()
        self.voiture = voiture
        self.dt = float(dt)
        self.track_length = float(track_length_m)
        self.total_length = float(track_length_m * laps)
        self.max_time = 2070.0
        self.energy_Wh = 0.0
        self.time = 0.0

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([60.0, self.total_length, 1e6, self.max_time], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.voiture.speed = float(self.voiture.speed_init)
        self.voiture.position = float(self.voiture.position_init)
        self.energy_Wh = 0.0
        self.time = 0.0
        # state for torque ramp / stuck detection / coast assist
        self._prev_T_motor_applied = 0.0
        self._stuck_steps = 0
        self._last_pos = float(self.voiture.position)
        # config tunables (peux modifier)
        self.aux_power_W = getattr(self, "aux_power_W", 10.0)               # consommation auxiliaire [W]
        # torque ramp chosen from motor peak (3.81 Nm) / desired trise (0.05 s) -> ~76 Nm/s
        self.torque_ramp_rate = getattr(self, "torque_ramp_rate", 76.2)    # Nm/s max variation (tuneable)
        self.coast_assist_fraction = getattr(self, "coast_assist_fraction", 0.25)  # max fraction of max torque used to oppose resistances when throttle==0
        self.coast_enabled = getattr(self, "coast_enabled", True)
        obs = np.array([self.voiture.speed, self.voiture.position, self.energy_Wh, self.time], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # robust action input
        a_np = np.asarray(action)
        if a_np.ndim == 0:
            throttle = float(np.clip(a_np, 0.0, 1.0))
        else:
            throttle = float(np.clip(a_np.ravel()[0], 0.0, 1.0))

        v = float(self.voiture.speed)

        # Resistances
        rho = 1.225
        CdA_val = float(np.atleast_1d(self.voiture.corp.CdA(0))[0])
        F_drag = 0.5 * rho * CdA_val * (v ** 2)
        F_roll = self.voiture.mass * 9.81 * self.voiture.roue.coeff_rr


        F_resist = F_drag + F_roll

        # Motor commanded torque
        T_motor_max = float(getattr(self.voiture.moteur, "max_motor_torque",
                                    getattr(self.voiture.moteur, "torque_constant_max", 30.0)))
        T_motor_cmd = throttle * T_motor_max

        # Apply torque ramp (limit instantaneous change)
        max_delta = self.torque_ramp_rate * self.dt
        prev = getattr(self, "_prev_T_motor_applied", 0.0)
        T_motor_applied = float(np.clip(T_motor_cmd, prev - max_delta, prev + max_delta))

        # If coasting enabled and throttle==0, optionally provide small assist torque to oppose resistances
        if self.coast_enabled and throttle <= 1e-6 and v > 0.01:
            # wheel torque needed to overcome resistances
            T_wheel_req = F_resist * self.voiture.roue.rayon
            # motor torque required (before efficiency & gear)
            denom = max(1e-9, (self.voiture.moteur.transmission_efficiency * self.voiture.moteur.gear_ratio))
            T_motor_req = T_wheel_req / denom
            # allow only a fraction of max torque as "cruise assist" (realistic controllers may provide some)
            T_assist = min(T_motor_req, self.coast_assist_fraction * T_motor_max)
            # ensure applied torque at least assist (but keep ramp limit)
            T_motor_applied = float(max(T_motor_applied, min(prev + max_delta, T_assist)))

        # store for next step
        self._prev_T_motor_applied = T_motor_applied

        # Wheel torque and traction force
        T_wheel = self.voiture.moteur.wheel_torque_from_motor(T_motor_applied)
        F_engine = float(T_wheel / max(1e-12, self.voiture.roue.rayon))

        # Net force -> acceleration
        F_net = F_engine - F_resist
        a = F_net / self.voiture.mass

        # integrate
        v_new = max(v + a * self.dt, 0.0)
        delta_s = 0.5 * (v + v_new) * self.dt

        # speed limit
        try:
            v_max = float(self.voiture.moteur.max_vehicle_speed_mps(self.voiture.roue.rayon))
        except Exception:
            v_max = np.inf
        if v_new > v_max:
            v_new = v_max
            delta_s = 0.5 * (v + v_new) * self.dt

        self.voiture.position += delta_s
        self.voiture.speed = v_new
        self.time += self.dt

        finished = False
        if self.voiture.position >= self.total_length:
            self.voiture.position = float(self.total_length)
            finished = True

        # Energy calculation (motor shaft + copper losses + aux)
        wheel_omega = v_new / max(1e-12, self.voiture.roue.rayon)    # rad/s
        motor_omega = wheel_omega * self.voiture.moteur.gear_ratio  # rad/s
        P_mech_motor = T_motor_applied * motor_omega   # W (deliver positive power)
        Kt = max(1e-9, float(getattr(self.voiture.moteur, "torque_constant", 1e-3)))
        I_motor = abs(T_motor_applied) / Kt
        P_cu = (I_motor ** 2) * float(getattr(self.voiture.moteur, "resistance", 0.0))
        inverter_eff = float(getattr(self.voiture.moteur, "inverter_efficiency", 0.97))
        P_aux = float(getattr(self, "aux_power_W", 10.0))
        # clamp regen: no regen model -> ignore negative P_mech_motor (no feed-in)
        P_elec = max(0.0, P_mech_motor / max(1e-6, inverter_eff)) + P_cu + P_aux

        E_wh = P_elec * (self.dt / 3600.0)
        if E_wh > 0.0:
            self.energy_Wh += E_wh

        # Reward & termination (tunable)
        reward = - (E_wh * 0.1)
        stuck_speed_thresh = 0.5
        if v_new < stuck_speed_thresh:
            reward -= 1.0

        # stuck detection
        pos_delta = abs(self.voiture.position - getattr(self, "_last_pos", self.voiture.position))
        self._last_pos = float(self.voiture.position)
        if pos_delta < 1e-3:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        truncated = False
        terminated = False
        if (self._stuck_steps * self.dt) >= 5.0:
            reward -= 2000.0
            truncated = True
        elif finished:
            reward += 10000.0
            terminated = True
        elif self.time > float(self.max_time):
            reward -= 1000.0
            truncated = True

        obs = np.array([self.voiture.speed, self.voiture.position, self.energy_Wh, self.time], dtype=np.float32)
        return obs, float(reward), terminated, truncated, {}

    def render(self):
        print(f"t={self.time:.1f}s | v={self.voiture.speed:.2f} m/s | pos={self.voiture.position:.1f} m | E={self.energy_Wh:.2f} Wh")


def run_episode_and_record(model, env, max_steps=20000):
    """Fait tourner un épisode avec `model` sur `env` et collecte des séries temporelles.

    Args:
        model: modèle Stable-Baselines (doit implémenter `predict`)
        env: instance d'environnement Gym/Gymnasium
        max_steps: protection contre boucles infinies

    Retourne un dictionnaire contenant arrays pour times, speeds, positions,
    energies, rewards, actions, et des métriques agrégées 'total_reward',
    'total_energy', 'total_distance_m'.
    """
    reset_res = env.reset()
    obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
    obs = np.asarray(obs).squeeze()

    # record initial state (t=0) so first point shows v=0 if reset sets it so
    times, speeds, positions, energies, rewards_list, actions = [], [], [], [], [], []
    t0 = float(getattr(env, "time", 0.0))
    v0 = float(getattr(env.voiture, "speed", (obs[0] if obs.size>0 else np.nan)))
    pos0 = float(getattr(env.voiture, "position", (obs[1] if obs.size>1 else np.nan)))
    E0 = float(getattr(env, "energy_Wh", np.nan))
    times.append(t0); speeds.append(v0); positions.append(pos0); energies.append(E0)

    total_reward = 0.0
    for step_idx in range(int(max_steps)):
        try:
            action, _ = model.predict(obs, deterministic=True)
        except Exception:
            action, _ = model.predict(np.expand_dims(obs, 0), deterministic=True)
        a_np = np.asarray(action)
        if a_np.ndim == 0:
            action_to_env = np.array([float(a_np)])
        else:
            action_to_env = a_np.ravel()[:1]

        step_res = env.step(action_to_env)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
            done = bool(terminated or truncated)
        elif len(step_res) == 4:
            obs, reward, done, info = step_res
            done = bool(done)
        else:
            raise ValueError(f"Unexpected step output length: {len(step_res)}")

        obs = np.asarray(obs).squeeze()
        t = float(getattr(env, "time", step_idx * getattr(env, "dt", 1.0)))
        v = float(getattr(getattr(env, "voiture", None), "speed", (obs[0] if obs.size>0 else np.nan)))
        pos = float(getattr(getattr(env, "voiture", None), "position", (obs[1] if obs.size>1 else np.nan)))
        E = float(getattr(env, "energy_Wh", np.nan))

        times.append(t); speeds.append(v); positions.append(pos); energies.append(E)
        rewards_list.append(float(np.array(reward).sum()))
        actions.append(np.asarray(action_to_env).copy())
        total_reward += float(np.array(reward).sum())

        if done:
            break

    total_energy_wh = float(energies[-1]) if len(energies) > 0 else float("nan")
    total_distance_m = float(positions[-1]) if len(positions) > 0 else float("nan")
    return {
        "times": np.array(times, dtype=float),
        "speeds": np.array(speeds, dtype=float),
        "positions": np.array(positions, dtype=float),
        "energies": np.array(energies, dtype=float),
        "rewards": np.array(rewards_list, dtype=float),
        "actions": np.array(actions, dtype=object),
        "total_reward": total_reward,
        "total_energy": total_energy_wh,
        "total_distance_m": total_distance_m
    }


def _ckpt_key(path):
    """Extraire un index de checkpoint à partir du nom de fichier.

    Exemple : 'ppo_checkpoint_gen_3.zip' -> retourne 3. Si l'extraction échoue
    retourne 0.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")
    try:
        return int(parts[-1])
    except Exception:
        return 0

ckpt_files = sorted(glob.glob("ppo_checkpoint_gen_*.zip"), key=_ckpt_key)
if os.path.exists("ppo_final_model.zip"):
    ckpt_files.append("ppo_final_model.zip")

if len(ckpt_files) == 0:
    print("Aucun checkpoint trouvé (ppo_checkpoint_gen_*.zip ou ppo_final_model.zip). L'évaluation sera ignorée.")
# optional: limit how many checkpoints to evaluate to avoid long runs during debugging
MAX_EVAL_CKPTS = 5
ckpt_files = ckpt_files[:MAX_EVAL_CKPTS]
print(f"Checkpoints to evaluate ({len(ckpt_files)}): {ckpt_files}")

# --- Initialize results list BEFORE using it in the loop ---
results = []

# In evaluation loop where env/model are recreated:
for ckpt in ckpt_files:
    print("Evaluating:", ckpt)
    corp = Corp(car_mass=38.0, pilot_mass=55.0, CdA_coeff=0.020)
    roue = Roue(rayon=0.25, mass=1.6, coeff_rr=0.0013, coeff_linear=0.0, coeff_alpha=80.0, T_b=1.8e-3)
    # Use real motor datasheet values (example ELVM6040V48FH):
    # torque_constant Kt = 0.137 Nm/A, resistance = 0.28 Ω, peak torque ≈ 3.81 Nm, rated/peak rpm 3000/4000
    moteur = Moteur(gear_ratio=gear_ratio_from_teeth, inertia=7e-5, torque_constant=0.137, resistance=0.28,
                    engine_cutoff_rpm=3000, transmission_efficiency=0.95, max_motor_torque=3.81)
    voiture = Voiture(corp=corp, roue=roue, moteur=moteur, n_wheels=3, position_init=0.0, speed_init=0.0)
    env = VehicleEnv(voiture=voiture, track_length_m=3832.0, laps=4, dt=0.1)

    model = PPO.load(ckpt)
    rec = run_episode_and_record(model, env)
    results.append((ckpt, rec))
# factory pour créer un nouvel environnement propre à chaque vector
def make_env():
    """Factory retournant une fonction qui crée un nouvel environnement.

    Utile pour wrapper dans `DummyVecEnv` qui attend un callable retournant
    une instance d'environnement propre.
    """
    def _init():
        corp = Corp(car_mass=38.0, pilot_mass=55.0, CdA_coeff=0.020)
        roue = Roue(rayon=0.25, mass=1.6, coeff_rr=0.0013, coeff_linear=0.0, coeff_alpha=80.0, T_b=1.8e-3)
        # vector env: use same realistic motor params, max_motor_torque = peak torque
        moteur = Moteur(gear_ratio=gear_ratio_from_teeth, inertia=5.8e-5, torque_constant=0.137, resistance=0.28,
                        engine_cutoff_rpm=4000, transmission_efficiency=0.95, max_motor_torque=3.81)
        voiture = Voiture(corp=corp, roue=roue, moteur=moteur, n_wheels=3, position_init=0.0, speed_init=0.0)
        env = VehicleEnv(voiture=voiture, track_length_m=3832.0, laps=4, dt=0.1)
        tmpdir = tempfile.mkdtemp(prefix="veh_monitor_")
        env = Monitor(env, tmpdir, allow_early_resets=True)
        return env
    return _init

# Create vectorized env (single env but wrapped)
env_vec = DummyVecEnv([make_env()])

# Build PPO model
model = PPO("MlpPolicy", env_vec, verbose=1, tensorboard_log=None)

# helpers to accept gym / gymnasium (4- or 5-tuple) and vectorized outputs
def _unwrap_reset(res):
    """Normalise la sortie de `reset()` pour renvoyer uniquement les observations.

    Gym/Gymnasium peuvent retourner soit obs soit (obs, infos). Cette
    fonction renvoie toujours obs.
    """
    # env_vec.reset() may return obs or (obs, infos)
    if isinstance(res, tuple):
        # (obs, infos) or (obs, infos_array)
        return res[0]
    return res

def _unwrap_step(res):
    """Normalise la sortie de `step()` en une forme 5-tuple
    (obs, reward, terminated, truncated, info).

    Gym renvoie parfois 4-tuple (obs, reward, done, info) tandis que
    Gymnasium utilise 5-tuple (obs, reward, terminated, truncated, info).
    Cette fonction convertit les deux formats vers le second.
    """
    # step may return 4-tuple (gym): obs, reward, done, info
    # or 5-tuple (gymnasium): obs, reward, terminated, truncated, info
    if len(res) == 5:
        obs, reward, terminated, truncated, info = res
        return obs, reward, terminated, truncated, info
    elif len(res) == 4:
        obs, reward, done, info = res
        # convert done -> terminated, keep truncated False
        return obs, reward, done, False, info
    else:
        raise ValueError(f"Unexpected step output length: {len(res)}")

# Training loop across GENERATIONS
MAX_EVAL_STEPS = 20000  # safeguard for evaluation loops

for gen in range(1, GENERATIONS + 1):
    print(f"--- Generation {gen}/{GENERATIONS} : apprentissage {TIMESTEPS_PER_GEN} timesteps ---")
    model.learn(total_timesteps=TIMESTEPS_PER_GEN)

    # evaluation rapide (deterministic)
    rewards = []  # <-- initialise ici pour éviter NameError
    for ep in range(EVAL_EPISODES):
        reset_res = env_vec.reset()
        obs = _unwrap_reset(reset_res)
        ep_reward = 0.0
        step_count = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            step_res = env_vec.step(action)
            obs, reward, terminated, truncated, info = _unwrap_step(step_res)

            # reduce arrays -> boolean
            if isinstance(terminated, np.ndarray):
                terminated_flag = bool(terminated.any())
            else:
                terminated_flag = bool(terminated)
            if isinstance(truncated, np.ndarray):
                truncated_flag = bool(truncated.any())
            else:
                truncated_flag = bool(truncated)
            done_flag = terminated_flag or truncated_flag

            ep_reward += float(np.array(reward).sum())

            step_count += 1
            if done_flag or step_count >= MAX_EVAL_STEPS:
                if step_count >= MAX_EVAL_STEPS:
                    print(f"Eval ep {ep} reached MAX_EVAL_STEPS ({MAX_EVAL_STEPS}), forcing stop.")
                break

        rewards.append(ep_reward)

    mean_reward = float(np.mean(rewards)) if len(rewards) > 0 else float("nan")
    print(f"Generation {gen} -> mean_reward over {EVAL_EPISODES} eval episodes: {mean_reward:.3f}")
    # sauvegarde checkpoint léger
    ckpt_path = f"ppo_checkpoint_gen_{gen}.zip"
    model.save(ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

# Final save
final_path = "ppo_final_model.zip"
model.save(final_path)
print(f"Training finished. Final model saved to: {final_path}")

# Assertions simples pour le test
obs0 = _unwrap_reset(env_vec.reset())
# obs0 should be shape (n_envs, obs_dim)
assert obs0.ndim >= 2 and obs0.shape[1] == env_vec.observation_space.shape[0], "Observation shape mismatch"
assert np.isfinite(mean_reward), "mean_reward is not finite"

print("Sanity checks passed.")

# --- Charger et évaluer tous les checkpoints ---
ckpt_files = sorted(glob.glob("ppo_checkpoint_gen_*.zip"),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
if os.path.exists("ppo_final_model.zip"):
    ckpt_files.append("ppo_final_model.zip")

if len(ckpt_files) == 0:
    raise FileNotFoundError("Aucun checkpoint trouvé (ppo_checkpoint_gen_*.zip ou ppo_final_model.zip).")

results = []
for ckpt in ckpt_files:
    print("Evaluating:", ckpt)
    # recréer env identique (non vectorisé)
    corp = Corp(car_mass=38.0, pilot_mass=55.0, CdA_coeff=0.020)
    roue = Roue(rayon=0.25, mass=1.6, coeff_rr=0.0013, coeff_linear=0.0, coeff_alpha=80.0, T_b=1.8e-3)
    moteur = Moteur(gear_ratio=12.0, inertia=7e-5, torque_constant=0.08, resistance=0.12,
                    engine_cutoff_rpm=6500, transmission_efficiency=0.95, max_motor_torque=35.0)
    voiture = Voiture(corp=corp, roue=roue, moteur=moteur, n_wheels=3, position_init=0.0, speed_init=0.0)
    env = VehicleEnv(voiture=voiture, track_length_m=3832.0, laps=4, dt=0.1)

    model = PPO.load(ckpt)
    rec = run_episode_and_record(model, env)
    results.append((ckpt, rec))

# --- Calcul des métriques résumé ---
summaries = []
for ckpt, rec in results:
    energy_wh = rec.get("total_energy", float("nan"))
    dist_km = rec.get("total_distance_m", float("nan")) / 1000.0
    km_per_kwh = np.nan
    if not np.isnan(energy_wh) and energy_wh > 0:
        km_per_kwh = dist_km / (energy_wh / 1000.0)
    summaries.append({
        "ckpt": os.path.basename(ckpt),
        "score": rec.get("total_reward", float("nan")),
        "energy_wh": energy_wh,
        "dist_km": dist_km,
        "km_per_kwh": km_per_kwh,
        "rec": rec
    })

# --- Affichage synthétique (console) ---
print("{:30s} {:8s} {:12s} {:8s} {:8s}".format("checkpoint","score","energy(Wh)","dist(km)","km/kWh"))
for s in summaries:
    print("{:30s} {:8.1f} {:12.2f} {:8.3f} {:8.2f}".format(
        s["ckpt"], s["score"], s["energy_wh"] if not np.isnan(s["energy_wh"]) else 0.0,
        s["dist_km"] if not np.isnan(s["dist_km"]) else 0.0,
        s["km_per_kwh"] if not np.isnan(s["km_per_kwh"]) else 0.0
    ))

# --- Plots individuels speed + energy avec annotation score & km/kWh ---
n = len(summaries)
cols = 2
rows = int(np.ceil(n / cols)) if n>0 else 1
fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows), squeeze=False)

for i, s in enumerate(summaries):
    r, c = divmod(i, cols)
    ax = axes[r][c]
    rec = s["rec"]
    times = rec["times"]
    speeds = rec["speeds"]
    energies = rec["energies"]
    positions = rec["positions"]

    # Affiche la vitesse en fonction de la distance (positions en m) si disponible
    if positions.size > 0 and speeds.size == positions.size:
        ax.plot(positions, speeds, label="v (m/s)", color="C0")
        x_for_energy = positions
    else:
        # fallback : trace la vitesse seule (index) si pas de positions cohérentes
        ax.plot(speeds, label="v (m/s)", color="C0")
        x_for_energy = None

    title = f"{s['ckpt']}\nscore={s['score']:.1f}  E={s['energy_wh']:.2f}Wh  km/kWh={s['km_per_kwh']:.2f}"
    ax.set_title(title)
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("speed (m/s)")
    ax.grid(True)

    # energy on twin axis, alignée sur la même abscisse (distance) si possible
    if energies.size > 0:
        ax2 = ax.twinx()
        if x_for_energy is not None and x_for_energy.size == energies.size:
            ax2.plot(x_for_energy, energies, color="orange", alpha=0.9, label="E cum (Wh)")
        else:
            ax2.plot(energies, color="orange", alpha=0.9, label="E cum (Wh)")
        ax2.set_ylabel("energy (Wh)", color="orange")
        ax2.tick_params(axis='y', labelcolor="orange")

# hide empty axes
for j in range(n, rows*cols):
    r, c = divmod(j, cols)
    axes[r][c].axis("off")

fig.tight_layout()

# Remplace la section de plot global par une figure par checkpoint :
import matplotlib.pyplot as plt
import numpy as np

# crée une figure distincte par checkpoint avec : vitesse, énergie cumulée, position/distance et temps de finish
for s in summaries:
    ckpt = s["ckpt"]
    rec = s["rec"]

    times = np.asarray(rec.get("times", np.array([])), dtype=float)
    positions = np.asarray(rec.get("positions", np.array([])), dtype=float)
    speeds = np.asarray(rec.get("speeds", np.array([])), dtype=float)
    energies = np.asarray(rec.get("energies", np.array([])), dtype=float)

    finish_time = float(times[-1]) if times.size > 0 else np.nan
    total_energy = float(s.get("energy_wh", np.nan))
    kmkwh = s.get("km_per_kwh", np.nan)

    # Choix de l'abscisse : préférence position -> temps -> pas d'indice
    if positions.size > 0 and positions.size == speeds.size:
        x = positions
        xlabel = "distance (m)"
    elif times.size > 0 and times.size == speeds.size:
        x = times
        xlabel = "time (s)"
    else:
        x = np.arange(speeds.size)
        xlabel = "step"

    fig, ax1 = plt.subplots(figsize=(10,4))

    # vitesse (axe gauche)
    ax1.plot(x, speeds, color="C0", label="v (m/s)")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("speed (m/s)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.3)

    # énergie cumulée (axe droit)
    ax2 = ax1.twinx()
    # si energies alignées sur la même abscisse, tracer en fonction de x, sinon tracer l'array seul
    if energies.size > 0 and energies.size == x.size:
        ax2.plot(x, energies, color="orange", label="E cum (Wh)", linewidth=1.5, alpha=0.9)
    else:
        ax2.plot(energies, color="orange", label="E cum (Wh)", linewidth=1.5, alpha=0.9)
    ax2.set_ylabel("energy (Wh)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # position en trait pointillé sur un 3e axe si disponible
    if positions.size > 0:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))
        ax3.plot(x if (positions.size==x.size) else positions, positions, color="C2", linestyle="--", label="position (m)")
        ax3.set_ylabel("position (m)", color="C2")
        ax3.tick_params(axis="y", labelcolor="C2")

    # légende combinée
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, loc="upper left", fontsize="small")

    # annotation finish time / énergie / km/kWh
    kmkwh_str = f"{kmkwh:.2f}" if (not np.isnan(kmkwh)) else "N/A"
    title = f"{ckpt} — finish_time={finish_time:.1f}s — E={total_energy:.2f}Wh — km/kWh={kmkwh_str}"
    ax1.set_title(title)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Sauvegarde d'un rec (ex: rec obtenu) — adapte le chemin
import json
import numpy as np
import pandas as pd

rec = rec  # objet retourné par run_episode_and_record

# npz
np.savez("eval_rec.npz",
         times=rec["times"], speeds=rec["speeds"], positions=rec["positions"],
         energies=rec["energies"], rewards=rec["rewards"])

# csv time series
df = pd.DataFrame({
    "time_s": rec["times"],
    "speed_mps": rec["speeds"],
    "position_m": rec["positions"],
    "energy_Wh": rec["energies"],
    "reward": rec["rewards"]
})
df.to_csv("eval_rec_timeseries.csv", index=False)

# json summary
summary = {
    "total_reward": float(rec["total_reward"]),
    "total_energy_Wh": float(rec["total_energy"]),
    "total_distance_m": float(rec["total_distance_m"]),
    "steps": int(len(rec["times"]))
}
with open("eval_rec_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
