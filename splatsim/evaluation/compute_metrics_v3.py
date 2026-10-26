#!/usr/bin/env python3
"""Métricas de evaluación offline para episodios SplatSim/GELLO (PKL).

Incluye:
  - Métricas de éxito por episodio (error de posición/orientación, distancia
    mínima, longitud de trayectoria, tiempo al objetivo, tasa de éxito).
  - Análisis de sesgo direccional del dataset (¿el movimiento se va hacia un lado?).
  - Análisis de jitter y suavidad de trayectoria (jerk normalizado, SPARC, etc.).
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # backend no interactivo, seguro para ejecución headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- #
# Estructuras de datos
# --------------------------------------------------------------------------- #

@dataclass
class EpisodeData:
    """Datos crudos extraídos de un episodio, ya alineados frame a frame."""

    episode_id: str
    num_frames: int
    ee_positions: np.ndarray  # (T, 3) — posición cruda de la MUÑECA
    gripper_positions: np.ndarray  # (T, 3) — posición del centro de las pinzas (con offset aplicado)
    ee_orientations: Optional[np.ndarray]  # (T, 4) quat xyzw, o None
    ee_orientation_source: Optional[str]  # 'ee_pos_quat' | 'euler_state' | None
    fruit_position: Optional[np.ndarray]  # (3,) — se asume estática en el episodio
    fruit_orientation: Optional[np.ndarray]  # (4,) quat xyzw, o None
    dt: Optional[float] = None  # paso de tiempo simulado entre frames ('sim_time'), o None
    warnings_list: list = field(default_factory=list)


@dataclass
class EpisodeMetrics:
    episode_id: str
    num_frames: int
    final_position_error: float
    min_distance: float
    path_length: float
    time_to_target: float  # en frames (índice); NaN si nunca se alcanza
    orientation_error_deg: float  # NaN si no se pudo calcular
    success: bool
    distance_curve: np.ndarray  # (T,) para graficar


@dataclass
class TrajectoryQuality:
    """Métricas de dirección, jitter y suavidad de una trayectoria."""

    episode_id: str
    displacement: np.ndarray  # (3,) desplazamiento neto final - inicial (m)
    signed_path_share: np.ndarray  # (3,) en [-1, 1]: 1 = movimiento monótono en +eje
    fruit_rel_start: Optional[np.ndarray]  # (3,) fruto - posición inicial (m)
    mean_speed: float  # m/s
    rms_acc: float  # m/s^2
    rms_jerk: float  # m/s^3
    normalized_jerk: float  # adimensional (menor = más suave)
    sparc: float  # negativo; más cercano a 0 = más suave
    jitter_rms: float  # m, residuo vs. trayectoria suavizada
    path_efficiency: float  # (0, 1], 1 = línea recta
    reversals_per_s: float  # cambios de sentido de velocidad por segundo


# --------------------------------------------------------------------------- #
# Carga de episodios
# --------------------------------------------------------------------------- #

def load_episode(episode_dir: Path) -> list[dict]:
    """Carga todos los frames .pkl de un episodio, ordenados por nombre de archivo.

    Parameters
    ----------
    episode_dir : Path
        Carpeta del episodio (ej. .../gello/000/) conteniendo archivos
        00000.pkl, 00001.pkl, ...

    Returns
    -------
    list[dict]
        Lista de diccionarios deserializados, uno por frame, en orden temporal.

    Notas
    -----
    No se modifica ni reinterpreta el contenido aquí; solo se deserializa.
    Si un archivo no puede deserializarse, se omite con una advertencia
    (no se aborta el episodio completo por un frame corrupto).
    """
    pkl_files = sorted(episode_dir.glob("*.pkl"))
    if not pkl_files:
        warnings.warn(f"[{episode_dir.name}] No se encontraron archivos .pkl en {episode_dir}")
        return []

    frames = []
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                frame = pickle.load(f)
            frames.append(frame)
        except Exception as exc:  # noqa: BLE001 — queremos capturar cualquier error de IO/unpickling
            warnings.warn(f"[{episode_dir.name}] No se pudo leer '{pkl_path.name}': {exc}")
    return frames


# --------------------------------------------------------------------------- #
# Extracción de campos (con validación explícita, sin suposiciones silenciosas)
# --------------------------------------------------------------------------- #

def _to_array(value) -> Optional[np.ndarray]:
    """Convierte tuplas/listas/arrays a np.ndarray float; devuelve None si falla."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
        return arr
    except (TypeError, ValueError):
        return None


def extract_ee_pose(
    frame: dict, episode_id: str, frame_idx: int, euler_seq: str = "xyz"
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Extrae posición y orientación del efector final de un frame.

    Estrategia (documentada en el docstring del módulo):
      - Posición: primeros 3 elementos de 'state', o de 'action' si 'state'
        no existe.
      - Orientación: cuaternión 'ee_pos_quat' (x,y,z,w) si existe;
        si no, fallback a Euler 'state'/'action'[3:6] con la secuencia
        indicada en euler_seq, convertido a cuaternión (x,y,z,w).

    Returns
    -------
    (position, orientation_quat_xyzw, source)
        Cualquiera de los dos primeros puede ser None si no se pudo extraer.
        `source` indica de dónde vino la orientación: 'ee_pos_quat',
        'euler_state', 'euler_action', o None.
    """
    position = None
    orientation = None
    source = None

    pose_field = None
    pose_field_name = None
    if "state" in frame:
        pose_field = _to_array(frame["state"])
        pose_field_name = "state"
    elif "action" in frame:
        pose_field = _to_array(frame["action"])
        pose_field_name = "action"

    if pose_field is None or pose_field.shape[0] < 3:
        warnings.warn(
            f"[{episode_id}][frame {frame_idx}] No se encontró 'state' ni 'action' "
            f"válidos para extraer la posición del efector."
        )
    else:
        position = pose_field[:3]

    # Orientación: prioridad a ee_pos_quat (cuaternión directo, sin ambigüedad de convención)
    if "ee_pos_quat" in frame:
        quat = _to_array(frame["ee_pos_quat"])
        if quat is not None and quat.shape[0] == 4:
            orientation = quat
            source = "ee_pos_quat"
        else:
            warnings.warn(
                f"[{episode_id}][frame {frame_idx}] 'ee_pos_quat' presente pero con "
                f"forma inesperada ({None if quat is None else quat.shape}); se ignora."
            )

    if orientation is None and pose_field is not None and pose_field.shape[0] >= 6:
        euler_angles = pose_field[3:6]
        try:
            rot = Rotation.from_euler(euler_seq, euler_angles, degrees=False)
            orientation = rot.as_quat()  # devuelve (x, y, z, w) por convención scipy
            source = f"euler_{pose_field_name}"
        except ValueError as exc:
            warnings.warn(
                f"[{episode_id}][frame {frame_idx}] No se pudo convertir Euler a "
                f"cuaternión desde '{pose_field_name}[3:6]': {exc}"
            )

    if orientation is None:
        warnings.warn(
            f"[{episode_id}][frame {frame_idx}] No se pudo determinar la orientación "
            f"del efector (ni 'ee_pos_quat' ni Euler en 'state'/'action' disponibles)."
        )

    return position, orientation, source


def extract_fruit_pose(
    frame: dict, episode_id: str, frame_idx: int
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extrae posición y orientación del fruto ('plastic_apple') de un frame.

    Returns
    -------
    (position, orientation_quat_xyzw)
        position: array (3,) o None.
        orientation: array (4,) en convención (x,y,z,w) o None.
    """
    position = None
    orientation = None

    if "plastic_apple_position" in frame:
        pos = _to_array(frame["plastic_apple_position"])
        if pos is not None and pos.shape[0] == 3:
            position = pos
        else:
            warnings.warn(
                f"[{episode_id}][frame {frame_idx}] 'plastic_apple_position' con forma "
                f"inesperada ({None if pos is None else pos.shape})."
            )
    else:
        warnings.warn(f"[{episode_id}][frame {frame_idx}] Falta el campo 'plastic_apple_position'.")

    if "plastic_apple_orientation" in frame:
        ori = _to_array(frame["plastic_apple_orientation"])
        if ori is not None and ori.shape[0] == 4:
            orientation = ori
        else:
            warnings.warn(
                f"[{episode_id}][frame {frame_idx}] 'plastic_apple_orientation' con forma "
                f"inesperada ({None if ori is None else ori.shape})."
            )
    # No emitimos warning si falta la orientación del fruto: es opcional (métrica 7
    # se omite por completo si no está disponible, sin penalizar el resto del script).

    return position, orientation


def compute_gripper_position(
    wrist_positions: np.ndarray,
    orientations: Optional[np.ndarray],
    offset: float,
    episode_id: str,
) -> np.ndarray:
    """Calcula la posición del centro de las pinzas del gripper a partir de la
    posición de la muñeca, desplazando por 'offset' metros en el eje Z LOCAL
    del efector, rotado por la orientación real de cada frame.

    Esto es necesario porque 'state'/'action' reportan la pose de la muñeca,
    no del punto de contacto entre las pinzas, y existe un desplazamiento
    físico (~0.18 m por defecto, confirmado por el usuario) entre ambos
    puntos. Al rotar el offset local con la orientación de cada frame, el
    cálculo es válido sin importar cómo esté orientada la muñeca en ese
    instante (p. ej. si se aproxima desde arriba en vez de horizontalmente).

    Parameters
    ----------
    wrist_positions : (T, 3)
        Posición cruda de la muñeca por frame.
    orientations : (T, 4) o None
        Cuaternión (x, y, z, w) del efector por frame. Si es None, no hay
        forma confiable de rotar el offset y se usa la posición cruda de
        la muñeca como fallback (con advertencia).
    offset : float
        Magnitud del desplazamiento en metros, aplicado en +Z local.
    episode_id : str
        Solo para mensajes de advertencia.

    Returns
    -------
    np.ndarray (T, 3) — posición estimada del centro de las pinzas.
    """
    if orientations is None:
        warnings.warn(
            f"[{episode_id}] No hay orientación del efector disponible; no se puede "
            f"aplicar el offset de gripper de forma confiable. Se usa la posición "
            f"cruda de la muñeca (gripper_pos == ee_pos) para este episodio."
        )
        return wrist_positions.copy()

    offset_local = np.array([0.0, 0.0, offset])
    rotations = Rotation.from_quat(orientations)  # (T,) batched rotation
    offset_global = rotations.apply(offset_local)  # (T, 3)
    return wrist_positions + offset_global


def build_episode_data(
    episode_dir: Path, euler_seq: str = "xyz", gripper_offset: float = 0.18
) -> Optional[EpisodeData]:
    """Carga un episodio completo y extrae las series de poses EE/fruto.

    El fruto se asume estático durante el episodio (es una posición fija en
    la escena); se usa la mediana de las posiciones detectadas frame a frame
    como valor robusto (en caso de ruido de sensado), advirtiendo si hay alta
    varianza entre frames (lo que indicaría que el supuesto de estaticidad
    no se cumple para ese episodio).
    """
    episode_id = episode_dir.name
    frames = load_episode(episode_dir)
    if not frames:
        return None

    ee_positions = []
    ee_orientations = []
    ee_orientation_sources = []
    fruit_positions = []
    fruit_orientations = []

    for idx, frame in enumerate(frames):
        ee_pos, ee_ori, ori_source = extract_ee_pose(frame, episode_id, idx, euler_seq=euler_seq)
        fruit_pos, fruit_ori = extract_fruit_pose(frame, episode_id, idx)

        ee_positions.append(ee_pos)
        ee_orientations.append(ee_ori)
        ee_orientation_sources.append(ori_source)
        fruit_positions.append(fruit_pos)
        fruit_orientations.append(fruit_ori)

    # Filtrar frames sin posición de EE válida (no se pueden imputar sin inventar datos)
    valid_idx = [i for i, p in enumerate(ee_positions) if p is not None]
    if len(valid_idx) < len(frames):
        warnings.warn(
            f"[{episode_id}] {len(frames) - len(valid_idx)} de {len(frames)} frames "
            f"no tenían posición de efector válida y fueron excluidos de las métricas "
            f"geométricas de este episodio."
        )
    if not valid_idx:
        warnings.warn(f"[{episode_id}] Ningún frame tiene posición de efector válida; episodio omitido.")
        return None

    ee_positions_arr = np.stack([ee_positions[i] for i in valid_idx])  # (T, 3)

    # Orientación del EE: solo se mantiene si TODOS los frames válidos la tienen
    valid_ori = [ee_orientations[i] for i in valid_idx]
    if all(o is not None for o in valid_ori):
        ee_orientations_arr = np.stack(valid_ori)  # (T, 4)
        sources_used = {ee_orientation_sources[i] for i in valid_idx}
        ori_source_final = sources_used.pop() if len(sources_used) == 1 else "mixed"
        if ori_source_final == "mixed":
            warnings.warn(
                f"[{episode_id}] La fuente de orientación del efector varió entre frames "
                f"dentro del mismo episodio (mezcla de {sources_used}); revisar consistencia "
                f"de formato del PKL."
            )
    else:
        ee_orientations_arr = None
        ori_source_final = None
        warnings.warn(
            f"[{episode_id}] Orientación del efector incompleta en algunos frames; "
            f"se omite el error de orientación para este episodio."
        )

    # Posición del fruto: tomar mediana de los frames válidos (robusta a outliers)
    valid_fruit_pos = [p for p in fruit_positions if p is not None]
    if not valid_fruit_pos:
        warnings.warn(f"[{episode_id}] No se encontró 'plastic_apple_position' en ningún frame.")
        fruit_position_final = None
    else:
        fruit_stack = np.stack(valid_fruit_pos)
        fruit_position_final = np.median(fruit_stack, axis=0)
        spread = np.std(fruit_stack, axis=0)
        if np.any(spread > 1e-3):
            warnings.warn(
                f"[{episode_id}] La posición de 'plastic_apple_position' varía entre frames "
                f"(std={spread}); se usa la mediana, pero verificar si el fruto se mueve "
                f"o si hay ruido de sensado."
            )

    valid_fruit_ori = [o for o in fruit_orientations if o is not None]
    fruit_orientation_final = np.median(np.stack(valid_fruit_ori), axis=0) if valid_fruit_ori else None

    gripper_positions_arr = compute_gripper_position(
        ee_positions_arr, ee_orientations_arr, gripper_offset, episode_id
    )

    # paso de tiempo real de la grabación, si los frames traen 'sim_time'
    sim_times = [f.get("sim_time") for f in frames]
    dt = None
    if len(sim_times) > 1 and None not in sim_times:
        dt = float(np.median(np.diff(sim_times)))

    return EpisodeData(
        dt=dt,
        episode_id=episode_id,
        num_frames=len(valid_idx),
        ee_positions=ee_positions_arr,
        gripper_positions=gripper_positions_arr,
        ee_orientations=ee_orientations_arr,
        ee_orientation_source=ori_source_final,
        fruit_position=fruit_position_final,
        fruit_orientation=fruit_orientation_final,
    )


# --------------------------------------------------------------------------- #
# Cómputo de métricas
# --------------------------------------------------------------------------- #

def compute_position_error(gripper_positions: np.ndarray, fruit_position: np.ndarray) -> np.ndarray:
    """Distancia euclidiana entre cada posición del GRIPPER (ya con offset aplicado)
    y la posición (fija) del fruto.

    Parameters
    ----------
    gripper_positions : (T, 3)
        Posición del centro de las pinzas, ya corregida por el offset
        muñeca→gripper (ver compute_gripper_position).
    fruit_position : (3,)

    Returns
    -------
    np.ndarray (T,) — distancia en cada frame.
    """
    return np.linalg.norm(gripper_positions - fruit_position[None, :], axis=1)


def compute_orientation_error(
    ee_orientation_final: np.ndarray, fruit_orientation: np.ndarray
) -> float:
    """Error angular (en grados) entre el cuaternión final del EE y el del fruto.

    Ambos cuaterniones deben estar en convención (x, y, z, w).

    Se calcula como el ángulo de la rotación relativa R_rel = R_fruit^-1 * R_ee,
    usando scipy.spatial.transform.Rotation, que es la forma estándar y
    numéricamente estable de obtener la distancia angular entre orientaciones
    (evita los problemas de doble cobertura de comparar componentes de
    cuaternión directamente).
    """
    r_ee = Rotation.from_quat(ee_orientation_final)
    r_fruit = Rotation.from_quat(fruit_orientation)
    r_rel = r_fruit.inv() * r_ee
    angle_deg = r_rel.magnitude() * (180.0 / np.pi)
    return float(angle_deg)


def compute_path_length(gripper_positions: np.ndarray) -> float:
    """Longitud total de la trayectoria del GRIPPER (con offset aplicado):
    suma de distancias entre frames consecutivos."""
    if gripper_positions.shape[0] < 2:
        return 0.0
    diffs = np.diff(gripper_positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_success(distance_curve: np.ndarray, position_threshold: float, sustain_frames=10):
    # sustain_frames = 10 equals 0.5 seconds at 20 fps
    success = False
    time_to_target = np.nan

    count = 0
    for i, d in enumerate(distance_curve):
        if d < position_threshold:
            count += 1
            if count == sustain_frames:
                success = True
                # First frame of the sustained interval
                time_to_target = float(i - sustain_frames + 1)
                break
        else:
            count = 0

    return success, time_to_target


def compute_episode_metrics(
    episode: EpisodeData, position_threshold: float
) -> EpisodeMetrics:
    """Calcula todas las métricas para un episodio ya cargado y parseado."""
    if episode.fruit_position is None:
        # Sin posición de fruto no se puede calcular NINGUNA métrica geométrica
        # relativa al objetivo. Se documenta con NaN explícito, no con cero.
        warnings.warn(
            f"[{episode.episode_id}] Sin posición de fruto disponible; todas las "
            f"métricas de distancia/éxito se reportan como NaN."
        )
        distance_curve = np.full(episode.num_frames, np.nan)
        return EpisodeMetrics(
            episode_id=episode.episode_id,
            num_frames=episode.num_frames,
            final_position_error=np.nan,
            min_distance=np.nan,
            path_length=compute_path_length(episode.gripper_positions),
            time_to_target=np.nan,
            orientation_error_deg=np.nan,
            success=False,
            distance_curve=distance_curve,
        )

    distance_curve = compute_position_error(episode.gripper_positions, episode.fruit_position)
    final_position_error = float(distance_curve[-1])
    min_distance = float(np.min(distance_curve))
    path_length = compute_path_length(episode.gripper_positions)

    below_threshold = np.where(distance_curve < position_threshold)[0]
    time_to_target = float(below_threshold[0]) if below_threshold.size > 0 else np.nan

    success = bool(final_position_error < position_threshold)

    success, time_to_target = compute_success(distance_curve, position_threshold)

    orientation_error_deg = np.nan
    if episode.ee_orientations is not None and episode.fruit_orientation is not None:
        try:
            ee_ori_final = episode.ee_orientations[-1]
            orientation_error_deg = compute_orientation_error(ee_ori_final, episode.fruit_orientation)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[{episode.episode_id}] Error al calcular orientation_error_deg: {exc}. "
                f"Se reporta NaN para esta métrica."
            )
    else:
        warnings.warn(
            f"[{episode.episode_id}] No hay datos suficientes de orientación (EE y/o fruto); "
            f"se omite orientation_error_deg (NaN)."
        )

    return EpisodeMetrics(
        episode_id=episode.episode_id,
        num_frames=episode.num_frames,
        final_position_error=final_position_error,
        min_distance=min_distance,
        path_length=path_length,
        time_to_target=time_to_target,
        orientation_error_deg=orientation_error_deg,
        success=success,
        distance_curve=distance_curve,
    )


# --------------------------------------------------------------------------- #
# Calidad de trayectoria (jitter / suavidad) y sesgo direccional del dataset
# --------------------------------------------------------------------------- #

def compute_sparc(
    speed: np.ndarray, fps: float, pad_level: int = 4, fc: float = 10.0, amp_th: float = 0.05
) -> float:
    """Spectral Arc Length (Balasubramanian et al., 2015) del perfil de rapidez.

    Valores más negativos = movimiento menos suave; cercano a 0 = muy suave.
    """
    if len(speed) < 3 or np.max(speed) <= 1e-9:
        return np.nan
    fc = min(fc, fps / 2.0)
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + pad_level))
    f = np.arange(nfft) / nfft * fps
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    sel = f <= fc
    f_sel, mag_sel = f[sel], mag[sel]
    idx = np.where(mag_sel >= amp_th)[0]
    if idx.size < 2:
        return 0.0
    f_sel = f_sel[idx[0]: idx[-1] + 1]
    mag_sel = mag_sel[idx[0]: idx[-1] + 1]
    span = f_sel[-1] - f_sel[0]
    if span <= 0:
        return 0.0
    return float(-np.sum(np.sqrt((np.diff(f_sel) / span) ** 2 + np.diff(mag_sel) ** 2)))


def compute_trajectory_quality(
    positions: np.ndarray,
    fruit_position: Optional[np.ndarray],
    episode_id: str,
    fps: float = 20.0,
    vel_deadband: float = 0.005,
) -> Optional[TrajectoryQuality]:
    """Métricas de jitter, suavidad y dirección para una trayectoria (T, 3) en metros.

    Parameters
    ----------
    positions : (T, 3)
        Trayectoria (gripper o muñeca) en metros.
    fruit_position : (3,) o None
        Posición del fruto, para calcular su posición relativa al punto de inicio.
    fps : float
        Frecuencia de muestreo, para convertir diferencias a velocidad/aceleración/jerk.
    vel_deadband : float
        Velocidad (m/s) por debajo de la cual un eje se considera en reposo al
        contar reversiones de sentido.
    """
    T = positions.shape[0]
    if T < 5:
        warnings.warn(f"[{episode_id}] Menos de 5 frames; se omite el análisis de calidad.")
        return None

    dt = 1.0 / fps
    duration = (T - 1) * dt

    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    path_len = compute_path_length(positions)
    straight = float(np.linalg.norm(positions[-1] - positions[0]))
    path_eff = straight / path_len if path_len > 1e-9 else np.nan

    # Jerk normalizado adimensional: sqrt(0.5 * ∫|jerk|² dt * T^5 / L²)
    jerk_int = float(np.sum(np.sum(jerk ** 2, axis=1)) * dt)
    nj = float(np.sqrt(0.5 * jerk_int * duration ** 5 / path_len ** 2)) if path_len > 1e-9 else np.nan

    # Jitter: residuo respecto a una versión suavizada de la misma trayectoria
    win = min(9, T)
    if win % 2 == 0:
        win -= 1
    smooth = savgol_filter(positions, window_length=win, polyorder=min(3, win - 1), axis=0)
    jitter = float(np.sqrt(np.mean(np.sum((positions - smooth) ** 2, axis=1))))

    # Reversiones de sentido por eje (con zona muerta para ignorar reposo)
    sgn = np.where(np.abs(vel) < vel_deadband, 0, np.sign(vel))
    reversals = 0
    for ax in range(3):
        s = sgn[:, ax]
        s = s[s != 0]
        if s.size > 1:
            reversals += int(np.sum(s[1:] != s[:-1]))

    # Consistencia direccional por eje: (recorrido neto) / (recorrido total) en [-1, 1]
    diffs = np.diff(positions, axis=0)
    total_abs = np.abs(diffs).sum(axis=0)
    share = np.divide(diffs.sum(axis=0), total_abs, out=np.zeros(3), where=total_abs > 1e-12)

    fruit_rel = None if fruit_position is None else (fruit_position - positions[0])

    return TrajectoryQuality(
        episode_id=episode_id,
        displacement=positions[-1] - positions[0],
        signed_path_share=share,
        fruit_rel_start=fruit_rel,
        mean_speed=float(np.mean(speed)),
        rms_acc=float(np.sqrt(np.mean(np.sum(acc ** 2, axis=1)))),
        rms_jerk=float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))),
        normalized_jerk=nj,
        sparc=compute_sparc(speed, fps),
        jitter_rms=jitter,
        path_efficiency=path_eff,
        reversals_per_s=reversals / duration,
    )


def _rayleigh_test(angles: np.ndarray) -> tuple[float, float]:
    """Test de Rayleigh de uniformidad circular. Devuelve (R_bar, p-value).

    R_bar cercano a 1 = direcciones concentradas; p bajo = distribución no uniforme.
    """
    n = len(angles)
    if n < 2:
        return np.nan, np.nan
    C, S = np.mean(np.cos(angles)), np.mean(np.sin(angles))
    R_bar = float(np.hypot(C, S))
    R = n * R_bar
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n)))
    return R_bar, min(p, 1.0)


def analyze_dataset_bias(qualities: list[TrajectoryQuality], left_axis: str = "+y") -> dict:
    """Evalúa si el dataset está sesgado hacia alguna dirección.

    Parameters
    ----------
    qualities : list[TrajectoryQuality]
    left_axis : str
        Eje (en el frame del robot) que corresponde a 'izquierda', ej. '+y' o '-x'.
        Ajustar según el frame de referencia del robot.
    """
    disp = np.stack([q.displacement for q in qualities])  # (N, 3)
    n = len(disp)
    result: dict = {"n": n, "axes": {}}

    for i, name in enumerate("xyz"):
        d = disp[:, i]
        t_p = stats.ttest_1samp(d, 0.0).pvalue if n >= 2 else np.nan
        n_pos, n_neg = int(np.sum(d > 0)), int(np.sum(d < 0))
        sign_p = stats.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if (n_pos + n_neg) > 0 else np.nan
        result["axes"][name] = {
            "mean_cm": float(np.mean(d) * 100),
            "std_cm": float(np.std(d) * 100),
            "frac_positive": n_pos / n,
            "p_ttest": float(t_p),
            "p_sign": float(sign_p),
            "mean_path_share": float(np.mean([q.signed_path_share[i] for q in qualities])),
        }

    azimuth = np.arctan2(disp[:, 1], disp[:, 0])
    r_bar, p_ray = _rayleigh_test(azimuth)
    result["azimuth"] = {
        "angles": azimuth,
        "R_bar": r_bar,
        "p_rayleigh": p_ray,
        "mean_deg": float(np.degrees(np.arctan2(np.mean(np.sin(azimuth)), np.mean(np.cos(azimuth))))),
    }

    sign = -1.0 if left_axis.startswith("-") else 1.0
    ax_idx = "xyz".index(left_axis.lstrip("+-"))
    result["left_frac"] = float(np.mean(sign * disp[:, ax_idx] > 0))
    result["left_axis"] = left_axis

    fruit = [q.fruit_rel_start for q in qualities if q.fruit_rel_start is not None]
    result["fruit_rel_mean_cm"] = (np.mean(np.stack(fruit), axis=0) * 100) if fruit else None
    return result


# --------------------------------------------------------------------------- #
# Visualización
# --------------------------------------------------------------------------- #

def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def generate_plots(all_metrics: list[EpisodeMetrics], output_dir: Path, position_threshold: float) -> None:
    """Genera todos los gráficos requeridos: curvas por episodio + histogramas/boxplots agregados."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Curvas de distancia por episodio (convertidas a cm) ---
    threshold_cm = position_threshold * 100.0
    for m in all_metrics:
        if np.all(np.isnan(m.distance_curve)):
            continue  # sin datos de fruto, no hay curva que graficar
        distance_curve_cm = m.distance_curve * 100.0
        fig, ax = plt.subplots(figsize=(7, 4.5))
        t = np.arange(len(distance_curve_cm))
        ax.plot(t, distance_curve_cm, color="#1f5fa8", linewidth=2, label="distancia al fruto")
        ax.axhline(
            threshold_cm,
            color="#c0392b",
            linestyle="--",
            linewidth=1.5,
            label=f"umbral éxito ({threshold_cm:.1f} cm)",
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distancia gripper–fruto (cm)")
        ax.set_title(f"Episodio {m.episode_id} — evolución de la distancia")
        ax.legend(loc="upper right", frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(output_dir / f"episode_{m.episode_id}_distance_curve.png", dpi=150)
        plt.close(fig)

    # --- Histograma de error final de posición (cm) ---
    final_errors_cm = np.array([m.final_position_error for m in all_metrics]) * 100.0
    final_errors_valid = final_errors_cm[~np.isnan(final_errors_cm)]
    if final_errors_valid.size > 0:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(final_errors_valid, bins=min(15, max(5, final_errors_valid.size)), color="#1f5fa8", edgecolor="white")
        ax.axvline(threshold_cm, color="#c0392b", linestyle="--", label=f"umbral ({threshold_cm:.1f} cm)")
        ax.set_xlabel("Error final de posición (cm)")
        ax.set_ylabel("N° de episodios")
        ax.set_title("Distribución del error final de posición")
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(output_dir / "final_position_error_histogram.png", dpi=150)
        plt.close(fig)

        # --- Boxplot de error final de posición (cm) ---
        fig, ax = plt.subplots(figsize=(4.5, 5))
        bp = ax.boxplot(final_errors_valid, patch_artist=True, widths=0.5)
        for box in bp["boxes"]:
            box.set_facecolor("#9fc5e8")
        ax.set_ylabel("Error final de posición (cm)")
        ax.set_title("Error final de posición (boxplot)")
        ax.set_xticks([])
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(output_dir / "final_position_error_boxplot.png", dpi=150)
        plt.close(fig)
    else:
        warnings.warn("No hay errores de posición válidos; se omiten los gráficos agregados de posición.")

    # --- Histograma de error de orientación ---
    ori_errors = np.array([m.orientation_error_deg for m in all_metrics])
    ori_errors_valid = ori_errors[~np.isnan(ori_errors)]
    if ori_errors_valid.size > 0:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(ori_errors_valid, bins=min(15, max(5, ori_errors_valid.size)), color="#2e8b57", edgecolor="white")
        ax.set_xlabel("Error de orientación (grados)")
        ax.set_ylabel("N° de episodios")
        ax.set_title("Distribución del error de orientación final")
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(output_dir / "orientation_error_histogram.png", dpi=150)
        plt.close(fig)
    else:
        warnings.warn(
            "No hay datos válidos de error de orientación en ningún episodio; "
            "se omite 'orientation_error_histogram.png'."
        )


def generate_quality_plots(qualities: list[TrajectoryQuality], bias: dict, output_dir: Path) -> None:
    """Gráficos de sesgo direccional y de suavidad/jitter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    disp_cm = np.stack([q.displacement for q in qualities]) * 100.0

    # --- Polar: dirección del desplazamiento neto en el plano XY ---
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.hist(bias["azimuth"]["angles"], bins=16, color="#1f5fa8", edgecolor="white")
    ax.set_title("Dirección del desplazamiento neto (plano XY)")
    fig.tight_layout()
    fig.savefig(output_dir / "bias_azimuth_polar.png", dpi=150)
    plt.close(fig)

    # --- Desplazamiento medio por eje con IC 95% ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    means = disp_cm.mean(axis=0)
    sems = disp_cm.std(axis=0, ddof=1) / np.sqrt(len(disp_cm)) if len(disp_cm) > 1 else np.zeros(3)
    ax.bar(["x", "y", "z"], means, yerr=1.96 * sems, color="#9fc5e8", capsize=5)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("Desplazamiento neto medio (cm), IC 95%")
    ax.set_title("Sesgo por eje")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "bias_mean_displacement.png", dpi=150)
    plt.close(fig)

    # --- Dispersión XY: desplazamientos y posición del fruto relativa al inicio ---
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(disp_cm[:, 0], disp_cm[:, 1], c="#1f5fa8", label="desplazamiento neto", alpha=0.8)
    fruit_rel = [q.fruit_rel_start for q in qualities if q.fruit_rel_start is not None]
    if fruit_rel:
        fr = np.stack(fruit_rel) * 100.0
        ax.scatter(fr[:, 0], fr[:, 1], c="#c0392b", marker="x", label="fruto (rel. al inicio)")
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False)
    ax.set_title("Cobertura espacial XY")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "bias_xy_scatter.png", dpi=150)
    plt.close(fig)

    # --- Histogramas de suavidad / jitter ---
    specs = [
        ("jitter_rms", 1000.0, "Jitter RMS (mm)"),
        ("normalized_jerk", 1.0, "Jerk normalizado"),
        ("sparc", 1.0, "SPARC (≈0 = suave)"),
        ("path_efficiency", 1.0, "Eficiencia de trayectoria"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for a, (attr, scale, label) in zip(axes.ravel(), specs):
        v = np.array([getattr(q, attr) for q in qualities], dtype=float) * scale
        v = v[~np.isnan(v)]
        if v.size:
            a.hist(v, bins=min(15, max(5, v.size)), color="#2e8b57", edgecolor="white")
        a.set_xlabel(label)
        a.set_ylabel("N° de episodios")
        _style_axes(a)
    fig.tight_layout()
    fig.savefig(output_dir / "smoothness_histograms.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Salida: CSV y resumen en consola
# --------------------------------------------------------------------------- #

def save_csv(all_metrics: list[EpisodeMetrics], output_path: Path) -> pd.DataFrame:
    """Guarda la tabla por episodio en metrics.csv y devuelve el DataFrame.

    Las métricas de distancia/longitud se convierten de metros a centímetros
    (sufijo '_cm' en el nombre de columna, para que la unidad sea explícita
    y no ambigua al leer el CSV). 'time_to_target' se mantiene en frames
    (índice), y 'orientation_error_deg' en grados.
    """
    rows = [
        {
            "episode_id": m.episode_id,
            "num_frames": m.num_frames,
            "final_position_error_cm": m.final_position_error * 100.0,
            "min_distance_cm": m.min_distance * 100.0,
            "path_length_cm": m.path_length * 100.0,
            "time_to_target": m.time_to_target,
            "orientation_error_deg": m.orientation_error_deg,
            "success": m.success,
        }
        for m in all_metrics
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def save_quality_csv(qualities: list[TrajectoryQuality], output_path: Path) -> pd.DataFrame:
    """Guarda las métricas de dirección/jitter/suavidad por episodio en quality_metrics.csv."""
    rows = [
        {
            "episode_id": q.episode_id,
            "disp_x_cm": q.displacement[0] * 100.0,
            "disp_y_cm": q.displacement[1] * 100.0,
            "disp_z_cm": q.displacement[2] * 100.0,
            "share_x": q.signed_path_share[0],
            "share_y": q.signed_path_share[1],
            "share_z": q.signed_path_share[2],
            "mean_speed_cm_s": q.mean_speed * 100.0,
            "rms_acc": q.rms_acc,
            "rms_jerk": q.rms_jerk,
            "normalized_jerk": q.normalized_jerk,
            "sparc": q.sparc,
            "jitter_rms_mm": q.jitter_rms * 1000.0,
            "path_efficiency": q.path_efficiency,
            "reversals_per_s": q.reversals_per_s,
        }
        for q in qualities
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def _summarize(values: np.ndarray) -> dict:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n_valid": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n_valid": int(valid.size),
    }


def print_summary(df: pd.DataFrame, position_threshold: float) -> None:
    """Imprime un resumen legible en consola con todas las métricas agregadas pedidas.

    Las métricas de distancia se reportan en centímetros (consistente con
    metrics.csv); 'position_threshold' se recibe en metros (como en la CLI)
    y se convierte a cm solo para mostrarlo aquí.
    """
    n_total = len(df)
    print("\n" + "=" * 70)
    print("RESUMEN DE EVALUACIÓN — compute_metrics.py")
    print("=" * 70)
    print(f"Episodios procesados: {n_total}")
    print(f"Umbral de éxito (position_threshold): {position_threshold * 100.0:.2f} cm\n")

    def _print_block(title: str, values: np.ndarray, unit: str = "cm"):
        s = _summarize(values)
        print(f"--- {title} ---")
        print(f"  media : {s['mean']:.4f} {unit}" if not np.isnan(s["mean"]) else "  media : NaN")
        print(f"  std   : {s['std']:.4f} {unit}" if not np.isnan(s["std"]) else "  std   : NaN")
        print(f"  min   : {s['min']:.4f} {unit}" if not np.isnan(s["min"]) else "  min   : NaN")
        print(f"  max   : {s['max']:.4f} {unit}" if not np.isnan(s["max"]) else "  max   : NaN")
        print(f"  (n episodios válidos: {s['n_valid']}/{n_total})\n")

    _print_block("1. Error final de posición", df["final_position_error_cm"].to_numpy())
    _print_block("2. Distancia mínima alcanzada", df["min_distance_cm"].to_numpy())
    _print_block("4. Longitud de trayectoria", df["path_length_cm"].to_numpy())

    # Tiempo al objetivo: media/min/max (sin std, según especificación)
    ttt = df["time_to_target"].to_numpy()
    ttt_valid = ttt[~np.isnan(ttt)]
    print("--- 5. Tiempo hasta alcanzar el objetivo (en frames) ---")
    if ttt_valid.size > 0:
        print(f"  media : {np.mean(ttt_valid):.2f}")
        print(f"  min   : {np.min(ttt_valid):.2f}")
        print(f"  max   : {np.max(ttt_valid):.2f}")
    else:
        print("  Ningún episodio alcanzó el umbral; todos los valores son NaN.")
    print(f"  (objetivo alcanzado en {ttt_valid.size}/{n_total} episodios)\n")

    success_rate = float(df["success"].mean()) if n_total > 0 else np.nan
    print("--- 6. Tasa de éxito ---")
    print(f"  success_rate : {success_rate:.2%}  ({int(df['success'].sum())}/{n_total})\n")

    ori = df["orientation_error_deg"].to_numpy()
    ori_valid = ori[~np.isnan(ori)]
    print("--- 7. Error de orientación ---")
    if ori_valid.size > 0:
        print(f"  media : {np.mean(ori_valid):.3f} deg")
        print(f"  std   : {np.std(ori_valid):.3f} deg")
        print(f"  max   : {np.max(ori_valid):.3f} deg")
        print(f"  (n episodios válidos: {ori_valid.size}/{n_total})")
    else:
        print("  No se pudo calcular para ningún episodio (faltan datos de orientación).")
    print("=" * 70 + "\n")


def print_quality_and_bias_summary(qualities: list[TrajectoryQuality], bias: dict) -> None:
    """Imprime el resumen de sesgo direccional y de jitter/suavidad."""
    print("\n" + "=" * 70)
    print("SESGO DIRECCIONAL DEL DATASET")
    print("=" * 70)
    print(f"Episodios analizados: {bias['n']}")
    for name, a in bias["axes"].items():
        flag = "  <-- sesgo significativo (p<0.05)" if (a["p_ttest"] < 0.05 or a["p_sign"] < 0.05) else ""
        print(
            f"  eje {name}: desp. medio {a['mean_cm']:+.2f} ± {a['std_cm']:.2f} cm | "
            f"% episodios en +{name}: {a['frac_positive']:.0%} | "
            f"consistencia de recorrido: {a['mean_path_share']:+.2f} | "
            f"p(t)={a['p_ttest']:.3g}, p(signo)={a['p_sign']:.3g}{flag}"
        )
    az = bias["azimuth"]
    print(
        f"\n  Azimut XY: dirección media {az['mean_deg']:+.1f}°, R̄={az['R_bar']:.3f}, "
        f"p(Rayleigh)={az['p_rayleigh']:.3g}  (R̄→1 = muy concentrado)"
    )
    print(f"  Episodios que terminan hacia la izquierda ({bias['left_axis']}): {bias['left_frac']:.0%}")
    if bias["fruit_rel_mean_cm"] is not None:
        fr = bias["fruit_rel_mean_cm"]
        print(f"  Posición media del fruto relativa al inicio: x={fr[0]:+.1f}, y={fr[1]:+.1f}, z={fr[2]:+.1f} cm")

    def _line(title: str, vals, unit: str = "", scale: float = 1.0):
        v = np.array(vals, dtype=float)
        v = v[~np.isnan(v)] * scale
        if v.size:
            print(f"  {title}: media {v.mean():.3f} | std {v.std():.3f} | min {v.min():.3f} | max {v.max():.3f} {unit}")

    print("\n" + "=" * 70)
    print("JITTER Y SUAVIDAD DE TRAYECTORIA")
    print("=" * 70)
    _line("Jitter RMS            ", [q.jitter_rms for q in qualities], "mm", 1000.0)
    _line("RMS aceleración       ", [q.rms_acc for q in qualities], "m/s²")
    _line("RMS jerk              ", [q.rms_jerk for q in qualities], "m/s³")
    _line("Jerk normalizado      ", [q.normalized_jerk for q in qualities])
    _line("SPARC (≈0 = suave)    ", [q.sparc for q in qualities])
    _line("Eficiencia trayectoria", [q.path_efficiency for q in qualities])
    _line("Reversiones/s         ", [q.reversals_per_s for q in qualities])
    _line("Rapidez media         ", [q.mean_speed for q in qualities], "cm/s", 100.0)
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------- #
# Orquestación principal
# --------------------------------------------------------------------------- #

def discover_episode_dirs(input_dir: Path) -> list[Path]:
    """Encuentra subcarpetas de episodio dentro de input_dir, ordenadas por nombre.

    Se asume que cada subcarpeta inmediata de input_dir es un episodio
    (formato observado: '000', '001', '002', ...). No se asume ningún
    patrón de nombre específico más allá de "es un directorio".
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"El directorio de entrada no existe: {input_dir}")
    episode_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not episode_dirs:
        warnings.warn(f"No se encontraron subcarpetas de episodio en {input_dir}")
    return episode_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Calcula métricas de evaluación offline para episodios SplatSim/GELLO (PKL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directorio raíz con subcarpetas de episodio (ej. ~/data/bc_data/gello/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./metrics_output",
        help=(
            "Directorio base de resultados. Cada dataset se guarda en su propia subcarpeta "
            "(ver --run_name), así no se sobrescriben resultados de otros datasets."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Nombre de la subcarpeta dentro de --output_dir. Por defecto se arma con las "
            "dos últimas partes de --input_dir (ej. ~/data/policy_trials/261026_203030 "
            "-> policy_trials_261026_203030)."
        ),
    )
    parser.add_argument(
        "--position_threshold",
        type=float,
        default=0.07,
        help="Umbral de distancia (m) para considerar éxito / objetivo alcanzado.",
    )
    parser.add_argument(
        "--euler_seq",
        type=str,
        default="xyz",
        help=(
            "Secuencia de ángulos de Euler usada como fallback para la orientación "
            "del efector cuando 'ee_pos_quat' no está disponible en el frame. "
            "Ver scipy.spatial.transform.Rotation.from_euler (seq)."
        ),
    )
    parser.add_argument(
        "--gripper_offset",
        type=float,
        default=0.18,
        help=(
            "Desplazamiento en metros entre la pose de la muñeca ('state'/'action') "
            "y el centro real de las pinzas del gripper, aplicado en el eje +Z local "
            "del efector y rotado por la orientación de cada frame. Validado "
            "numéricamente contra un frame real de aproximación confirmada "
            "(ver docstring del módulo); +Z local minimiza la distancia "
            "gripper-fruto, no +X."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help=(
            "Frecuencia de muestreo (para velocidad/aceleración/jerk). Solo se usa en "
            "episodios sin 'sim_time'; si los frames lo traen, se usa su dt real."
        ),
    )
    parser.add_argument(
        "--left_axis",
        type=str,
        default="+y",
        help="Eje del frame del robot que corresponde a 'izquierda' (ej. '+y', '-x').",
    )
    parser.add_argument(
        "--quality_source",
        choices=["gripper", "wrist"],
        default="gripper",
        help=(
            "Posición usada para jitter/suavidad/sesgo. 'gripper' incluye el offset rotado "
            "por la orientación (amplifica el ruido de orientación); 'wrist' usa la posición cruda."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce la verbosidad de las advertencias impresas en consola.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    run_name = args.run_name or "_".join(input_dir.parts[-2:])
    output_dir = Path(args.output_dir).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resultados en: {output_dir}")

    # parámetros usados, para poder comparar corridas después
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({**vars(args), "input_dir": str(input_dir)}, f, indent=2)

    if args.quiet:
        warnings.filterwarnings("once")
    else:
        warnings.filterwarnings("always")

    episode_dirs = discover_episode_dirs(input_dir)
    print(f"Encontrados {len(episode_dirs)} episodios en {input_dir}")

    all_metrics: list[EpisodeMetrics] = []
    all_qualities: list[TrajectoryQuality] = []
    for ep_dir in episode_dirs:
        print(f"Procesando episodio '{ep_dir.name}'...")
        episode_data = build_episode_data(
            ep_dir, euler_seq=args.euler_seq, gripper_offset=args.gripper_offset
        )
        if episode_data is None:
            warnings.warn(f"Episodio '{ep_dir.name}' fue omitido por falta de datos válidos.")
            continue
        metrics = compute_episode_metrics(episode_data, position_threshold=args.position_threshold)
        all_metrics.append(metrics)

        positions = (
            episode_data.gripper_positions
            if args.quality_source == "gripper"
            else episode_data.ee_positions
        )
        fps = 1.0 / episode_data.dt if episode_data.dt else args.fps
        print(f"  fps: {fps:.2f} ({'sim_time' if episode_data.dt else '--fps'})")
        quality = compute_trajectory_quality(
            positions, episode_data.fruit_position, episode_data.episode_id, fps=fps
        )
        if quality is not None:
            all_qualities.append(quality)

    if not all_metrics:
        print("No se pudo calcular ninguna métrica: revisar advertencias anteriores.")
        return

    csv_path = output_dir / "metrics.csv"
    df = save_csv(all_metrics, csv_path)
    print(f"\nTabla por episodio guardada en: {csv_path}")

    generate_plots(all_metrics, output_dir, position_threshold=args.position_threshold)
    print(f"Gráficos guardados en: {output_dir}")

    print_summary(df, position_threshold=args.position_threshold)

    # --- Sesgo direccional + jitter/suavidad ---
    if all_qualities:
        bias = analyze_dataset_bias(all_qualities, left_axis=args.left_axis)
        quality_csv_path = output_dir / "quality_metrics.csv"
        save_quality_csv(all_qualities, quality_csv_path)
        print(f"Métricas de calidad por episodio guardadas en: {quality_csv_path}")
        generate_quality_plots(all_qualities, bias, output_dir)
        print_quality_and_bias_summary(all_qualities, bias)
    else:
        print("No se pudo calcular el análisis de calidad/sesgo (episodios muy cortos o sin datos).")


if __name__ == "__main__":
    main()