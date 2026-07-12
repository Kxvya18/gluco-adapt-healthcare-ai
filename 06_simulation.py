

import numpy as np
import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware_config import (
    AsymmetricClinicalLoss,
    AdaptiveSamplingPolicy,
    PowerModeConfig,
    WearableHardwareConstants,
    ClinicalConstants
)
from sensors import NIROpticalSensor, ElectrochemicalSweatSensor, IMUMotionDetector, PPGContextLayer
from predictor import QuantizedLSTMPredictor, PhysiologicalGlucoseGenerator, SimulationPredictor
                                                  

def clarke_error_grid(y_ref: np.ndarray, y_pred: np.ndarray) -> dict:

    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    n = len(y_ref)

    for ref, pred in zip(y_ref, y_pred):
        zone = _classify_clarke_zone(ref, pred)
        zones[zone] += 1

    total = sum(zones.values())
    percentages = {k: v / total * 100 for k, v in zones.items()}
    return {
        'counts': zones,
        'percentages': percentages,
        'zone_AB_pct': percentages['A'] + percentages['B']
    }


def _classify_clarke_zone(ref: float, pred: float) -> str:
    """Classify a single (reference, predicted) pair into Clarke zone."""
                                                                     
    if ref < 70 and pred < 70:
        return 'A'
    if ref > 240 and pred > 240:
        return 'A'
    if abs(pred - ref) / max(ref, 0.01) <= 0.20:
        return 'A'

                                  
    if ref >= 180 and pred <= 70:
        return 'E'
    if ref <= 70 and pred >= 180:
        return 'E'

                                         
    if ref >= 240 and pred <= 70:
        return 'D'
    if ref <= 70 and pred >= 240:
        return 'D'
    if 70 <= ref <= 180 and pred < 70:
        return 'D'
    if 70 <= ref <= 180 and pred > 240:
        return 'D'

                            
    if ref > 240 and 70 <= pred <= 180:
        return 'C'
    if ref < 70 and 70 <= pred <= 180:
        return 'C'

                               
    return 'B'


def compute_mard(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Relative Difference (MARD).
    Primary accuracy metric for CGM systems per ISO 15197:2013.
    """
    relative_diff = np.abs(y_pred - y_ref) / np.maximum(y_ref, 0.01)
    return float(np.mean(relative_diff) * 100.0)


def detect_hypoglycemia_lead_time(
    true_glucose: np.ndarray,
    risk_scores: np.ndarray,
    t_hours: np.ndarray,
    alert_threshold: float = 0.30
) -> dict:
    
    hypo_threshold = ClinicalConstants.HYPO_THRESHOLD            
    events = []

    in_hypo = False
    hypo_start_idx = None
    alert_triggered_idx = None

                                                                 
    alert_lookback_idx = 0

    for i, (g, r, t) in enumerate(zip(true_glucose, risk_scores, t_hours)):
                                                           
        if g < hypo_threshold and not in_hypo:
            in_hypo = True
            hypo_start_idx = i

                                                        
            lookback_start = max(0, i - 60)
            pre_event_alerts = [
                j for j in range(lookback_start, i)
                if risk_scores[j] >= alert_threshold
            ]

            if pre_event_alerts:
                first_alert_idx = pre_event_alerts[0]
                lead_minutes = (t - t_hours[first_alert_idx]) * 60.0
                events.append({
                    'hypo_start_hour': float(t),
                    'alert_hour': float(t_hours[first_alert_idx]),
                    'lead_time_min': float(lead_minutes),
                    'glucose_at_alert': float(true_glucose[first_alert_idx])
                })
            else:
                events.append({
                    'hypo_start_hour': float(t),
                    'alert_hour': float(t),
                    'lead_time_min': 0.0,
                    'glucose_at_alert': float(g)
                })

        if g >= hypo_threshold and in_hypo:
            in_hypo = False

    lead_times = [e['lead_time_min'] for e in events if e['lead_time_min'] > 0]

    return {
        'events': events,
        'n_events_detected': len(lead_times),
        'mean_lead_time_min': float(np.mean(lead_times)) if lead_times else 0.0,
        'min_lead_time_min': float(np.min(lead_times)) if lead_times else 0.0
    }


                                                              
                              
 
                                                            
 
                                                    
                                                        
                                                                 
                                                        
 
                                                             
                                                            
                                                   
                                                     
                                                   
                                                                  
                                                      
                                                              

def run_unified_pipeline(
    duration_hours: float = 24.0,
    augmentation_enabled: bool = True,
    verbose: bool = True
) -> tuple:
    """
    Unified GLUCO-ADAPT pipeline.
    augmentation_enabled=False -> context layer only (existing Generic wearable HW)
    augmentation_enabled=True  -> full system: context + NIR + sweat (full-system configuration)

    Context layer always runs. Augmentation activates when risk score
    crosses alert threshold (NIR on) or burst threshold (sweat on).
    When NIR active, its estimate becomes the primary LSTM input.
    When dormant, LSTM runs on PPG context signal only.
    """
    tag = ("FULL SYSTEM: Context + NIR + Sweat [Full Configuration]"
           if augmentation_enabled else
           "CONTEXT LAYER ONLY: PPG + IMU + Temp [Existing Generic wearable HW]")
    if verbose:
        print(f"\n[{tag}]")
        print("1. Generating physiological trajectory + sensor signals")

    rng = np.random.RandomState(42)

    data_gen  = PhysiologicalGlucoseGenerator(seed=42)
    ppg_ctx   = PPGContextLayer(seed=42)
    imu_det   = IMUMotionDetector(window_samples=10)
    loss_fn   = AsymmetricClinicalLoss()
    predictor = SimulationPredictor(horizon_min=15, window_min=30)
    nir_sensor = NIROpticalSensor(seed=42)   if augmentation_enabled else None
    electro    = ElectrochemicalSweatSensor(seed=99) if augmentation_enabled else None

                                                                  
    cfg = PowerModeConfig()
    if augmentation_enabled:
        cfg.sparse_power_mw = (WearableHardwareConstants.CPU_DEEP_SLEEP_MW
            + PPGContextLayer.PPG_BASELINE_MW
            + WearableHardwareConstants.NIR_STANDBY_MW
            + WearableHardwareConstants.ELECTRO_POWERDOWN_MW
            + WearableHardwareConstants.IMU_MW)                  
        cfg.alert_power_mw  = (WearableHardwareConstants.CPU_IDLE_MW
            + PPGContextLayer.TOTAL_SENSING_MW
            + WearableHardwareConstants.NIR_CONTINUOUS_MW
            + WearableHardwareConstants.ELECTRO_WARMUP_MW * 0.1
            + WearableHardwareConstants.IMU_MW)                   
        cfg.burst_power_mw  = (WearableHardwareConstants.CPU_NEURAL_ENGINE_MW
            + PPGContextLayer.TOTAL_SENSING_MW
            + WearableHardwareConstants.NIR_CONTINUOUS_MW
            + WearableHardwareConstants.ELECTRO_ACTIVE_MW
            + WearableHardwareConstants.IMU_MW)                    
    else:
        cfg.sparse_power_mw = (WearableHardwareConstants.CPU_DEEP_SLEEP_MW
            + PPGContextLayer.PPG_BASELINE_MW
            + WearableHardwareConstants.IMU_MW)                   
        cfg.alert_power_mw  = (WearableHardwareConstants.CPU_IDLE_MW
            + PPGContextLayer.TOTAL_SENSING_MW
            + WearableHardwareConstants.IMU_MW)                   
        cfg.burst_power_mw  = (WearableHardwareConstants.CPU_NEURAL_ENGINE_MW
            + PPGContextLayer.TOTAL_SENSING_MW
            + WearableHardwareConstants.IMU_MW)                    
    policy = AdaptiveSamplingPolicy(cfg)

    t_hours, true_glucose = data_gen.generate(
        duration_hours=duration_hours, sampling_rate_hz=1.0/60.0, include_hypo=True)
    n = len(t_hours)
    exercise_periods = [(10.0, 1.0, 0.7), (15.0, 0.5, 0.9)]
    imu_accel  = imu_det.simulate_imu(t_hours, exercise_periods)
    ppg_raw    = ppg_ctx.get_context_signal(true_glucose, imu_accel, t_hours)
    temp_dev   = 1.5 * np.sin(2 * np.pi * t_hours / 24.0) + 0.3 * rng.randn(n)
    optical_raw = (nir_sensor.simulate(true_glucose, imu_accel, temp_dev)
                   if augmentation_enabled else None)

                   
    operational_modes = np.full(n, 'sparse', dtype=object)
    risk_scores       = np.zeros(n)
    power_mw          = np.zeros(n)
    motion_suppressed = np.zeros(n, dtype=bool)
    nir_active        = np.zeros(n, dtype=bool)
    primary_estimates = np.full(n, 100.0)
    drift_correction  = 0.0
    electro_events    = []
    glucose_history, roc_history, t_history = [], [], []

    if verbose:
        print("2. Running unified adaptive sampling loop")

    for i in range(n):
                                  
        accel_z = imu_accel[i] + 1.0
        motion_detected, motion_uncertainty = imu_det.update(0.0, 0.0, accel_z)
        motion_suppressed[i] = motion_detected

                                                
        ppg_est = (float(np.clip(ppg_raw[i], 30.0, 400.0)) if not motion_detected
                   else (primary_estimates[i-1] if i > 0 else 100.0))

                                                                           
        current_mode = policy.current_mode

                                                                         
        nir_est = None
        sensor_uncertainty = PPGContextLayer.feature_noise_std

        if augmentation_enabled and not motion_detected and current_mode in ('alert', 'burst'):
            nir_active[i] = True
            corrected_nir  = optical_raw[i] - drift_correction
            nir_est        = float(np.clip(corrected_nir, 30.0, 400.0))
            drift_unc      = abs(drift_correction) * 0.1
            _, sensor_uncertainty = nir_sensor.get_reading_with_uncertainty(
                nir_est, motion_uncertainty, drift_unc)

                                                                         
            if electro.should_activate(t_hours[i], current_mode):
                sweat_reading, correction = electro.get_correction(
                    true_glucose[i], nir_est, t_hours[i])
                drift_correction += correction * 0.3
                electro_events.append({
                    'idx': i, 'hour': float(t_hours[i]),
                    'sweat_reading': float(sweat_reading),
                    'drift_correction': float(correction),
                    'mode': current_mode
                })

                                                                       
        primary_estimates[i] = nir_est if nir_est is not None else ppg_est

                                    
        glucose_history.append(primary_estimates[i])
        t_history.append(t_hours[i])
        dt = (t_hours[i] - t_hours[i-1]) * 60.0 if i > 0 else 1.0
        roc = ((glucose_history[-1] - glucose_history[-2]) / dt
               if len(glucose_history) > 1 and dt > 0 else 0.0)
        roc_history.append(roc)
        accel_g = ((roc_history[-1] - roc_history[-2]) / dt
                   if len(roc_history) > 1 and dt > 0 else 0.0)

        pred_glucose, pred_uncertainty = predictor.predict(
            glucose_history, roc_history, t_history, horizon_steps=3)

        combined_uncertainty = np.sqrt(pred_uncertainty**2 + sensor_uncertainty**2)
        if motion_detected:
            combined_uncertainty = np.sqrt(combined_uncertainty**2 + motion_uncertainty**2)

                                                        
        risk_score = loss_fn.compute_risk_score(
            predicted_glucose   = pred_glucose,
            pred_uncertainty    = combined_uncertainty,
            optical_glucose_est = primary_estimates[i],
            rate_of_change      = roc,
            acceleration        = accel_g
        )
        risk_scores[i] = risk_score

                               
        mode, _, mode_power = policy.update(risk_score)
        operational_modes[i] = mode
        power_mw[i]          = mode_power

    if verbose:
        print("3. Computing evaluation metrics")

    valid_mask = ~motion_suppressed
    mard     = compute_mard(true_glucose[valid_mask], primary_estimates[valid_mask])
    ceg      = clarke_error_grid(true_glucose[valid_mask], primary_estimates[valid_mask])
    hypo_det = detect_hypoglycemia_lead_time(true_glucose, risk_scores, t_hours)

    unique_m, counts_m = np.unique(operational_modes, return_counts=True)
    mode_pct = {str(m): float(c/n*100) for m, c in zip(unique_m, counts_m)}
    mean_pw  = float(np.mean(power_mw))
    baseline_pw = cfg.burst_power_mw

    results = {
        'mode':                  'FULL_SYSTEM' if augmentation_enabled else 'CONTEXT_LAYER',
        'label':                 ('Full System: Context + NIR + Sweat (Full Configuration)'
                                  if augmentation_enabled else
                                  'Context Layer Only: PPG + IMU + Temp (Existing HW)'),
        'augmentation_enabled':  augmentation_enabled,
        'duration_hours':        duration_hours,
        'total_samples':         n,
        'mean_true_glucose':     float(np.mean(true_glucose)),
        'glucose_min':           float(np.min(true_glucose)),
        'glucose_max':           float(np.max(true_glucose)),
        'time_in_range_pct':     float(np.mean((true_glucose >= 70) & (true_glucose <= 180)) * 100),
        'hypo_samples':          int(np.sum(true_glucose < 70)),
        'severe_hypo_samples':   int(np.sum(true_glucose < 54)),
        'mard_pct':              mard,
        'clarke_zone_A_pct':     ceg['percentages']['A'],
        'clarke_zone_AB_pct':    ceg['zone_AB_pct'],
        'clarke_zone_D_pct':     ceg['percentages']['D'],
        'clarke_zone_E_pct':     ceg['percentages']['E'],
        'hypo_events_detected':  hypo_det['n_events_detected'],
        'mean_lead_time_min':    hypo_det['mean_lead_time_min'],
        'min_lead_time_min':     hypo_det.get('min_lead_time_min', 0.0),
        'hypo_event_details':    hypo_det['events'],
        'electro_corrections':   len(electro_events),
        'electro_event_details': electro_events,
        'motion_suppressed_pct': float(np.mean(motion_suppressed) * 100),
        'nir_active_pct':        float(np.mean(nir_active) * 100),
        'valid_samples':         int(np.sum(valid_mask)),
        'mean_power_mw':         mean_pw,
        'burst_baseline_power_mw': baseline_pw,
        'power_reduction_pct':   (1.0 - mean_pw / baseline_pw) * 100.0,
        'within_power_budget':      mean_pw <= WearableHardwareConstants.GLUCOSE_POWER_BUDGET_MW,
        'glucose_budget_mw':     float(WearableHardwareConstants.GLUCOSE_POWER_BUDGET_MW),
        'mode_distribution_pct': mode_pct,
        'mean_risk_score':       float(np.mean(risk_scores)),
        'max_risk_score':        float(np.max(risk_scores)),
        'pct_time_above_alert':  float(np.mean(risk_scores >= 0.30) * 100),
        'pct_time_in_burst':     float(np.mean(risk_scores >= 0.70) * 100),
    }

    if verbose:
        nir_pct = float(np.mean(nir_active)*100) if augmentation_enabled else 0.0
        print(f"   MARD: {mard:.1f}%  | Clarke A+B: {ceg['zone_AB_pct']:.1f}%"
              f"  | Lead time: {hypo_det['mean_lead_time_min']:.0f} min"
              f"  | Power: {mean_pw:.1f}mW"
              + (f"  | NIR active: {nir_pct:.1f}% of time" if augmentation_enabled else ""))

    return results, t_hours, true_glucose, primary_estimates, risk_scores, operational_modes, power_mw, nir_active


                              
def run_simulation(duration_hours=24.0, verbose=True):
    res, t, tg, est, risk, modes, pw, nir = run_unified_pipeline(
        duration_hours, augmentation_enabled=True, verbose=verbose)
    return res, t, tg, est, risk, modes, pw

def run_context_layer_only(duration_hours=24.0, verbose=True):
    res, t, tg, est, risk, modes, pw, nir = run_unified_pipeline(
        duration_hours, augmentation_enabled=False, verbose=verbose)
    return res, t, tg, est, risk, modes, pw



def print_results(results: dict):
    """Print formatted simulation results."""

    print("\n" + "="*60)
    print("GLUCO-ADAPT SIMULATION RESULTS")
    print("Hardware: Generic wearable wearable processor (simulated)")
    print("="*60)

    print(f"\nSimulation: {results['duration_hours']:.0f} hours, "
          f"{results['total_samples']} samples @ 1/min")

    print("\n── GLUCOSE STATISTICS ──────────────────────────────")
    print(f"  Mean glucose:           {results['mean_true_glucose']:.1f} mg/dL")
    print(f"  Range:                  {results['glucose_min']:.1f} – {results['glucose_max']:.1f} mg/dL")
    print(f"  Time in range (70-180): {results['time_in_range_pct']:.1f}%")
    print(f"  Hypoglycemic samples:   {results['hypo_samples']} ({results['hypo_samples']/results['total_samples']*100:.1f}%)")
    print(f"  Severe hypo (<54):      {results['severe_hypo_samples']}")

    print("\n── SENSOR ACCURACY (optical estimates vs ground truth) ──")
    print(f"  MARD:                   {results['mard_pct']:.1f}%")
    print(f"  Clarke Zone A:          {results['clarke_zone_A_pct']:.1f}%")
    print(f"  Clarke Zone A+B:        {results['clarke_zone_AB_pct']:.1f}%")
    print(f"  Clarke Zone D+E:        {results['clarke_zone_D_pct']+results['clarke_zone_E_pct']:.1f}%  [target: <5%]")
    print(f"  Motion-suppressed:      {results['motion_suppressed_pct']:.1f}% of samples gated by IMU")
    print(f"  Drift corrections:      {results['electro_corrections']} electrochemical anchors applied")

    print("\n── SAFETY METRICS ──────────────────────────────────")
    print(f"  Hypo events detected:   {results['hypo_events_detected']}")
    print(f"  Mean lead time:         {results['mean_lead_time_min']:.1f} min before crossing 70 mg/dL")
    print(f"  Min lead time:          {results['min_lead_time_min']:.1f} min")

    print("\n── POWER METRICS (wearable hardware budget) ────────────")
    print(f"  Mean glucose subsystem: {results['mean_power_mw']:.2f} mW")
    print(f"  Power budget:      {results['glucose_budget_mw']:.1f} mW")
    print(f"  Within budget:          {'✓ YES' if results['within_power_budget'] else '✗ NO'}")
    print(f"  vs. fixed burst rate:   {results['power_reduction_pct']:.1f}% reduction")

    print("\n── MODE DISTRIBUTION ───────────────────────────────")
    for mode, pct in results['mode_distribution_pct'].items():
        bar = '█' * int(pct / 2)
        print(f"  {mode.capitalize():8s}: {pct:5.1f}%  {bar}")

    print("\n── RISK SCORE STATISTICS ───────────────────────────")
    print(f"  Mean risk score:        {results['mean_risk_score']:.3f}")
    print(f"  Max risk score:         {results['max_risk_score']:.3f}")
    print(f"  Time above alert (0.3): {results['pct_time_above_alert']:.1f}%")
    print(f"  Time in burst (0.7):    {results['pct_time_in_burst']:.1f}%")

    print("\n" + "="*60)


def generate_plots(
    results: dict,
    t_hours: np.ndarray,
    true_glucose: np.ndarray,
    optical_estimates: np.ndarray,
    risk_scores: np.ndarray,
    operational_modes: np.ndarray,
    power_mw: np.ndarray,
    output_dir: str = './results'
):
    """
    Generate the full GLUCO-ADAPT visualisation suite.

    Six panels in a single figure:
      1. Glucose trajectories  — true vs NIR optical estimate, hypo zones,
                                  sweat electrochemical correction markers
      2. Operational mode      — colour-coded timeline (sparse/alert/burst)
      3. Risk score            — asymmetric loss hazard score over time,
                                  threshold lines
      4. Power consumption     — instantaneous mW vs. fixed-rate baseline
      5. Clarke Error Grid     — clinical accuracy scatter
      6. Summary metrics bar   — MARD, Clarke zones, mode distribution,
                                  power reduction
    """

    os.makedirs(output_dir, exist_ok=True)

                                                                
    C = {
        'true':    '#1a1a2e',                               
        'nir':     '#4cc9f0',                              
        'sweat':   '#f77f00',                                          
        'sparse':  '#2dc653',                              
        'alert':   '#f4d35e',                             
        'burst':   '#e63946',                             
        'risk':    '#9b2226',                             
        'budget':  '#457b9d',                                    
        'actual':  '#e9c46a',                               
        'bg':      '#f8f9fa',                                    
        'panel':   '#ffffff',                                  
        'grid':    '#dee2e6',                       
        'hypo':    '#ffb3b3',                                 
        'target':  '#b7e4c7',                                   
    }

                                                                   
    mode_color_map = {'sparse': C['sparse'], 'alert': C['alert'], 'burst': C['burst']}
    mode_num = np.array([
        {'sparse': 0, 'alert': 1, 'burst': 2}[str(m)] for m in operational_modes
    ], dtype=float)

                                                      
    electro_hours = [e['hour'] for e in results.get('electro_event_details', [])]
    electro_sweat = [e['sweat_reading'] for e in results.get('electro_event_details', [])]

                                                                
    fig = plt.figure(figsize=(20, 24), facecolor=C['bg'])
    fig.suptitle(
        'GLUCO-ADAPT: Adaptive Risk-Weighted Glucose Monitoring\n'
        'Generic wearable wearable processor Simulation  |  24-Hour Demonstration',
        fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e'
    )

    gs = gridspec.GridSpec(
        6, 2,
        figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.07, right=0.96,
        top=0.94, bottom=0.04
    )

                                                                
                                                               
                                                                
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.set_facecolor(C['panel'])

                     
    ax1.axhspan(0,   54,  alpha=0.15, color=C['burst'],  zorder=0, label='Severe Hypo (<54)')
    ax1.axhspan(54,  70,  alpha=0.10, color=C['burst'],  zorder=0, label='Hypo (54–70)')
    ax1.axhspan(70,  180, alpha=0.08, color=C['sparse'], zorder=0, label='Target (70–180)')
    ax1.axhline(70,  color=C['burst'],  lw=1.2, ls='--', alpha=0.6, zorder=1)
    ax1.axhline(180, color='#aaa',      lw=0.8, ls='--', alpha=0.4, zorder=1)
    ax1.axhline(54,  color=C['burst'],  lw=1.0, ls=':',  alpha=0.8, zorder=1)

                                              
    points = np.array([t_hours, optical_estimates]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_mode = ListedColormap([C['sparse'], C['alert'], C['burst']])
    norm_mode = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_mode.N)
    lc = LineCollection(segs, cmap=cmap_mode, norm=norm_mode, linewidth=1.6, alpha=0.85, zorder=3)
    lc.set_array(mode_num[:-1])
    ax1.add_collection(lc)

                                      
    ax1.plot(t_hours, true_glucose, color=C['true'], lw=2.2, alpha=0.9,
             label='True Glucose (reference)', zorder=4)

                                  
    if electro_hours:
        ax1.scatter(
            electro_hours, electro_sweat,
            color=C['sweat'], s=60, zorder=5, marker='D',
            label=f'Sweat Correction Events (n={len(electro_hours)})',
            edgecolors='white', linewidths=0.8
        )

                                               
    nir_sparse = mpatches.Patch(color=C['sparse'], label='NIR — Sparse Mode')
    nir_alert  = mpatches.Patch(color=C['alert'],  label='NIR — Alert Mode')
    nir_burst  = mpatches.Patch(color=C['burst'],  label='NIR — Burst Mode')

    ax1.set_xlim(0, t_hours[-1])
    ax1.set_ylim(30, 220)
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Glucose (mg/dL)', fontsize=11)
    ax1.set_title(
        'Panel 1 — Glucose Trajectories: True vs NIR Optical Estimate\n'
        'NIR trace coloured by operational mode  |  ◆ = sweat drift-correction events',
        fontsize=11, fontweight='bold', pad=8
    )
    ax1.legend(
        handles=[
            mpatches.Patch(color=C['true'], label='True Glucose'),
            nir_sparse, nir_alert, nir_burst,
            mpatches.Patch(color=C['sweat'], label='Sweat Correction'),
        ],
        loc='upper right', fontsize=9, framealpha=0.9
    )
    ax1.grid(axis='y', color=C['grid'], lw=0.6, zorder=0)

                             
    ax1.text(0.5, 71.5, '70 mg/dL — Hypoglycaemia threshold',
             fontsize=8, color=C['burst'], alpha=0.8)

                                                                
                                         
                                                                
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_facecolor(C['panel'])

                                                                    
    mode_colors_array = np.array([
        {'sparse': [0.18, 0.78, 0.33, 1],
         'alert':  [0.96, 0.83, 0.37, 1],
         'burst':  [0.90, 0.22, 0.27, 1]}[str(m)]
        for m in operational_modes
    ])
    dt = t_hours[1] - t_hours[0]
    for i in range(0, len(t_hours), 1):
        c = mode_colors_array[i]
        ax2.barh(0, dt, left=t_hours[i], height=1.0, color=c, linewidth=0)

    ax2.set_xlim(0, t_hours[-1])
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_title(
        'Panel 2 — Operational Mode Timeline  '
        '(Green = Sparse | Yellow = Alert | Red = Burst)',
        fontsize=11, fontweight='bold', pad=6
    )

                                    
    mdp = results['mode_distribution_pct']
    for mode_name, color in [('sparse', C['sparse']), ('alert', C['alert']), ('burst', C['burst'])]:
        pct = mdp.get(mode_name, 0)
        if pct > 3:
            ax2.text(
                t_hours[-1] * 0.5, 0,
                f"{mode_name.upper()} {pct:.1f}%",
                fontsize=9, ha='center', va='center', color='white',
                fontweight='bold', alpha=0.0                                
            )

            
    ax2.legend(
        handles=[
            mpatches.Patch(color=C['sparse'], label=f"Sparse  {mdp.get('sparse',0):.1f}%  (1 sample/5 min)"),
            mpatches.Patch(color=C['alert'],  label=f"Alert   {mdp.get('alert', 0):.1f}%  (1 sample/min)"),
            mpatches.Patch(color=C['burst'],  label=f"Burst   {mdp.get('burst', 0):.1f}%  (1 sample/10 sec)"),
        ],
        loc='upper right', fontsize=9, framealpha=0.9
    )

                                                                
                                         
                                                                
    ax3 = fig.add_subplot(gs[3, 0])
    ax3.set_facecolor(C['panel'])

    ax3.fill_between(t_hours, 0, risk_scores,
                     where=(risk_scores < 0.30),
                     color=C['sparse'], alpha=0.4, label='Risk < 0.3 (Sparse)')
    ax3.fill_between(t_hours, 0, risk_scores,
                     where=(risk_scores >= 0.30) & (risk_scores < 0.70),
                     color=C['alert'], alpha=0.5, label='Risk 0.3–0.7 (Alert)')
    ax3.fill_between(t_hours, 0, risk_scores,
                     where=(risk_scores >= 0.70),
                     color=C['burst'], alpha=0.6, label='Risk > 0.7 (Burst)')

    ax3.axhline(0.30, color=C['alert'], lw=1.4, ls='--', label='Alert threshold (0.30)')
    ax3.axhline(0.70, color=C['burst'], lw=1.4, ls='--', label='Burst threshold (0.70)')
    ax3.plot(t_hours, risk_scores, color=C['risk'], lw=0.8, alpha=0.7)

    ax3.set_xlim(0, t_hours[-1])
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel('Time (hours)', fontsize=10)
    ax3.set_ylabel('Risk Score', fontsize=10)
    ax3.set_title('Panel 3 — Asymmetric Clinical\nHazard Score', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, framealpha=0.9)
    ax3.grid(color=C['grid'], lw=0.5)

                        
    ax3.text(0.97, 0.95,
             f"Mean: {results['mean_risk_score']:.3f}\nMax: {results['max_risk_score']:.3f}",
             transform=ax3.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                                                                
                                                 
                                                                
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.set_facecolor(C['panel'])

    baseline_pw = results['burst_baseline_power_mw']
    budget_pw   = results['glucose_budget_mw']

    ax4.fill_between(t_hours, 0, power_mw, color=C['actual'], alpha=0.7,
                     label=f"Adaptive power (mean {results['mean_power_mw']:.1f} mW)")
    ax4.axhline(baseline_pw, color=C['burst'],  lw=1.6, ls='--',
                label=f'Fixed burst baseline ({baseline_pw:.0f} mW)')
    ax4.axhline(budget_pw,   color=C['budget'], lw=1.6, ls='-.',
                label=f'Hardware budget ({budget_pw:.0f} mW)')

    ax4.set_xlim(0, t_hours[-1])
    ax4.set_ylim(0, baseline_pw * 1.15)
    ax4.set_xlabel('Time (hours)', fontsize=10)
    ax4.set_ylabel('Power (mW)', fontsize=10)
    ax4.set_title('Panel 4 — Power Consumption\nvs  Hardware Budget', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8, framealpha=0.9)
    ax4.grid(color=C['grid'], lw=0.5)

    ax4.text(0.97, 0.30,
             f"Reduction vs baseline:\n{results['power_reduction_pct']:.1f}%",
             transform=ax4.transAxes, ha='right', va='top', fontsize=9,
             color=C['budget'], fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                                                                
                                                
                                                                
    ax5 = fig.add_subplot(gs[4:, 0])
    ax5.set_facecolor(C['panel'])

                                                           
                                    
    ref_line = np.linspace(0, 400, 400)
    ax5.fill_between(ref_line, ref_line * 0.80, ref_line * 1.20,
                     alpha=0.12, color=C['sparse'], zorder=0, label='Zone A (clinically accurate)')

                                                 
    sample_step = max(1, len(true_glucose) // 500)                          
    ref_s   = true_glucose[::sample_step]
    pred_s  = optical_estimates[::sample_step]
    mode_s  = operational_modes[::sample_step]
    colors_s = [mode_color_map.get(str(m), '#888') for m in mode_s]

    ax5.scatter(ref_s, pred_s, c=colors_s, s=18, alpha=0.55, zorder=3, edgecolors='none')
    ax5.plot([0, 400], [0, 400], 'k--', lw=1.0, alpha=0.4, label='Perfect agreement')
    ax5.plot([0, 400], [0 * 1.20, 400 * 1.20], color='#aaa', lw=0.7, ls=':', alpha=0.5)
    ax5.plot([0, 400], [0 * 0.80, 400 * 0.80], color='#aaa', lw=0.7, ls=':', alpha=0.5)

                          
    ax5.axvline(70,  color=C['burst'], lw=0.9, ls='--', alpha=0.5)
    ax5.axhline(70,  color=C['burst'], lw=0.9, ls='--', alpha=0.5)
    ax5.axvline(180, color='#aaa',     lw=0.7, ls='--', alpha=0.4)
    ax5.axhline(180, color='#aaa',     lw=0.7, ls='--', alpha=0.4)

    ax5.set_xlim(30, 220)
    ax5.set_ylim(30, 220)
    ax5.set_xlabel('Reference Glucose (mg/dL)', fontsize=10)
    ax5.set_ylabel('NIR Estimated Glucose (mg/dL)', fontsize=10)
    ax5.set_title(
        'Panel 5 — Clarke Error Grid\n'
        f"Zone A: {results['clarke_zone_A_pct']:.1f}%  |  "
        f"A+B: {results['clarke_zone_AB_pct']:.1f}%  |  "
        f"MARD: {results['mard_pct']:.1f}%",
        fontsize=11, fontweight='bold'
    )
    ax5.legend(
        handles=[
            mpatches.Patch(color=C['sparse'], label=f"Sparse mode points"),
            mpatches.Patch(color=C['alert'],  label=f"Alert mode points"),
            mpatches.Patch(color=C['burst'],  label=f"Burst mode points"),
            mpatches.Patch(color=C['sparse'], alpha=0.3, label='Zone A band (±20%)'),
        ],
        fontsize=8, framealpha=0.9
    )
    ax5.grid(color=C['grid'], lw=0.5, zorder=0)

                                                                
                                                         
                                                                
    ax6 = fig.add_subplot(gs[4:, 1])
    ax6.set_facecolor(C['panel'])
    ax6.axis('off')

    ax6.set_title('Panel 6 — Summary Metrics Dashboard',
                  fontsize=11, fontweight='bold', pad=8)

                       
    mdp = results['mode_distribution_pct']
    metrics = [
        ('SENSOR ACCURACY', None, None),
        ('MARD',                    f"{results['mard_pct']:.1f}%",
         '✓' if results['mard_pct'] < 15 else '⚠'),
        ('Clarke Zone A',           f"{results['clarke_zone_A_pct']:.1f}%",  '✓'),
        ('Clarke Zone A+B',         f"{results['clarke_zone_AB_pct']:.1f}%", '✓'),
        ('Clarke Zone D+E',
         f"{results['clarke_zone_D_pct']+results['clarke_zone_E_pct']:.1f}%",
         '✓' if results['clarke_zone_D_pct']+results['clarke_zone_E_pct'] < 5 else '⚠'),
        ('', None, None),
        ('SAFETY', None, None),
        ('Hypo events detected',    str(results['hypo_events_detected']),     ''),
        ('Mean lead time',          f"{results['mean_lead_time_min']:.0f} min", '✓'),
        ('Motion-gated samples',    f"{results['motion_suppressed_pct']:.1f}%", ''),
        ('Sweat corrections',       str(results['electro_corrections']),       ''),
        ('', None, None),
        ('POWER (wearable processor)', None, None),
        ('Mean glucose subsystem',  f"{results['mean_power_mw']:.1f} mW",
         '✓' if results['within_power_budget'] else '⚠'),
        ('Power budget',       f"{results['glucose_budget_mw']:.0f} mW",  ''),
        ('vs. fixed burst',         f"−{results['power_reduction_pct']:.1f}%", '✓'),
        ('', None, None),
        ('MODE DISTRIBUTION', None, None),
        ('Sparse',  f"{mdp.get('sparse', 0):.1f}%  (1 sample/5 min)",  ''),
        ('Alert',   f"{mdp.get('alert',  0):.1f}%  (1 sample/min)",    ''),
        ('Burst',   f"{mdp.get('burst',  0):.1f}%  (1 sample/10 sec)", ''),
    ]

    y_pos = 0.97
    line_h = 0.044
    header_color = '#1a1a2e'
    val_color    = '#333333'
    ok_color     = '#2dc653'
    warn_color   = '#f77f00'

    for label, value, status in metrics:
        if value is None:
                            
            ax6.text(0.03, y_pos, label,
                     transform=ax6.transAxes, fontsize=9.5, fontweight='bold',
                     color=header_color, va='top')
            y_pos -= line_h * 0.7
            ax6.plot([0.03, 0.97], [y_pos + 0.01, y_pos + 0.01],
                     color='#dee2e6', lw=0.8, transform=ax6.transAxes)
            y_pos -= line_h * 0.3
        elif label == '':
            y_pos -= line_h * 0.5
        else:
            ax6.text(0.05, y_pos, label,
                     transform=ax6.transAxes, fontsize=9, color=val_color, va='top')
            ax6.text(0.70, y_pos, value,
                     transform=ax6.transAxes, fontsize=9, color=val_color,
                     va='top', ha='left', fontweight='semibold')
            if status == '✓':
                ax6.text(0.94, y_pos, status,
                         transform=ax6.transAxes, fontsize=10, color=ok_color,
                         va='top', ha='center', fontweight='bold')
            elif status == '⚠':
                ax6.text(0.94, y_pos, status,
                         transform=ax6.transAxes, fontsize=10, color=warn_color,
                         va='top', ha='center', fontweight='bold')
            y_pos -= line_h

                                       
    hw_text = (
        "Hardware Target: Generic wearable processor\n"
        "CPU: Dual-core 64-bit  |  low-power ML accelerator: 4-core\n"
        "Battery: 308 mAh @ 3.8V  |  Budget: 18-hr window\n"
        "NIR: 850/940/1150/1450 nm  |  Sweat: GOx enzymatic"
    )
    ax6.text(0.5, 0.04, hw_text,
             transform=ax6.transAxes, fontsize=8, ha='center', va='bottom',
             color='#444', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#edf2fb', alpha=0.9, edgecolor='#adb5bd'))

                                                               
    out_path = os.path.join(output_dir, 'gluco_adapt_full_dashboard.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    plt.close(fig)
    print(f"Dashboard saved → {out_path}")

                                                               
    _plot_nir_wavelength_diagram(output_dir, C)

                                                               
    _plot_asymmetric_loss_surface(output_dir, C)

    return out_path


def _plot_nir_wavelength_diagram(output_dir: str, C: dict):
    """
    Show the four NIR wavelength channels and their roles.
    Generic wearable hardware context: 940 nm already present.
    Additional channels: 850, 1150, 1450 nm.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C['bg'])
    fig.suptitle(
        'NIR Multi-Wavelength Channel Architecture\n',
        fontsize=13, fontweight='bold', color='#1a1a2e'
    )

    wavelengths = [850, 940, 1150, 1450]
    roles = ['Scattering\nReference', 'PPG / Baseline\n(existing HW)', 'Glucose\nAbsorption', 'Water\nReference']
    existing = [False, True, False, False]
    colors_wl = ['#9b59b6', '#3498db', '#e74c3c', '#2ecc71']

    ax = axes[0]
    ax.set_facecolor(C['panel'])
    bars = ax.bar(
        [str(w)+'nm' for w in wavelengths],
        [1]*4,
        color=colors_wl, alpha=0.75, edgecolor='white', linewidth=1.5
    )
    for bar, role, ex in zip(bars, roles, existing):
        ax.text(bar.get_x() + bar.get_width()/2, 0.5, role,
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        if ex:
            ax.text(bar.get_x() + bar.get_width()/2, 1.05, '★ existing HW',
                    ha='center', va='bottom', fontsize=8, color='#f39c12', fontweight='bold')
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.set_title('Wavelength Roles', fontsize=11)
    ax.set_xlabel('NIR Wavelength', fontsize=10)
    ax.grid(axis='y', color=C['grid'], lw=0.4)

                                                                  
    ax2 = axes[1]
    ax2.set_facecolor(C['panel'])
    wl_range = np.linspace(700, 1600, 500)

    def gaussian(x, mu, sigma, amp):
        return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

    glucose_abs = gaussian(wl_range, 1150, 80, 0.6) + gaussian(wl_range, 1400, 60, 0.3)
    water_abs   = gaussian(wl_range, 1450, 120, 1.0) + gaussian(wl_range, 970, 60, 0.5)
    hgb_abs     = gaussian(wl_range, 800, 70, 0.8) + gaussian(wl_range, 940, 50, 0.6)

    ax2.fill_between(wl_range, glucose_abs, alpha=0.35, color='#e74c3c', label='Glucose absorption')
    ax2.fill_between(wl_range, water_abs,   alpha=0.25, color='#2ecc71', label='Water absorption')
    ax2.fill_between(wl_range, hgb_abs,     alpha=0.25, color='#3498db', label='Haemoglobin absorption')

    for wl, col in zip(wavelengths, colors_wl):
        ax2.axvline(wl, color=col, lw=2, ls='--', alpha=0.9, label=f'{wl} nm')

    ax2.set_xlim(700, 1600)
    ax2.set_ylim(0, 1.15)
    ax2.set_xlabel('Wavelength (nm)', fontsize=10)
    ax2.set_ylabel('Relative Absorption (schematic)', fontsize=10)
    ax2.set_title('Absorption Spectra & Wavelength Selection\n(schematic — not measured data)', fontsize=11)
    ax2.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(color=C['grid'], lw=0.4)

    plt.tight_layout()
    out = os.path.join(output_dir, 'gluco_adapt_nir_channels.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"NIR channel diagram saved → {out}")


def _plot_asymmetric_loss_surface(output_dir: str, C: dict):
    """
    Visualise the asymmetric clinical loss function surface:
    how the hazard score varies with predicted glucose and rate-of-change.
    This is the core full-system configuration — show it explicitly.
    """
    from hardware_config import AsymmetricClinicalLoss
    loss_fn = AsymmetricClinicalLoss()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C['bg'])
    fig.suptitle(
        'Asymmetric Clinical Safety Loss Function',
        fontsize=13, fontweight='bold', color='#1a1a2e'
    )

                                                                    
    ax = axes[0]
    ax.set_facecolor(C['panel'])

    g_range = np.linspace(40, 200, 300)
    optical_est = 100.0                         

    risk_stable  = [loss_fn.compute_risk_score(g, 3.0,  optical_est, 0.0,  0.0)  for g in g_range]
    risk_falling = [loss_fn.compute_risk_score(g, 5.0,  optical_est, -2.0, -0.5) for g in g_range]
    risk_fast    = [loss_fn.compute_risk_score(g, 8.0,  optical_est, -4.0, -1.5) for g in g_range]

    ax.fill_betweenx([0, 1], 0, 70,  alpha=0.08, color=C['burst'],  zorder=0)
    ax.fill_betweenx([0, 1], 70, 180, alpha=0.05, color=C['sparse'], zorder=0)
    ax.axvline(70,  color=C['burst'], lw=1.2, ls='--', alpha=0.6)
    ax.axvline(54,  color=C['burst'], lw=1.0, ls=':',  alpha=0.8)
    ax.axhline(0.30, color=C['alert'], lw=1.2, ls='--', alpha=0.7, label='Alert threshold')
    ax.axhline(0.70, color=C['burst'], lw=1.2, ls='--', alpha=0.7, label='Burst threshold')

    ax.plot(g_range, risk_stable,  color='#2c7bb6', lw=2.2, label='Stable (ROC=0)')
    ax.plot(g_range, risk_falling, color='#f4a261', lw=2.2, label='Falling (ROC=−2 mg/dL/min)')
    ax.plot(g_range, risk_fast,    color=C['burst'], lw=2.2, label='Fast fall (ROC=−4, accel=−1.5)')

    ax.set_xlim(40, 200)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Predicted Glucose (mg/dL)', fontsize=10)
    ax.set_ylabel('Clinical Hazard Score', fontsize=10)
    ax.set_title('Risk Score vs Predicted Glucose\n(current optical estimate = 100 mg/dL)', fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(color=C['grid'], lw=0.5)

    ax.text(56, 0.88, 'Severe\nHypo\nZone', fontsize=8, color=C['burst'],
            ha='center', va='center', alpha=0.8)
    ax.text(125, 0.92, 'BURST', fontsize=8, color=C['burst'], fontweight='bold', alpha=0.6)
    ax.text(125, 0.55, 'ALERT', fontsize=8, color='#b5830a', fontweight='bold', alpha=0.6)
    ax.text(125, 0.15, 'SPARSE', fontsize=8, color='#1a7a3c', fontweight='bold', alpha=0.6)

                                                
    ax2 = axes[1]
    ax2.set_facecolor(C['panel'])

    g_vals   = np.linspace(50, 180, 80)
    roc_vals = np.linspace(-5, 2, 80)
    G, ROC   = np.meshgrid(g_vals, roc_vals)
    Z        = np.zeros_like(G)

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            Z[i, j] = loss_fn.compute_risk_score(
                G[i, j], 4.0, G[i, j], ROC[i, j], ROC[i, j] * 0.3
            )

    im = ax2.contourf(G, ROC, Z, levels=20, cmap='RdYlGn_r', alpha=0.85)
    ax2.contour(G, ROC, Z, levels=[0.30, 0.70], colors=['orange', 'red'],
                linewidths=1.8, linestyles=['--', '--'])
    cbar = fig.colorbar(im, ax=ax2, shrink=0.85)
    cbar.set_label('Clinical Hazard Score', fontsize=9)
    cbar.ax.axhline(0.30, color='orange', lw=1.5, ls='--')
    cbar.ax.axhline(0.70, color='red',    lw=1.5, ls='--')

    ax2.axvline(70,  color='white', lw=1.5, ls='--', alpha=0.8, label='Hypo threshold (70)')
    ax2.axhline(0,   color='white', lw=1.0, ls=':',  alpha=0.5, label='Zero ROC')

    ax2.set_xlabel('Predicted Glucose (mg/dL)', fontsize=10)
    ax2.set_ylabel('Rate of Change (mg/dL/min)', fontsize=10)
    ax2.set_title('Risk Score Heatmap\nGlucose × Rate-of-Change', fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.8, loc='upper right')

                      
    ax2.text(160, -4.0, 'BURST', color='white', fontsize=9, fontweight='bold', ha='center')
    ax2.text(155,  1.5, 'SPARSE', color='white', fontsize=9, fontweight='bold', ha='center')

    plt.tight_layout()
    out = os.path.join(output_dir, 'gluco_adapt_loss_surface.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"Asymmetric loss surface saved → {out}")


if __name__ == "__main__":
    print("=" * 65)
    print("GLUCO-ADAPT — Unified Architecture Simulation")
    print("  Context layer always-on (PPG + IMU + Temp)")
    print("  Augmentation triggers on risk score (NIR + Sweat)")
    print("=" * 65)

    os.makedirs('./results', exist_ok=True)
    import shutil

                                                                
    res_a, t_a, tg_a, est_a, risk_a, modes_a, pw_a, nir_a = run_unified_pipeline(
        duration_hours=24.0, augmentation_enabled=False, verbose=True)

    res_b, t_b, tg_b, est_b, risk_b, modes_b, pw_b, nir_b = run_unified_pipeline(
        duration_hours=24.0, augmentation_enabled=True, verbose=True)

                                                                
    print("\n" + "=" * 65)
    print("COMPARISON SUMMARY")
    print("=" * 65)
    metrics_compare = [
        ("MARD",            "mard_pct",           "%",    True),
        ("Clarke A+B",      "clarke_zone_AB_pct",  "%",   False),
        ("Clarke D+E",      "clarke_zone_D_pct",   "%",   True),
        ("Hypo lead time",  "mean_lead_time_min",  " min", False),
        ("Mean power",      "mean_power_mw",       " mW", True),
        ("Power reduction", "power_reduction_pct", "%",   False),
    ]
    print(f"{'Metric':<22} {'Context Layer':>18} {'Full System':>18}")
    print("-" * 62)
    for label, key, unit, lower_better in metrics_compare:
        va = res_a.get(key, 0)
        vb = res_b.get(key, 0)
        winner = "← Context" if (va < vb) == lower_better else "← Full"
        print(f"{label:<22} {va:>12.1f}{unit}  {vb:>12.1f}{unit}   {winner}")

                                                                
    for res, fname in [(res_a, 'results_context_layer.json'),
                       (res_b, 'results_full_system.json')]:
        summary = {k: v for k, v in res.items()
                   if k not in ('hypo_event_details', 'electro_event_details')}
        with open(f'./results/{fname}', 'w') as f:
            json.dump(summary, f, indent=2)
    print("\nJSON results saved to ./results/")

                                                                
    _C = {
        'true':   '#1a1a2e', 'nir':    '#4cc9f0', 'sweat':  '#f77f00',
        'sparse': '#2dc653', 'alert':  '#f4d35e', 'burst':  '#e63946',
        'risk':   '#9b2226', 'budget': '#457b9d', 'actual': '#e9c46a',
        'bg':     '#f8f9fa', 'panel':  '#ffffff', 'grid':   '#dee2e6',
        'nir_on': '#4cc9f0',                          
    }
    mode_color_map = {'sparse': _C['sparse'], 'alert': _C['alert'], 'burst': _C['burst']}

                                                                
    print("\nGenerating visualisations...")
    generate_plots(res_a, t_a, tg_a, est_a, risk_a, modes_a, pw_a,
                   output_dir='./results')
    shutil.copy('./results/gluco_adapt_full_dashboard.png',
                './results/dashboard_context_layer_only.png')

    generate_plots(res_b, t_b, tg_b, est_b, risk_b, modes_b, pw_b,
                   output_dir='./results')
    shutil.copy('./results/gluco_adapt_full_dashboard.png',
                './results/dashboard_full_system.png')

                                                               
                      
                                                                
                                                            
                                       
                              
    print("Generating unified architecture comparison figure...")

    fig_cmp, axes = plt.subplots(4, 2, figsize=(20, 22),
                                  facecolor=_C['bg'],
                                  gridspec_kw={'height_ratios': [3, 1, 2, 2]})
    fig_cmp.suptitle(
        'GLUCO-ADAPT — Unified Architecture: Context Layer Always-On, Augmentation Risk-Triggered\n'
        'Left: Context Layer Only  |  '
        'Right: Full System — NIR + Sweat activates when risk ≥ 0.30',
        fontsize=12, fontweight='bold', y=0.995, color='#1a1a2e'
    )

    from matplotlib.colors import ListedColormap, BoundaryNorm

    for col, (res, t_h, tg, est, risks, modes, pw, nir_act, col_tag) in enumerate([
        (res_a, t_a, tg_a, est_a, risk_a, modes_a, pw_a, nir_a,
         'Context Layer Only\n(PPG + IMU + Temp )\n'
         'Context runs continuously. No augmentation.'),
        (res_b, t_b, tg_b, est_b, risk_b, modes_b, pw_b, nir_b,
         'Full System — Full Configuration\n(Context + NIR + Sweat — augmented hardware)\n'
         'NIR activates at risk ≥ 0.30  |  Sweat at risk ≥ 0.70 or every 25 min'),
    ]):
        mode_num = np.array(
            [{'sparse': 0, 'alert': 1, 'burst': 2}[str(m)] for m in modes],
            dtype=float)

                                                                 
        ax = axes[0][col]
        ax.set_facecolor(_C['panel'])
        ax.axhspan(0,  54,  alpha=0.12, color=_C['burst'],  zorder=0)
        ax.axhspan(54, 70,  alpha=0.07, color=_C['burst'],  zorder=0)
        ax.axhspan(70, 180, alpha=0.06, color=_C['sparse'], zorder=0)
        ax.axhline(70,  color=_C['burst'], lw=1.2, ls='--', alpha=0.7)
        ax.axhline(54,  color=_C['burst'], lw=0.9, ls=':',  alpha=0.8)

        cmap_m = ListedColormap([_C['sparse'], _C['alert'], _C['burst']])
        norm_m = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_m.N)
        pts  = np.array([t_h, est]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap_m, norm=norm_m,
                              linewidth=1.8, alpha=0.85, zorder=3)
        lc.set_array(mode_num[:-1])
        ax.add_collection(lc)
        ax.plot(t_h, tg, color=_C['true'], lw=2.2, alpha=0.9,
                label='True glucose', zorder=4)

                                 
        if res.get('electro_event_details'):
            eh = [e['hour'] for e in res['electro_event_details']]
            es = [e['sweat_reading'] for e in res['electro_event_details']]
            ax.scatter(eh, es, color=_C['sweat'], s=55, zorder=5, marker='D',
                       label=f'Sweat drift corrections (n={len(eh)})',
                       edgecolors='white', lw=0.8)

        ax.set_xlim(0, t_h[-1]); ax.set_ylim(30, 225)
        ax.set_title(col_tag, fontsize=9, fontweight='bold', pad=6)
        ax.set_ylabel('Glucose (mg/dL)', fontsize=9)
        ax.tick_params(labelbottom=False)
        ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
        ax.grid(axis='y', color=_C['grid'], lw=0.5)
        ax.text(0.01, 0.97,
                f"MARD: {res['mard_pct']:.1f}%\nZone A+B: {res['clarke_zone_AB_pct']:.1f}%\n"
                f"Zone D+E: {res['clarke_zone_D_pct']+res['clarke_zone_E_pct']:.1f}%",
                transform=ax.transAxes, fontsize=8.5, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

                                                                
        ax1 = axes[1][col]
        ax1.set_facecolor(_C['panel'])
        if col == 1:
                                                 
            ax1.fill_between(t_h, 0, nir_act.astype(float),
                             color=_C['nir_on'], alpha=0.7,
                             label=f"NIR active ({res.get('nir_active_pct',0):.1f}% of time)")
            ax1.fill_between(t_h, 0, (~nir_act).astype(float) * 0.15,
                             color=_C['sparse'], alpha=0.3,
                             label='Context only (PPG)')
            ax1.axhline(0.5, color='#888', lw=0.6, ls=':')
            ax1.set_yticks([0, 1])
            ax1.set_yticklabels(['Context\nOnly', 'NIR\nActive'], fontsize=7)
            ax1.set_title('Augmentation Layer — NIR Active Windows\n'
                          '(triggers at risk ≥ 0.30, gates off during motion)',
                          fontsize=9, fontweight='bold')
            ax1.legend(fontsize=8, framealpha=0.9, loc='upper right')
        else:
            ax1.fill_between(t_h, 0, 0.15, color=_C['sparse'], alpha=0.4)
            ax1.set_yticks([])
            ax1.set_title('Augmentation Layer — Not Present\n'
                          '(context layer only — PPG + IMU + Temp)',
                          fontsize=9, fontweight='bold')
            ax1.text(0.5, 0.5, 'No NIR / No Sweat\nContext layer only',
                     transform=ax1.transAxes, ha='center', va='center',
                     fontsize=10, color='#888', style='italic')
        ax1.set_xlim(0, t_h[-1]); ax1.set_ylim(-0.05, 1.1)
        ax1.tick_params(labelbottom=False)
        ax1.grid(axis='x', color=_C['grid'], lw=0.4)

                                                                 
        ax2 = axes[2][col]
        ax2.set_facecolor(_C['panel'])
        ax2.fill_between(t_h, 0, risks,
                         where=(risks < 0.30),
                         color=_C['sparse'], alpha=0.45,
                         label='Sparse zone (context only)')
        ax2.fill_between(t_h, 0, risks,
                         where=(risks >= 0.30) & (risks < 0.70),
                         color=_C['alert'], alpha=0.55,
                         label='Alert zone (NIR activates)')
        ax2.fill_between(t_h, 0, risks,
                         where=(risks >= 0.70),
                         color=_C['burst'], alpha=0.65,
                         label='Burst zone (NIR + Sweat active)')
        ax2.plot(t_h, risks, color=_C['risk'], lw=0.8, alpha=0.7)
        ax2.axhline(0.30, color=_C['alert'], lw=1.4, ls='--',
                    label='NIR trigger threshold (0.30)')
        ax2.axhline(0.70, color=_C['burst'], lw=1.4, ls='--',
                    label='Sweat trigger threshold (0.70)')
        ax2.set_xlim(0, t_h[-1]); ax2.set_ylim(0, 1.05)
        ax2.set_title('Asymmetric Clinical Hazard Score\n'
                      '(drives augmentation activation)',
                      fontsize=9, fontweight='bold')
        ax2.set_ylabel('Risk Score R', fontsize=9)
        ax2.tick_params(labelbottom=False)
        ax2.legend(fontsize=7.5, framealpha=0.9, ncol=2)
        ax2.grid(color=_C['grid'], lw=0.4)
        mdp = res['mode_distribution_pct']
        ax2.text(0.01, 0.97,
                 f"Sparse {mdp.get('sparse',0):.0f}%  "
                 f"Alert {mdp.get('alert',0):.0f}%  "
                 f"Burst {mdp.get('burst',0):.0f}%",
                 transform=ax2.transAxes, fontsize=8, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

                                                                 
        ax3 = axes[3][col]
        ax3.set_facecolor(_C['panel'])
        ax3.fill_between(t_h, 0, pw, color=_C['actual'], alpha=0.75,
                         label=f"Adaptive power (mean {res['mean_power_mw']:.1f} mW)")
        ax3.axhline(res['burst_baseline_power_mw'], color=_C['burst'],
                    lw=1.5, ls='--',
                    label=f"Fixed burst baseline ({res['burst_baseline_power_mw']:.0f} mW)")
        ax3.axhline(res['glucose_budget_mw'], color=_C['budget'],
                    lw=1.5, ls='-.',
                    label=f"Power budget ({res['glucose_budget_mw']:.0f} mW)")
        ax3.set_xlim(0, t_h[-1])
        ax3.set_ylim(0, res['burst_baseline_power_mw'] * 1.15)
        ax3.set_title('Power Consumption — Wearable Hardware',
                      fontsize=9, fontweight='bold')
        ax3.set_xlabel('Time (hours)', fontsize=9)
        ax3.set_ylabel('Power (mW)', fontsize=9)
        ax3.legend(fontsize=8, framealpha=0.9)
        ax3.grid(color=_C['grid'], lw=0.4)
        ax3.text(0.99, 0.97,
                 f"−{res['power_reduction_pct']:.1f}% vs fixed burst\n"
                 f"Hypo lead time: {res['mean_lead_time_min']:.0f} min",
                 transform=ax3.transAxes, fontsize=9, va='top', ha='right',
                 color='#1a4a1a', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    cmp_path = './results/gluco_adapt_unified_comparison.png'
    fig_cmp.savefig(cmp_path, dpi=150, bbox_inches='tight', facecolor=_C['bg'])
    plt.close(fig_cmp)
    print(f"Unified comparison figure saved → {cmp_path}")

    _plot_nir_wavelength_diagram('./results', _C)
    _plot_asymmetric_loss_surface('./results', _C)

    print("\nAll outputs in ./results/")
    print("  gluco_adapt_unified_comparison.png — unified architecture comparison (4 rows)")
    print("  dashboard_context_layer_only.png   — context layer dashboard")
    print("  dashboard_full_system.png          — full system dashboard")
    print("  gluco_adapt_nir_channels.png       — NIR wavelength architecture")
    print("  gluco_adapt_loss_surface.png       — asymmetric loss oracle surface")