
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from hardware_config import ClinicalConstants, WearableHardwareConstants

class PPGContextLayer:
                                                                          
    PPG_BASELINE_MW   = 0.5                                              
    PPG_GLUCOSE_MW    = 1.2                                                          
    TEMP_SENSOR_MW    = 0.1                                 
    TOTAL_SENSING_MW  = PPG_BASELINE_MW + PPG_GLUCOSE_MW + TEMP_SENSOR_MW                                         
    feature_noise_std = 22.0                                              

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)                                                                          
        self.ptt_coeff   = -0.8
        self.pi_coeff    =  0.3
        self.temp_coeff  =  0.4                                                                
        self.drift_rate_per_hour = 3.0

    def get_context_signal(
        self,
        true_glucose: np.ndarray,
        imu_accel: np.ndarray,
        t_hours: np.ndarray
    ) -> np.ndarray:
    
        n = len(true_glucose)

        ptt_noise  = self.rng.normal(0, 8.0, n)
        ptt_signal = true_glucose + ptt_noise * self.ptt_coeff * 2

        pi_noise   = self.rng.normal(0, 5.0, n)
        pi_signal  = true_glucose + pi_noise * self.pi_coeff * 3

        temp_noise = self.rng.normal(0, 3.0, n)
        temp_signal= true_glucose + temp_noise * self.temp_coeff * 4

        combined = 0.5 * ptt_signal + 0.3 * pi_signal + 0.2 * temp_signal
        combined += self.rng.normal(0, self.feature_noise_std, n)
        combined += imu_accel * 8.0                             
        combined += self.drift_rate_per_hour * t_hours                     

        return np.clip(combined, 30.0, 400.0)                                               

class IMUMotionDetector:
                                          
    MOTION_THRESHOLD_G = 0.3

    def __init__(self, window_samples: int = 10):
        self.window_samples = window_samples
        self._accel_buffer = np.zeros(window_samples)
        self._buffer_idx = 0
        self._artifact_uncertainty = 0.0                                                                                                
        self.uncertainty_decay = 0.05                         

    def update(self, accel_x: float, accel_y: float, accel_z: float) -> Tuple[bool, float]:
                                                                    
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2) - 1.0                                        
        self._accel_buffer[self._buffer_idx % self.window_samples] = abs(accel_mag)
        self._buffer_idx += 1          
        rms_accel = np.sqrt(np.mean(self._accel_buffer**2))

        motion_detected = rms_accel > self.MOTION_THRESHOLD_G

        if motion_detected:
                                                                  
            motion_intensity = min(rms_accel / self.MOTION_THRESHOLD_G, 5.0)
            self._artifact_uncertainty = min(
                self._artifact_uncertainty + motion_intensity * 10.0,
                50.0                                          
            )
        else:
                                                  
            self._artifact_uncertainty = max(
                0.0,
                self._artifact_uncertainty * (1.0 - self.uncertainty_decay)
            )

        return motion_detected, self._artifact_uncertainty

    def simulate_imu(self, t_hours: np.ndarray, exercise_periods: list) -> np.ndarray:
    
        np.random.seed(123)                   
        n = len(t_hours)

                                                
        baseline_accel = 0.05 + 0.02 * np.abs(np.random.randn(n))

        accel = baseline_accel.copy()

        for start_hr, duration_hr, intensity in exercise_periods:
            mask = (t_hours >= start_hr) & (t_hours <= start_hr + duration_hr)
            n_motion = np.sum(mask)
            if n_motion > 0:
                                                                               
                t_rel = t_hours[mask] - start_hr
                exercise_accel = (
                    intensity * 1.5 * np.abs(np.sin(2 * np.pi * 2.0 * t_rel * 3600))
                    + 0.1 * np.abs(np.random.randn(n_motion))
                )
                accel[mask] += exercise_accel

        return accel
                                            

class NIROpticalSensor:

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)                                          
        self.baseline_noise_std   = 5.0                                                       
        self.scatter_noise_std    = 3.0                                      
        self.temp_drift_coeff     = 2.5                                                                                                                                                             
        self.melanin_bias         = 0.0                                                                                                             
        self.isf_lag_min          = 10.0                                                                  
        self.drift_rate_per_hour  = 1.0

    def _apply_noise(
        self,
        true_glucose: np.ndarray,
        motion_artifacts: np.ndarray,
        temperature_deviation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        n = len(true_glucose)                    
        baseline = self.rng.normal(0, self.baseline_noise_std, n)                                         
        scatter = self.rng.normal(0, self.scatter_noise_std, n)         
        if temperature_deviation is not None:
            temp_effect = self.temp_drift_coeff * temperature_deviation
        else:
            temp_effect = np.zeros(n)                                                    
        motion_effect = motion_artifacts * 3.0                                                                                                  
        t_hours = np.linspace(0, len(true_glucose) / 3600.0, n)
        drift = self.drift_rate_per_hour * t_hours

        noisy = true_glucose + baseline + scatter + temp_effect + motion_effect + drift
        return np.clip(noisy, 30.0, 400.0)

    def simulate(
        self,
        true_glucose: np.ndarray,
        motion_artifacts: np.ndarray,
        temperature_deviation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self._apply_noise(true_glucose, motion_artifacts, temperature_deviation)

    def get_reading_with_uncertainty(
        self,
        raw_reading: float,
        motion_uncertainty: float,
        drift_uncertainty: float
    ) -> Tuple[float, float]:
        sensor_uncertainty = self.baseline_noise_std
        total_uncertainty = np.sqrt(
            sensor_uncertainty**2 +
            motion_uncertainty**2 +
            drift_uncertainty**2
        )
        return raw_reading, total_uncertainty
class ElectrochemicalSweatSensor:

    def __init__(self, seed: int = 99):
        self.rng = np.random.RandomState(seed)                                                             
        self.noise_fraction   = 0.12                        
        self.baseline_noise   = 3.0                                                                                        
        self.lag_min          = ClinicalConstants.SWEAT_BLOOD_LAG_MIN                        
        self.warmup_s         = WearableHardwareConstants.ELECTRO_WARMUP_S                                           
        self._nir_drift_history = []
        self._last_correction_t = None

    def should_activate(self, current_t_hours: float, mode: str) -> bool:
        if mode == "sparse":
            return False                                    

        if self._last_correction_t is None:
            self._last_correction_t = current_t_hours
            return True                             

        elapsed_hours = current_t_hours - self._last_correction_t
        lag_hours = self.lag_min / 60.0

        return elapsed_hours >= lag_hours

    def get_correction(
        self,
        true_glucose: float,
        current_nir_estimate: float,
        t_hours: float
    ) -> Tuple[float, float]:                                                            
        sweat_reading = (
            true_glucose
            + self.rng.normal(0, self.baseline_noise)
            + true_glucose * self.rng.normal(0, self.noise_fraction)
        )
        sweat_reading = np.clip(sweat_reading, 30.0, 400.0)                                                               
                                                                   
        drift_correction = current_nir_estimate - sweat_reading

        self._last_correction_t = t_hours

        return float(sweat_reading), float(drift_correction)