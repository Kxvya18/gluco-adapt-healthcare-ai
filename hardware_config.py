
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
class WearableHardwareConstants:
                           
    CPU_DEEP_SLEEP_MW      = 0.5                                                     
    CPU_IDLE_MW            = 3.0                                        
    CPU_ACTIVE_MW          = 15.0                                          
    CPU_NEURAL_ENGINE_MW   = 25.0                                                                                                           
                                                       
    NIR_STANDBY_MW         = 0.1
    NIR_SINGLE_SAMPLE_MJ   = 0.05                                                
    NIR_CONTINUOUS_MW      = 2.0                                                                     
                                                                           
    ELECTRO_POWERDOWN_MW   = 0.0                         
    ELECTRO_WARMUP_MW      = 5.0                                             
    ELECTRO_ACTIVE_MW      = 3.5                        
    ELECTRO_WARMUP_S       = 90.0                                        
                                                                    
    IMU_MW                 = 0.8                                                                 
    BATTERY_MAH            = 308.0
    BATTERY_V              = 3.8
    BATTERY_WH             = BATTERY_MAH * BATTERY_V / 1000.0             
                       
    TARGET_HOURS           = 18.0
                             
    MAX_AVG_POWER_MW       = (BATTERY_WH * 1000.0) / TARGET_HOURS          
                                                                                                             
    GLUCOSE_POWER_BUDGET_MW = 15.0


                                                              
                    
                                                              

class ClinicalConstants:
                
    SEVERE_HYPO_THRESHOLD  = 54.0                                            
    HYPO_THRESHOLD         = 70.0                                          
    TARGET_LOW             = 70.0                             
    TARGET_HIGH            = 180.0                            
    HYPER_THRESHOLD        = 250.0                                                                                               
    SWEAT_BLOOD_LAG_MIN    = 25.0                                     
    CLARKE_ZONE_A_THRESHOLD = 0.20                                    
    CLARKE_ZONE_A_ABS       = 15.0                                                                                           
    TARGET_MARD             = 15.0

class AsymmetricClinicalLoss:

    def __init__(self):
                                                                              
        self.base_alpha     = 10.0                                                            
        self.kappa          = 1.5                                                              
        self.beta           = 2.0                                                                               
        self.lambda_unc     = 0.5                                                             
        self._norm          = 300.0

    def alpha(self, glucose: float) -> float:
       
        hypo_proximity = max(0.0, ClinicalConstants.HYPO_THRESHOLD - glucose)
        normalized_proximity = hypo_proximity / ClinicalConstants.HYPO_THRESHOLD
        return self.base_alpha * np.exp(self.kappa * normalized_proximity)

    def pointwise_loss(self, y_pred: float, y_est: float) -> float:
        
        a = self.alpha(y_est)
        underestimate_loss = a * max(0.0, y_est - y_pred) ** self.beta
        overestimate_loss  = max(0.0, y_pred - y_est)
        return underestimate_loss + overestimate_loss

    def compute_risk_score(
        self,
        predicted_glucose: float,
        pred_uncertainty: float,
        optical_glucose_est: float,                                      
        rate_of_change: float,                                                          
        acceleration: float
    ) -> float:
       
                                                   
        base_loss = self.pointwise_loss(predicted_glucose, optical_glucose_est)                                                
        accel_penalty = max(0.0, -acceleration) * 50.0                                   
        roc_penalty = max(0.0, -rate_of_change) * 2.0                                                                                    
        uncertainty_inflation = 1.0 + self.lambda_unc * (pred_uncertainty ** 2) / 100.0                                        
        severe_hypo_bonus = 0.0
        if predicted_glucose < ClinicalConstants.SEVERE_HYPO_THRESHOLD:
            severe_hypo_bonus = 500.0
        elif predicted_glucose < ClinicalConstants.HYPO_THRESHOLD:
            severe_hypo_bonus = 100.0

        raw_risk = (base_loss + accel_penalty + roc_penalty + severe_hypo_bonus)
        raw_risk *= uncertainty_inflation                  
        normalized = raw_risk / self._norm
        return float(np.clip(normalized, 0.0, 1.0))
                                  
@dataclass
class PowerModeConfig:
    sparse_nir_rate_hz:    float = 1.0 / 300.0                       
    sparse_cpu_state:      str   = "deep_sleep"
    sparse_power_mw:       float = (
        WearableHardwareConstants.CPU_DEEP_SLEEP_MW +
        WearableHardwareConstants.NIR_STANDBY_MW +
        WearableHardwareConstants.ELECTRO_POWERDOWN_MW +
        WearableHardwareConstants.IMU_MW
    )           
                                                      
    alert_nir_rate_hz:     float = 1.0 / 60.0                        
    alert_cpu_state:       str   = "idle"
    alert_power_mw:        float = (
        WearableHardwareConstants.CPU_IDLE_MW +
        WearableHardwareConstants.NIR_CONTINUOUS_MW +
        WearableHardwareConstants.ELECTRO_WARMUP_MW * 0.1 +                    
        WearableHardwareConstants.IMU_MW
    )           
                                             
    burst_nir_rate_hz:     float = 1.0 / 10.0                         
    burst_cpu_state:       str   = "neural_engine"
    burst_power_mw:        float = (
        WearableHardwareConstants.CPU_NEURAL_ENGINE_MW +
        WearableHardwareConstants.NIR_CONTINUOUS_MW +
        WearableHardwareConstants.ELECTRO_ACTIVE_MW +
        WearableHardwareConstants.IMU_MW
    )            
                     
    sparse_to_alert_threshold:  float = 0.30
    alert_to_burst_threshold:   float = 0.70
    burst_to_alert_threshold:   float = 0.40                                                 
    alert_to_sparse_threshold:  float = 0.20                                                    


class AdaptiveSamplingPolicy:

    def __init__(self, config: PowerModeConfig = None):
        self.config = config or PowerModeConfig()
        self._current_mode = "sparse"
        self._mode_duration = 0                           
        self._min_mode_duration = 3                                                

    def update(self, risk_score: float) -> Tuple[str, float, float]:
    
        prev_mode = self._current_mode
        self._mode_duration += 1

                                                             
        if risk_score >= self.config.alert_to_burst_threshold:
            self._current_mode = "burst"
        elif risk_score >= self.config.sparse_to_alert_threshold:
            if self._current_mode == "sparse":
                self._current_mode = "alert"
                                                                                                                                       
        elif self._mode_duration >= self._min_mode_duration:
            if self._current_mode == "burst" and risk_score < self.config.burst_to_alert_threshold:
                self._current_mode = "alert"
                self._mode_duration = 0
            elif self._current_mode == "alert" and risk_score < self.config.alert_to_sparse_threshold:
                self._current_mode = "sparse"
                self._mode_duration = 0
                            
        if self._current_mode != prev_mode:
            self._mode_duration = 0

                                
        if self._current_mode == "burst":
            return ("burst",
                    self.config.burst_nir_rate_hz,
                    self.config.burst_power_mw)
        elif self._current_mode == "alert":
            return ("alert",
                    self.config.alert_nir_rate_hz,
                    self.config.alert_power_mw)
        else:
            return ("sparse",
                    self.config.sparse_nir_rate_hz,
                    self.config.sparse_power_mw)

    @property
    def current_mode(self) -> str:
        return self._current_mode
