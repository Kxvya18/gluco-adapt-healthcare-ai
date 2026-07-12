

import numpy as np
from typing import Tuple, List
from hardware_config import ClinicalConstants
class FixedPointQuantizer:

    def __init__(self, scale: float = None, n_bits: int = 8):
        self.n_bits = n_bits
        self.qmin = -(2 ** (n_bits - 1) - 1)        
        self.qmax =  (2 ** (n_bits - 1) - 1)        
        self.scale = scale

    def calibrate(self, tensor: np.ndarray) -> 'FixedPointQuantizer':

        max_abs = np.max(np.abs(tensor))
        self.scale = max_abs / self.qmax if max_abs > 0 else 1.0
        return self

    def quantize(self, tensor: np.ndarray) -> np.ndarray:
        
        if self.scale is None:
            self.calibrate(tensor)
        q = np.round(tensor / self.scale).astype(np.int8)
        return np.clip(q, self.qmin, self.qmax)

    def dequantize(self, q_tensor: np.ndarray) -> np.ndarray:
        
        return q_tensor.astype(np.float32) * self.scale

    def quantize_dequantize(self, tensor: np.ndarray) -> np.ndarray:
       
        return self.dequantize(self.quantize(tensor))                                                 

class QuantizedLSTMCell:

    def __init__(self, input_size: int, hidden_size: int, seed: int = 7):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        rng = np.random.RandomState(seed)                                     
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        combined_size = input_size + hidden_size
                                                
        W_raw = rng.normal(0, scale, (4 * hidden_size, combined_size)).astype(np.float32)
        b_raw = np.zeros(4 * hidden_size, dtype=np.float32)                     
        self.W_quantizer = FixedPointQuantizer()
        self.W_q = self.W_quantizer.calibrate(W_raw).quantize(W_raw)
        self.b = b_raw                                                                                 
        self.h = np.zeros(hidden_size, dtype=np.float32)
        self.c = np.zeros(hidden_size, dtype=np.float32)

    def reset_state(self):
       
        self.h = np.zeros(self.hidden_size, dtype=np.float32)
        self.c = np.zeros(self.hidden_size, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
                                           
        combined = np.concatenate([self.h, x]).astype(np.float32)                                  
        W_float = self.W_quantizer.dequantize(self.W_q)                              
        z = W_float @ combined + self.b                                         
        hs = self.hidden_size
        f = self._sigmoid(z[0*hs : 1*hs])                
        i = self._sigmoid(z[1*hs : 2*hs])               
        g = np.tanh(z[2*hs : 3*hs])                    
        o = self._sigmoid(z[3*hs : 4*hs])                                       
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)

        return self.h.copy()

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
class QuantizedLSTMPredictor:
   
    def __init__(self, hidden_size: int = 32, input_window_min: int = 30, seed: int = 42):
        self.hidden_size      = hidden_size
        self.input_window_min = input_window_min
        self.input_size       = 3                                                      
               
        self.lstm = QuantizedLSTMCell(self.input_size, hidden_size, seed=seed)
                                    
        rng = np.random.RandomState(seed + 1)
        self.W_out = rng.normal(0, 0.1, (1, hidden_size)).astype(np.float32)
        self.b_out = np.array([100.0], dtype=np.float32)                                   
                   
        self.out_quantizer = FixedPointQuantizer()
        self.W_out_q = self.out_quantizer.calibrate(self.W_out).quantize(self.W_out)
                                                           
        self.glucose_mean = 120.0
        self.glucose_std  = 40.0
        self.roc_mean     = 0.0
        self.roc_std      = 30.0
                                                      
        self.n_mc_samples = 5                                               
        self.dropout_rate = 0.1

    def _normalize_input(self, glucose: float, roc: float, t_hour: float) -> np.ndarray:
        
        g_norm = (glucose - self.glucose_mean) / self.glucose_std
        roc_norm = (roc - self.roc_mean) / self.roc_std
        tod_sin = np.sin(2 * np.pi * t_hour / 24.0)                      
        return np.array([g_norm, roc_norm, tod_sin], dtype=np.float32)

    def _single_forward(
        self,
        history_glucose: np.ndarray,
        history_roc: np.ndarray,
        t_hours: np.ndarray,
        dropout_seed: int
    ) -> float:
        self.lstm.reset_state()
        rng = np.random.RandomState(dropout_seed)

        for idx in range(len(history_glucose)):
            x = self._normalize_input(
                history_glucose[idx],
                history_roc[idx] if idx < len(history_roc) else 0.0,
                t_hours[idx] if idx < len(t_hours) else 0.0
            )
                                                            
            dropout_mask = (rng.random(self.input_size) > self.dropout_rate).astype(np.float32)
            x = x * dropout_mask

            h = self.lstm.step(x)
                             
        W_out_float = self.out_quantizer.dequantize(self.W_out_q)
        pred_norm = float((W_out_float @ h + self.b_out).ravel()[0])
             
        pred_glucose = pred_norm * self.glucose_std + self.glucose_mean
        return float(np.clip(pred_glucose, 30.0, 400.0))

    def predict(
        self,
        history_glucose: List[float],
        history_roc: List[float],
        t_hours_history: List[float],
        horizon_steps: int = 3
    ) -> Tuple[float, float]:
    
        if len(history_glucose) < 2:
                                                                                 
            current = history_glucose[-1] if history_glucose else 100.0
            return current, 20.0

                              
        window = self.input_window_min
        h_glucose = np.array(history_glucose[-window:], dtype=np.float32)
        h_roc = np.array(history_roc[-window:] if len(history_roc) >= len(h_glucose)
                         else [0.0]*len(h_glucose), dtype=np.float32)
        h_t = np.array(t_hours_history[-window:] if len(t_hours_history) >= len(h_glucose)
                       else [0.0]*len(h_glucose), dtype=np.float32)

                            
        mc_predictions = []
        for mc_seed in range(self.n_mc_samples):
            pred = self._single_forward(h_glucose, h_roc, h_t, dropout_seed=mc_seed * 17 + 3)
            mc_predictions.append(pred)

        predicted_glucose = float(np.mean(mc_predictions))
        uncertainty = float(np.std(mc_predictions))

                                                                                     
        if len(history_roc) > 0 and history_roc[-1] != 0:
                                                                              
                                                       
            linear_proj = predicted_glucose + horizon_steps * 1.0 * history_roc[-1] / 60.0
                                                                             
            predicted_glucose = 0.7 * predicted_glucose + 0.3 * linear_proj

        return float(np.clip(predicted_glucose, 30.0, 400.0)), max(uncertainty, 1.0)

class PhysiologicalGlucoseGenerator:

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        duration_hours: float = 24.0,
        sampling_rate_hz: float = 1.0 / 60.0,                
        include_hypo: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
    
        n = int(duration_hours * 3600 * sampling_rate_hz)
        t = np.linspace(0, duration_hours, n)

        basal = 100 + 15 * np.sin(2 * np.pi * (t - 6) / 24)                                                
        basal += 5 * np.sin(2 * np.pi * t / 1.5)
        glucose = basal.copy()                                                
        meals = [(7, 50), (12, 70), (18, 60), (21, 30)]                   
        for meal_t, carbs in meals:
            idx = np.argmin(np.abs(t - meal_t))
            end = min(idx + int(2 * 3600 * sampling_rate_hz), n)
            t_rel = t[idx:end] - meal_t
                                                                
            meal_response = carbs * 0.5 * (
                (1 - np.exp(-t_rel / 0.5)) - (1 - np.exp(-t_rel / 2.5))
            )
            glucose[idx:end] += meal_response
            
        if include_hypo:
            hypo_times = [8.5, 16.0, 22.5]
            for ht in hypo_times:
                if ht > duration_hours:
                    continue
                idx = np.argmin(np.abs(t - ht))
                end = min(idx + int(1.5 * 3600 * sampling_rate_hz), n)
                t_rel = t[idx:end] - ht

                                                                        
                drop = -30 * (1 - np.exp(-t_rel / 0.25))
                recovery = 30 * (1 - np.exp(-(t_rel - 0.5) / 1.0)) * (t_rel > 0.5)
                glucose[idx:end] += drop + recovery

                          
        exercises = [(10.0, 1.0, 0.7), (15.0, 0.5, 0.9)]
        for start, dur, intensity in exercises:
            if start > duration_hours:
                continue
            mask = (t >= start) & (t <= start + dur)
            t_rel = t[mask] - start
                                                                
            stress = intensity * 15 * np.exp(-t_rel / 0.17)
            uptake = -intensity * 20 * (1 - np.exp(-t_rel / 1.0))
            glucose[mask] += stress + uptake

                              
        glucose = np.clip(glucose, 30.0, 400.0)

        return t, glucose

class SimulationPredictor:
    def __init__(self, horizon_min: int = 15, window_min: int = 30):
        self.horizon_min = horizon_min
        self.window_min  = window_min
        self._residuals  = []                                                  

    def predict(
        self,
        history_glucose: List[float],
        history_roc: List[float],
        t_hours_history: List[float],
        horizon_steps: int = 3
    ) -> Tuple[float, float]:
    
        if len(history_glucose) < 3:
            current = history_glucose[-1] if history_glucose else 100.0
            return current, 15.0                                     

        n = len(history_glucose)
        current = history_glucose[-1]

                                                                            
        weights = np.exp(np.linspace(-1, 0, min(n, self.window_min)))
        recent = np.array(history_glucose[-len(weights):])
        weighted_mean = np.average(recent, weights=weights[-len(recent):])

                                            
        lookback = min(5, n)
        if lookback >= 2:
            recent_vals = np.array(history_glucose[-lookback:])
            x = np.arange(lookback)
            slope = np.polyfit(x, recent_vals, 1)[0]                                 
        else:
            slope = 0.0

                         
        projection = current + slope * horizon_steps

                                                                          
        mean_reversion_strength = 0.15
        predicted = (
            (1 - mean_reversion_strength) * projection +
            mean_reversion_strength * weighted_mean
        )

                                                            
        if len(history_glucose) >= 5:
            recent_std = np.std(history_glucose[-10:]) if len(history_glucose) >= 10 else np.std(history_glucose)
            trend_uncertainty = abs(slope) * horizon_steps * 0.5
            uncertainty = np.sqrt(recent_std**2 + trend_uncertainty**2)
            uncertainty = max(uncertainty, 2.0)                               
        else:
            uncertainty = 10.0

        return float(np.clip(predicted, 30.0, 400.0)), float(uncertainty)
