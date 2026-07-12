# Gluco-Adapt: Healthcare AI Simulation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Healthcare AI](https://img.shields.io/badge/Healthcare%20AI-Glucose%20Monitoring-green)
![Wearable AI](https://img.shields.io/badge/Wearable%20AI-Adaptive%20Sensing-orange)
![Simulation](https://img.shields.io/badge/Mode-Simulation-lightgrey)

Gluco-Adapt is an initial-stage healthcare AI architecture simulation for adaptive, non-invasive glucose monitoring on generic wearable hardware. It should be understood as a conceptual and computational prototype rather than a production-grade or clinically validated system. The project models a multimodal sensing pipeline with near-infrared optical sensing, sweat-based correction, IMU motion gating, a PPG context layer, quantized LSTM-style prediction, uncertainty-aware evaluation, and risk-weighted adaptive sampling.

The repository is intentionally curated as a compact GitHub-ready release centered on the runnable simulation, the generated evaluation artifacts, and the architectural logic that underpins the system. It is meant to communicate the design of the platform, the interaction between its sensing layers, and the behaviour of its risk-aware control policy.

## Highlights

- Generic wearable hardware configuration and power-mode modelling
- NIR optical sensor simulation with motion-aware reliability behaviour
- Electrochemical sweat-sensor correction with physiological lag handling
- PPG context layer for low-power trend and contextual estimation
- Quantized LSTM-style glucose trajectory predictor
- Asymmetric clinical risk loss for hypoglycaemia-aware decision making
- Adaptive sampling policy governed by predicted risk and uncertainty
- Evaluation outputs for context-only and full-system simulation settings

## Architectural Positioning

This repository represents the early architectural layer of the Gluco-Adapt concept. It captures the scaffolding, signal flow, and decision logic of the system, but it should not be interpreted as a final, end-to-end medical device implementation. In practical terms, the code demonstrates how the sensing modalities, prediction stack, and power-management policy can be orchestrated inside a single adaptive pipeline.

The design is useful for:

- illustrating an integrated wearable sensing architecture
- demonstrating how multimodal signals can be fused into a single control loop
- exploring the trade-off between clinical sensitivity and battery efficiency
- providing a reproducible baseline for future development and refinement

## System Flow

```text
Synthetic physiology
        |
        v
NIR + sweat + PPG context + IMU motion signals
        |
        v
Quantized glucose trajectory prediction
        |
        v
Clinical risk scoring + uncertainty evaluation
        |
        v
Adaptive sampling and power-mode decisions
        |
        v
JSON metrics + result figures
```

The pipeline is deliberately sequential: physiological trajectories are synthesised first, sensor channels are then layered on top, prediction is performed over a rolling history window, and the resulting risk score governs both sensing intensity and power-state transitions.

## Repository Structure

```text
.
├── hardware_config.py        # Hardware constants, clinical thresholds, risk loss, power modes
├── sensors.py                # NIR, sweat, IMU, and PPG context sensor models
├── predictor.py              # Quantized predictor and physiological glucose generator
├── 06_simulation.py          # Main simulation and evaluation runner
└── results/
        ├── results_context_layer.json
        ├── results_full_system.json
        ├── gluco_adapt_loss_surface.png
        ├── gluco_adapt_nir_channels.png
        └── gluco_adapt_unified_comparison.png
```

## Quick Start

Install the required Python packages:

```bash
pip install numpy matplotlib
```

Run the simulation:

```bash
python 06_simulation.py
```

The script writes metrics and figures to `results/`.

## Output Interpretation

The generated artifacts document two complementary operating regimes:

- the context-layer-only mode, which serves as a lower-complexity baseline
- the full-system mode, which enables NIR and sweat augmentation under the adaptive policy

Comparing these outputs gives a clearer picture of how the architecture changes when additional sensing and correction layers are activated. The result files are therefore not incidental logs; they are the empirical record of the design choices embodied by the simulation.

## Key Outputs

| Output | Description |
|---|---|
| `results/results_context_layer.json` | Metrics for the context-layer-only simulation setting |
| `results/results_full_system.json` | Metrics for the full multimodal Gluco-Adapt system |
| `results/gluco_adapt_loss_surface.png` | Clinical risk/loss surface visualization |
| `results/gluco_adapt_nir_channels.png` | NIR channel behavior visualization |
| `results/gluco_adapt_unified_comparison.png` | Unified comparison of system behavior and outcomes |

## Main Components

| File | Role |
|---|---|
| `hardware_config.py` | Defines wearable power constants, clinical thresholds, asymmetric risk loss, and adaptive power modes |
| `sensors.py` | Simulates optical, sweat, motion, and PPG-context sensing layers |
| `predictor.py` | Implements quantized prediction logic and synthetic physiological glucose generation |
| `06_simulation.py` | Runs the end-to-end simulation, computes metrics, and generates result artifacts |

## Scope

This repository intentionally preserves an early architectural and demonstrative version of Gluco-Adapt. It is not the complete or final production system. Draft materials, redundant design copies, unrelated dashboard exports, and extraneous artefacts are excluded so that the repository remains focused on the core runnable simulation and its principal outputs.
