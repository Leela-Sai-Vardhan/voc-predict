---
license: apache-2.0
task_categories:
  - tabular-classification
tags:
  - gas-sensor
  - mox-sensor
  - bme688
  - synthetic
  - chemistry
  - iot
  - environmental-monitoring
language:
  - en
pretty_name: Synthetic MOX Gas Sensor Temperature Profiles
size_categories:
  - 1K<n<10K
extra_gated_prompt: >-
  This dataset is in validation phase.
  Access is granted to verified researchers and organizations.
  Please describe your intended use case.
extra_gated_fields:
  Full name: text
  Organization: text
  Intended use: text
  I agree to use this data for research purposes only:
    type: checkbox
---

# Synthetic MOX Gas Sensor Temperature Profiles

Synthetic dataset for training gas classifiers on MOX sensor arrays (BME688-type). I generated this because real sensor data is either proprietary or tiny. The physics model is simplified but the patterns are realistic enough to train a decent baseline.

## What's in here

5000 sensor readings across 200 different chemical compounds. Each reading is a 10-step temperature cycle (ramp up to 400°C then back down) with the corresponding resistance profile — basically what you'd get from a BME688 or similar hotplate MOX sensor.

The compounds span 10 categories:
- alcohols — ethanol, methanol, isopropanol, etc.
- aldehydes — formaldehyde, acetaldehyde, benzaldehyde, etc.
- ketones — acetone, MEK, cyclohexanone, etc.
- hydrocarbons — methane, propane, toluene, benzene, etc.
- VOCs — ethyl acetate, THF, acetic acid, etc.
- food_aromas — vanillin, eugenol, linalool, menthol, etc.
- combustion_gases — CO, CO2, NO2, H2S, NH3, etc.
- nature_environment — pinene, geosmin, isoprene, plant volatiles
- industrial_solvents — toluene, xylene, DCM, NMP, etc.
- agricultural — ammonia, amines, organic acids from livestock/compost

Each compound has 25 variations with different concentrations, humidity levels, and ambient temperatures.

## Schema

| Field | Type | Description |
|-------|------|-------------|
| `compound` | string | Common name of the chemical compound |
| `formula` | string | Molecular formula |
| `cas_number` | string | CAS registry number |
| `category` | string | One of 10 compound categories |
| `hazard_level` | string | low / moderate / high |
| `concentration_ppm` | float | Gas concentration in parts per million |
| `temperature_profile` | list[int] | 10-step heater temperature cycle in °C |
| `resistance_profile` | list[int] | Corresponding sensor resistance in Ohms |
| `humidity_rh` | float | Relative humidity (%) during measurment |
| `ambient_temp_c` | float | Ambient temperature in °C |

## Physics model

The resistance profiles follow a simplified but physically-grounded MOX sensor model:

1. **Temperature dependence**: Base resistance decreases exponentially with heater temperature (semiconductor behavior)
2. **Gas interaction**: Reducing gases (most organics, CO, H2S) decrease resistance on the SnO2 surface. Oxidizing gases (NO2, O3, Cl2) increase it.
3. **Concentration**: Power-law relationship — `R ∝ C^(-α)` for reducing gases, `R ∝ C^(+α)` for oxidizing
4. **Sensitivity envelope**: Gaussian peak around each compound's characteristic temperature. This is what makes temperature cycling useful for discrimination.
5. **Environmental factors**: Humidity slightly lowers baseline resistance. Ambient temperature shifts the sensitivity peak.
6. **Noise**: 2-5% Gaussian noise on every reading

This is obviously not a full electrochemical simulation. Real sensors have drift, aging, cross-sensitivity between gases, and nonlinear humidity effects that aren't captured here. But the core pattern — that different gases produce different resistance "fingerprints" across a temperature cycle — is realistic.

## Usage

```python
from datasets import load_dataset
import pandas as pd

# From HuggingFace
ds = load_dataset("Laborator/gas-sensor-profiles")

# Or just load the parquet
df = pd.read_parquet("data/gas_sensor_profiles.parquet")

# Quick classifier example
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X = np.array(df["resistance_profile"].tolist())
y = df["category"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(f"Category accuracy: {clf.score(X_test, y_test):.3f}")
```

## Generation

Dataset was generated with `generate_dataset.py` included in this repo. You can re-run it to get new random variations (change the seed). Takes about 2 seconds.

## Safety note

All compounds in this dataset are commonly encountered household, industrial, food-grade, or environmental chemicals. No chemical weapons, nerve agents, explosives precursors, or controlled substances are included.

## License

Apache 2.0
