# VOC-Predict Lab Session Protocol

Standard operating procedure for collecting ZnO MOX sensor data that feeds the
`time`/`resistance` pipeline. Purpose: keep lab recordings consistent and
debuggable so a failure a month from now is traceable to sensor drift or
algorithm, not a missing metadata field.

---

## 1. Pre-session checklist

- [ ] Sensor ID assigned (e.g., `ZnO-01`) and sensor age in hours noted
- [ ] Sensor powered and warmed up; heater at constant voltage/PWM (record value)
- [ ] Lab ambient temperature and relative humidity measured and recorded
- [ ] 10-minute clean-air purge running before the first pulse
- [ ] Data filename decided: `YYYY-MM-DD-<session>_<sensor>.csv`
- [ ] Metadata sidecar template ready: `<recording>.meta.json`

---

## 2. Per-session execution (mandatory)

Every lab session must run, in order:

1. **10-minute clean-air purge** — establishes a fresh baseline `R0`.
2. **Reference pulse** — single pulse of a known gas (default: **50 ppm ethanol**).
   Record its features in the sidecar; this is your drift canary.
3. **Target gas pulses** — unknown/validation gases, each under the same heater
   and ambient conditions as the reference.

For each pulse save:
- Raw `time` (seconds), `resistance` (ohms) CSV
- `<recording>.meta.json` sidecar (see §3)

Do not change heater voltage or ambient conditions mid-session.

---

## 3. Metadata sidecar (`<recording>.meta.json`)

Save one JSON file per recording, next to the raw data.

```json
{
  "session_id": "2026-11-12-01",
  "sensor_id": "ZnO-01",
  "sensor_age_hours": 48,
  "ambient_temp_c": 23.5,
  "ambient_humidity_rh": 45.0,
  "heater_voltage_v": 5.0,
  "heater_pwm_duty": 100,
  "baseline_r0_ohms": 9800000.0,
  "baseline_start_timestamp": "2026-11-12T09:00:00Z",
  "gas": "ethanol",
  "concentration_ppm": 50,
  "reference_pulse": true,
  "notes": "first lab session, fresh sensor"
}
```

Fields:

| Field | Purpose |
|---|---|
| `session_id` | Groups all recordings from one lab session |
| `sensor_id` | Physical sensor identity (trace aging per unit) |
| `sensor_age_hours` | Cumulative powered hours — drift predictor |
| `ambient_temp_c` / `ambient_humidity_rh` | MOX is sensitive to both |
| `heater_voltage_v` / `heater_pwm_duty` | Heater program determines selectivity |
| `baseline_r0_ohms` | Fresh R0 from the purge; drift baseline |
| `baseline_start_timestamp` | When purge started |
| `gas` / `concentration_ppm` | Ground truth for training/validation |
| `reference_pulse` | `true` only for the known-gas canary pulse |

No model code reads this yet; it is diagnostic-only. Record it anyway.

---

## 4. Drift canary (reference pulse)

After each session, compare the reference pulse's extracted features
(`peak_response`, `response_time`, `rise_rate`, `tau_ad_1`, `tau_ad_2`) against the
previous session's reference.

- If features drift **>20%** from the established baseline, treat the sensor as
  degraded: recalibrate or retire before collecting more data.
- Log the comparison in the session notes; do not silently keep collecting.

This is the only early-warning system for sensor aging until a drift-compensation
model is built.

---

## 5. Data handoff to pipeline

Raw CSVs go into `data sets/` (or a dated subfolder). Convert each to the pipeline
contract:

```python
# minimal conversion sketch
import pandas as pd
df = pd.read_csv("data sets/2026-11-12-01_ZnO-01.csv")  # columns: time, resistance
from pipeline import extract_features
f = extract_features(df)
```

Keep the raw CSV and `.meta.json` together; never store features without source.

---

## 6. Post-session

- [ ] All CSVs have matching `.meta.json` sidecars
- [ ] Reference pulse features logged and compared to prior session
- [ ] Sensor age hours updated for next session
- [ ] Raw data backed up (features are derivable; raw is not)

---

## Related

`ARCHITECTURE.md` — §13 metadata sidecar, §14 LPC1769 compute limits, §15 drift
calibration. `pipeline.extract_features` is the single consumer of the
`time`/`resistance` contract.
