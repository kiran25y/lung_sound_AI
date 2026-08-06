# Data layout

Nothing here is committed — point the pipeline at your own hospital recordings.

## 0. Generating the manifest automatically (flat folders, labels in filenames)

If your wavs are in flat folders with the label in the filename (e.g.
`seoul_p001_wheeze_steth.wav`), don't write the CSV by hand — generate it:

```bash
# preview how filenames are parsed (writes nothing):
python tools/build_manifest.py --root /path/to/audio --patient-regex "(p\d+)" --dry-run

# write it:
python tools/build_manifest.py --root /path/to/audio --patient-regex "(p\d+)" --out data/manifest.csv
```

It infers `label`, `hospital`, `device`, `patient_id` from each filename (and parent
folder), prints a summary, and lists anything it couldn't classify so you can adjust
`--label-map` / `--hospital-map` / `--device-map` / `--patient-regex` and re-run.
**Check the "patients" count** in the summary — clips from the same patient must share
a `patient_id`, or the patient-level split is defeated.

## 1. Clip manifest — `data/manifest.csv`
One row per audio clip. Columns:

| column | meaning | example |
|---|---|---|
| `filepath` | absolute/relative path to a wav | `/data/seoul/p001_a1.wav` |
| `patient_id` | unique per patient (drives the split) | `seoul_p001` |
| `label` | `normal` \| `crackle` \| `wheeze` | `crackle` |
| `hospital` | `seoul` \| `sungbook` | `seoul` |
| `device` | `stethoscope` \| `smartphone` | `stethoscope` |
| `split` | optional manual override, else auto | `` |

The loader assigns splits **by `patient_id`** so no patient leaks across train/test:
- `seoul` + `stethoscope`  -> train / internal_val / internal_test
- `sungbook` + `stethoscope` -> **external_test** (never trained on)
- `*` + `smartphone`         -> smartphone_test (+ optional small train slice)

## 2. Noise
```
data/noise/
  musan_noise/        # MUSAN, NOISE category only (drop the music/ and speech/ folders)
  self_recorded/      # your own PCB recordings of ward/clinic/home ambience
  desed_synthetic/    # DESED_synthetic soundscapes + tsv labels (Phase-2 detector only)
```
