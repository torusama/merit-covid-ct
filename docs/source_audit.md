# Source audit

Audit baseline: `aec40ed` on `main`, after `git pull --ff-only` (already up to date).
Read the actual source, all analysis documentation, CSV rows, and summary including
per-case trajectory lengths before implementation. Two untracked comparison scripts
were present before work and are retained without edits. No CT or mask is modified.

| File | Purpose and existing functions | Input | Output | Dependencies | Reusable logic | Findings | Placement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| README.md | Research plan, downloads, protocol | Human reader | Instructions | None | Dataset roles | Confirmed: promotes speculative spacing/alignment/gating extensions; no implementation supports those claims | Root; rewritten to describe current foundation |
| requirements.txt | Tool dependencies | pip | Installed tools | NumPy, nibabel, Pillow, OpenCV, matplotlib, HF Hub, requests | Existing tool environment | Missing: package metadata, test and optional training dependencies | Root; retained with core additions |
| .gitignore | Exclude datasets, archives, outputs, models | Git paths | Ignore rules | Git | Existing exclusions | Missing: pytest/build metadata exclusions | Root |
| tools/download_datasets.py | Dispatch two dataset download/preparation workflows | --dataset | Datasets and legacy cases.json | Sibling scripts | CLI behavior | Confirmed: bare sibling imports work by file execution, fail under module import; imports downloader dependencies before --help | Retain CLI; lazy package-aware imports |
| tools/download_covid19_20.py | Fetch 199 pairs, check count/shape/affine | Public HF snapshot | Train files and list manifest | HF Hub, nibabel, NumPy | Count, pairing and geometry checks | Confirmed: schema is list of id/image/mask/shape/spacing_mm; no patient ID or intensity evidence; potential: labels and units not validated | Retain acquisition tool |
| tools/download_zenodo20.py | Resumable downloads and size verification | Zenodo record API | Four ZIP archives | requests | Partial download recovery | Potential: size-only verification cannot prove byte identity; no checksum enforcement | Retain acquisition tool |
| tools/prepare_zenodo20.py | Extract NIfTI, normalize identifiers, validate all pairs | Four ZIP archives | Preserved source filenames and legacy cases.json | nibabel, NumPy, zipfile | Safe basename extraction, indexing, geometry checks | Confirmed: IDs need not equal filenames; lung path omitted from manifest; potential: equal-size existing files are trusted | Retain CLI; share ID/path resolver and add lung path |
| tools/audit_zenodo20.py | Component loss, burden, continuity, intensity, encoding and geometry audit | Local external CT/masks | CSV and JSON | OpenCV, nibabel, NumPy | Descriptive analysis, not training preprocessing | Confirmed: constructs infection/lung filenames instead of consuming manifest; >0 assumes dedicated infection encoding | Retain descriptive tool; repair path resolution |
| tools/plot_zenodo20_audit.py | Plot case balance and lesion trajectories | Saved CSV/JSON | Two PNGs | matplotlib, NumPy | Existing plots | Confirmed: docstring names nonexistent analyze_dataset.py; potential: fixed examples require original 20 cases | Retain visualization tool |
| tools/view_slices.py | GUI, keyboard navigation, PNG preview/export | cases.json and volumes | GUI/PNG | tkinter, Pillow, nibabel, NumPy | Viewer panels, manifests | Confirmed: lung filename constructed; no geometry validation; display-only intensity heuristic varies by slice | Retain viewer; resolve masks from manifest and check alignment |
| tools/compare_datasets.py (pre-existing, untracked) | Pairwise slice plots, full-volume stats, histograms | Both local manifests | outputs PNG/CSV | matplotlib, nibabel, NumPy | Exploratory displays only | Confirmed: absolute local root, execution on import, manual unquoted CSV, lung filename construction, assumes axial ordering/HU; potential: memory cost and external-label-driven decisions | Preserve user file; excluded from package business logic |
| tools/compare_stats_fast.py (pre-existing, untracked) | Header stats and sampled intensity/labels | Both datasets | outputs CSV/plots | matplotlib, nibabel, NumPy | Header inspection idea | Confirmed: absolute root, side effects on import, inferred lung paths, HU labeling without provenance | Preserve user file; exploratory only |
| analysis/zenodo20/README.md | Historic audit provenance and definitions | Saved audit | Human documentation | None | Explicit source limitations | States mixed Kaggle/Zenodo provenance; do not reinterpret as proof of current archive identity | Preserved byte-for-byte |
| analysis/zenodo20/case_statistics.csv | 20 volume summaries | Historic audit | Stored evidence | CSV reader | Shape, spacing, orientation and resize observations | Historical result, not a new run | Preserved byte-for-byte |
| analysis/zenodo20/summary.json | Aggregates and 20 slice trajectories | Historic audit | Stored evidence | JSON reader | Geometry and intensity cautions | Reports 38/19/6 lost components at 128/256/352 and one lost positive slice at 128/256; nearest neighbor cannot guarantee lesion survival | Preserved byte-for-byte |
| analysis/zenodo20/*.png | Dataset overview and trajectories | Historic audit | Stored figures | Image reader | Human review | Historical descriptive figures, not model performance | Preserved byte-for-byte |

## Evidence and limits

Confirmed source defects are distinguished above from potential data risks. Original
tools check some pair geometry, but provide no shared training loader, configuration,
slab sampler, leakage validator, metrics, reconstruction, model interface or tests.
Those are missing features, not proof of a failed training system.

The existing manifests contain 199 and 20 records. Patient identity is not verified
by filenames. The primary manifest's spacing distribution can be inspected without
claiming that all scans share spacing or HU units. Z spacing means center-to-center
distance; it is not slice thickness. Historical external descriptive statistics must
not be used for threshold selection, preprocessing tuning, or model selection.

New training preprocessing uses explicit intensity provenance, paired orientation,
nearest-neighbor labels, retained geometry and visible resize-loss diagnostics.
The viewer's heuristic window is never imported into training. Original MERIT source
and a complete specification are unavailable in this repository; no TEP/MER/PCG
implementation can be certified here.
