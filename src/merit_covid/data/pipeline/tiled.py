"""Read the existing NIfTI V3 fold-0 tile export; never crop or normalize it.

Manifest validation uses original volume filenames for the saved fingerprint.
Only the Dataset expands a volume into its sixteen tiles. Production cohort
counts are fixed; the private counts argument is only for software fixtures.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from .cases import Case, load_cases
from .config import PreprocessConfig
from .geometry import _validate_header, affine_mm, restore_volume
from .preprocessing import read_arrays
from .slab import sample_arrays
from .splits import validate_splits

SCHEMA = "merit_v3_fold0_tiling128_nifti_v1"
MODE = "tiling128_v3_fold0"
COHORT_COUNTS = (199, 13705, (127, 32, 40))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(root, value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts):
        raise ValueError(f"Unsafe relative dataset path: {value!r}")
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes tile root: {value}")
    return path


def validate_tile_policy(config):
    if (config.intensity_encoding != "native" or config.size_hw is not None
            or config.slab_size != 11 or config.sampling_mode != "index"
            or config.boundary_policy != "edge" or config.duplicate_slice_policy != "unique"
            or config.zscore_fit is not None):
        raise ValueError("Precomputed V3 tiles require native pass-through, size_hw=null, slab11, index/edge/unique; no normalization/resize")


def require_fold0(fold):
    if type(fold) is not int or fold != 0:
        raise ValueError("FOLD0_ONLY: V3 was fitted using fold-0 TRAIN; fold 1-4/CV would leak preprocessing statistics")


def _geometry(record, name, assume_mm):
    try:
        shape = record["shape_xyz"]
        affine = np.asarray(record["affine"], dtype=np.float64)
        unit = record["spatial_unit"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing/invalid {name} geometry") from exc
    if (len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape)
            or affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError(f"Invalid {name} shape/affine")
    if record.get("orientation") != "".join(nib.aff2axcodes(affine)):
        raise ValueError(f"Invalid {name} orientation")
    if not np.allclose(record.get("spacing", []), nib.affines.voxel_sizes(affine), atol=1e-4, rtol=1e-4):
        raise ValueError(f"Invalid {name} spacing")
    factors = {"mm": 1., "meter": 1000., "micron": .001}
    if unit not in factors and not (unit == "unknown" and assume_mm):
        raise ValueError(f"{name}: unknown units require explicit assume_mm")
    mm = affine.copy(); mm[:3] *= factors.get(unit, 1.)
    return shape, affine, mm


def full_metadata(record, assume_mm, require_mask=True):
    """Validate full-volume orientation/affine relation without allocating voxels."""
    full = record["full_lps_geometry"]
    original = record["original_ct_geometry"]
    shape, affine, mm = _geometry(full, "full LPS", assume_mm)
    osh, oa, omm = _geometry(original, "original CT", assume_mm)
    if shape != [512, 512, record["depth"]] or nib.aff2axcodes(affine) != ("L", "P", "S"):
        raise ValueError("Full geometry must be LPS [512,512,depth]")
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(oa),
                                               nib.orientations.axcodes2ornt(("L", "P", "S")))
    expected = omm @ nib.orientations.inv_ornt_aff(transform, osh)
    canonical_shape = np.asarray(osh)[np.argsort(transform[:, 0].astype(int))].tolist()
    if shape != canonical_shape or not np.allclose(mm, expected, atol=1e-4, rtol=0):
        raise ValueError("Full/original CT orientation and affine relationship is inconsistent")
    if require_mask:
        gsh, ga, gmm = _geometry(record["original_gt_geometry"], "original GT", assume_mm)
        gt_transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(ga),
                                                      nib.orientations.axcodes2ornt(("L", "P", "S")))
        if (np.asarray(gsh)[np.argsort(gt_transform[:, 0].astype(int))].tolist() != shape
                or not np.allclose(gmm @ nib.orientations.inv_ornt_aff(gt_transform, gsh), mm, atol=1e-4, rtol=0)):
            raise ValueError("Original GT is not aligned with full LPS CT")
    inverse = nib.orientations.ornt_transform(nib.orientations.io_orientation(affine),
                                             nib.orientations.io_orientation(oa))
    return dict(case_id=record["case_id"], original_shape_xyz=osh, canonical_shape_xyz=shape,
        original_affine=oa.tolist(), original_affine_mm=omm.tolist(), canonical_affine_mm=mm.tolist(),
        canonical_to_original_orientation=inverse.tolist(), original_orientation=list(nib.aff2axcodes(oa)),
        canonical_orientation=["L", "P", "S"], native_spacing_mm=nib.affines.voxel_sizes(mm).tolist(),
        native_size_hw=[512, 512], processed_size_hw=[512, 512], z_resampled=False,
        spatial_unit_in_header=original["spatial_unit"], assume_mm=assume_mm,
        source_full_lps_geometry=deepcopy(full), source_original_ct_geometry=deepcopy(original))


@dataclass(frozen=True)
class TiledCase:
    case_id: str
    group_id: str
    manifest: Path
    record: dict
    metadata: dict

    @property
    def image(self):
        return self.path(0, "ct")  # Compatibility only; source_identity binds ALL sixteen tiles.

    @property
    def mask(self):
        return self.path(0, "gt")

    def path(self, tile_id, role):
        return safe_path(self.manifest.parent, self.record["tiles"][tile_id][role]["file"])


class TileBundle:
    def __init__(self, root, *, assume_mm=False, require_mask=True, _counts=None):
        self.root = Path(root).resolve()
        self.assume_mm, self.require_mask = assume_mm, require_mask
        n, masters, part_counts = COHORT_COUNTS if _counts is None else _counts
        if not (self.root / "READY_ALL_199.json").is_file():
            raise FileNotFoundError(f"Expected complete TILING128_V3_FOLD0_NIFTI root with READY_ALL_199.json: {self.root}; RAW or V3 full512 is not tiled input")
        ready = read_json(self.root / "READY_ALL_199.json")
        self.manifest_sha256 = file_sha256(self.root / "TILE_MANIFEST.json")
        if (ready.get("status") != "PASS" or ready.get("cases") != n or ready.get("tile_files") != n * 32
                or ready.get("fold_scope") != "FOLD0_ONLY" or ready.get("tile_manifest_sha256") != self.manifest_sha256):
            raise ValueError("READY_ALL_199 metadata/manifest SHA mismatch or incomplete export")
        manifest = read_json(self.root / "TILE_MANIFEST.json")
        expected = dict(schema=SCHEMA, status="COMPLETE", source="ALREADY_NORMALIZED_V3_FOLD0",
                        cases_exported=n, ct_tile_files=n*16, gt_tile_files=n*16,
                        master_slices=masters, tile_masters=masters*16)
        if (any(manifest.get(k) != v for k, v in expected.items())
                or manifest.get("normalization_applied_during_tiling") is not False):
            raise ValueError("Not the complete precomputed V3 fold0 Tiling128 dataset (RAW/512 input is invalid)")
        meta = self.root / "source_metadata"
        source_records = read_json(meta / "cases.json")
        if not isinstance(source_records, list):
            raise ValueError("Source cases.json must retain the exporter's list of original volume records")
        for source_record in source_records:
            for role in ("image", "mask"):
                safe_path(meta, source_record.get(role))
        self.source_cases, self.identity = load_cases(meta / "cases.json", expected_count=n, check_files=False)
        self.split_path = meta / "splits_covid19_20_5fold.json"
        self.split = read_json(self.split_path)
        validate_splits(self.split, self.source_cases)  # Original filenames; never fingerprint tile filenames.
        self.split_sha256_lf = hashlib.sha256(self.split_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        fold = next(f for f in self.split["folds"] if f["fold"] == 0)
        if tuple(len(fold[p]["case_ids"]) for p in ("train", "validation", "test")) != tuple(part_counts):
            raise ValueError("Fold0 must retain original volume partitions 127/32/40")
        partitions = {cid: p for p in ("train", "validation", "test") for cid in fold[p]["case_ids"]}
        export = read_json(meta / "EXPORT_COMPLETE.json")
        if (export.get("status") != "COMPLETE_WITH_PROVENANCE_LIMITATION"
                or export.get("policy") != "train_fovaware_clipped_zscore_v1"
                or export.get("count") != n or export.get("train_count") != part_counts[0]):
            raise ValueError("Source metadata is not the verified V3 fold0 export")
        if export.get("new_manifest_fingerprint", self.split["manifest_fingerprint"]) != self.split["manifest_fingerprint"]:
            raise ValueError("V3 source fingerprint differs from the copied original-volume split")
        source_config = yaml.safe_load((meta / "merit_precomputed_v3.yaml").read_text(encoding="utf-8"))
        validate_tile_policy(PreprocessConfig(**source_config["preprocess"]))
        source_files = export.get("files", [])
        source_by_id = {r["case_id"]: r for r in source_files}
        by_id = {c.case_id: c for c in self.source_cases}
        entries = manifest.get("cases", [])
        if (len(entries) != n or len(source_by_id) != n or len(source_files) != n
                or set(source_by_id) != set(by_id) or {r["case_id"] for r in entries} != set(by_id)):
            raise ValueError("Tile/source case IDs missing or duplicated")
        self.cases = []; paths = set()
        for entry in entries:
            cid = entry["case_id"]
            if entry["case_manifest"] != f"cases/{cid}/case_export.json":
                raise ValueError("Wrong case_manifest path/case_id")
            path = safe_path(self.root, entry["case_manifest"])
            record = read_json(path)
            if (record.get("schema") != SCHEMA or record.get("status") != "COMPLETE"
                    or record.get("case_id") != cid or type(record.get("depth")) is not int
                    or record["depth"] < 1 or record["depth"] != entry["depth"]
                    or record.get("partition_fold0") != partitions[cid]
                    or entry["partition_fold0"] != partitions[cid]
                    or source_by_id[cid]["partition"] != partitions[cid]):
                raise ValueError("Wrong case_id, depth or partition in case_export")
            if (record.get("source_ct_sha256") != source_by_id[cid]["ct_output_sha256"]
                    or record.get("source_gt_sha256") != source_by_id[cid]["gt_output_sha256"]):
                raise ValueError("Case export source SHA does not match frozen V3 source metadata")
            metadata = full_metadata(record, assume_mm, require_mask)
            tiles = record.get("tiles", [])
            if len(tiles) != 16 or {t["tile_id"] for t in tiles} != set(range(16)):
                raise ValueError(f"{cid}: missing/duplicate tile; need IDs 0..15")
            tiles.sort(key=lambda t: t["tile_id"])
            for tile in tiles:
                tid = tile["tile_id"]; row, col = divmod(tid, 4)
                crop = dict(x0=col*128, x1=(col+1)*128, y0=row*128, y1=(row+1)*128)
                translation = np.eye(4); translation[:3, 3] = [crop["x0"], crop["y0"], 0]
                if (type(tid) is not int or tile.get("row") != row or tile.get("col") != col
                        or type(tile.get("row")) is not int or type(tile.get("col")) is not int
                        or tile.get("xyz_crop") != crop or tile.get("shape_xyz") != [128, 128, record["depth"]]
                        or not np.allclose(tile["affine"], np.asarray(record["full_lps_geometry"]["affine"]) @ translation, atol=1e-4, rtol=0)):
                    raise ValueError(f"{cid}: wrong tile_id/XY crop/depth/translated affine")
                for role, dtype, suffix in (("ct", "float32", "ct"), ("gt", "uint16", "seg")):
                    desc = tile[role]
                    name = f"{cid}_t{tid:02d}_{suffix}.nii.gz"
                    target = safe_path(path.parent, desc["file"])
                    if (desc["file"] != name or desc.get("dtype") != dtype or target in paths
                            or not isinstance(desc.get("sha256"), str) or len(desc["sha256"]) != 64
                            or any(c not in "0123456789abcdef" for c in desc["sha256"])):
                        raise ValueError(f"{cid}: invalid/reused tile file, dtype or SHA")
                    paths.add(target)
                    if (role == "ct" or require_mask) and not target.is_file():
                        raise FileNotFoundError(f"Missing {role} tile: {target}")
            self.cases.append(TiledCase(cid, by_id[cid].group_id, path, record, metadata))
        if sum(c.record["depth"] for c in self.cases) != masters:
            raise ValueError("Depth totals disagree with 13705 masters / 219280 tile-masters")
        self.cases.sort(key=lambda c: c.case_id)
        self.manifest = manifest
        self.source_export = export

    def source_identity(self):
        """Small JSON hashes + tile file stats; no rehash of 6368 CT/GT at each epoch.

        Full byte integrity is established by explicit --hash-files preflight.
        Conservative size/mtime binding also refuses unnoticed remount changes.
        """
        meta_paths = [self.root / "TILE_MANIFEST.json", self.root / "READY_ALL_199.json",
                      *sorted((self.root / "source_metadata").glob("*.json")),
                      self.root / "source_metadata/merit_precomputed_v3.yaml"]
        return dict(mode=MODE, fold_scope="FOLD0_ONLY", tile_manifest_sha256=self.manifest_sha256,
            metadata={p.relative_to(self.root).as_posix(): file_sha256(p) for p in meta_paths},
            implementation_sha256=implementation_identity(),
            cases=[dict(case_id=c.case_id, group_id=c.group_id, case_manifest_sha256=file_sha256(c.manifest),
                tiles=[dict(tile_id=t["tile_id"], **{role: self._file_identity(c.path(t["tile_id"], role), t[role]["sha256"])
                       for role in ("ct", "gt")}) for t in c.record["tiles"]]) for c in self.cases])

    @staticmethod
    def _file_identity(path, sha):
        stat = path.stat()
        return dict(path=str(path), declared_sha256=sha, size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def preflight(self, *, hash_files=False, voxels=False, case_ids=None):
        selected = self.cases if case_ids is None else [c for c in self.cases if c.case_id in set(case_ids)]
        if not selected or (case_ids is not None and {c.case_id for c in selected} != set(case_ids)):
            raise ValueError("Unknown/empty preflight selection")
        checked = 0
        for case in selected:
            for tile in case.record["tiles"]:
                for role in (("ct", "gt") if self.require_mask else ("ct",)):
                    path = case.path(tile["tile_id"], role)
                    inspect_tile(case, tile["tile_id"], role, self.assume_mm)
                    if hash_files and file_sha256(path) != tile[role]["sha256"]:
                        raise ValueError(f"SHA mismatch: {path}")
                    if voxels:
                        values = nib.load(path).get_fdata(dtype=np.float64 if role == "gt" else np.float32, caching="unchanged")
                        if not np.isfinite(values).all() or (role == "gt" and not np.isin(values, [0, 1]).all()):
                            raise ValueError(f"Nonfinite/nonbinary tile: {path}")
                        del values
                    checked += 1
        return dict(status="PASS", original_cases=len(selected), files_checked=checked,
                    headers=True, hash_files=hash_files, voxels=voxels, fold_scope="FOLD0_ONLY")


def implementation_identity():
    """Freeze executable model/loss/data/trainer sources for tile checkpoint resume."""
    package = Path(__file__).resolve().parents[2]
    files = []
    for directory in ("models/unetrpp", "models/merit_unetrpp", "data/pipeline"):
        files.extend((package / directory).rglob("*.py"))
    for name in ("training", "activation_checkpointing", "factorized_attention", "numerical_safety",
                 "normalization", "slab_inference", "tiled_inference", "prediction"):
        files.append(package / "engine" / (name + ".py"))
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(files)}


def inspect_tile(case, tile_id, role, assume_mm):
    tile = case.record["tiles"][tile_id]
    image = nib.load(case.path(tile_id, role))
    _validate_header(image, case.case_id, role)
    if (list(image.shape) != tile["shape_xyz"] or nib.aff2axcodes(image.affine) != ("L", "P", "S")
            or not np.allclose(image.affine, tile["affine"], atol=1e-4, rtol=0)
            or image.header.get_xyzt_units()[0] != case.record["full_lps_geometry"]["spatial_unit"]
            or image.get_data_dtype() != np.dtype(tile[role]["dtype"])
            or float(image.dataobj.slope) != 1. or float(image.dataobj.inter) != 0.):
        raise ValueError(f"{case.case_id}: tile header shape/affine/dtype/scaling mismatch ({role}, {tile_id})")
    affine_mm(image, assume_mm)
    return image


class TiledCTSlabDataset(Dataset):
    """One original volume block = 16*Z samples; lazy cache is ONE tile pair.

    Existing samplers still group by original case via ends. Random order inside
    a case can reload compressed tiles frequently; no throughput claim is made.
    """
    def __init__(self, cases, config=None, *, require_mask=True):
        self.cases = list(cases); self.config = config or PreprocessConfig()
        validate_tile_policy(self.config)
        self.require_mask = require_mask; self.partition = None
        if (not self.cases or any(not isinstance(c, TiledCase) for c in self.cases)
                or len({c.case_id for c in self.cases}) != len(self.cases)):
            raise ValueError("Tiled Dataset requires nonempty unique original TiledCase volumes")
        self.ends = [0]
        for case in self.cases:
            if case.metadata["assume_mm"] != self.config.assume_mm:
                raise ValueError("Tile metadata/config unit assumption mismatch")
            self.ends.append(self.ends[-1] + 16 * case.record["depth"])
        self._cached_key = None; self._cached = None

    def __len__(self):
        return self.ends[-1]

    def __getstate__(self):
        return {**self.__dict__, "_cached_key": None, "_cached": None}

    def sample_key(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError(index)
        case_index = bisect_right(self.ends, index) - 1
        case = self.cases[case_index]
        tile, master = divmod(int(index) - self.ends[case_index], case.record["depth"])
        return case_index, tile, master

    def __getitem__(self, index):
        case_index, tile_id, master = self.sample_key(index)
        case = self.cases[case_index]; key = (case_index, tile_id)
        if self._cached_key != key:
            self._cached = None; self._cached_key = None
            inspect_tile(case, tile_id, "ct", self.config.assume_mm)
            if self.require_mask:
                inspect_tile(case, tile_id, "gt", self.config.assume_mm)
            pair = Case(case.case_id, case.path(tile_id, "ct"), case.path(tile_id, "gt") if self.require_mask else None, case.group_id)
            self._cached = read_arrays(pair, self.config, require_mask=self.require_mask)
            self._cached_key = key
        image, mask, local = self._cached
        sample = sample_arrays(image, mask, master, 11, local, sampling_mode="index",
                               boundary_policy="edge", duplicate_slice_policy="unique")
        tile = case.record["tiles"][tile_id]
        metadata = {**local, "case_id": case.case_id, "group_id": case.group_id,
                    "tile_id": tile_id, "xyz_crop": tile["xyz_crop"], "full_volume_metadata": case.metadata,
                    "tile_source": MODE, "case_manifest": str(case.manifest)}
        return {**{k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in sample.items()},
                "case_id": case.case_id, "group_id": case.group_id, "tile_id": tile_id, "metadata": metadata}


def reconstruct_gt(case, *, assume_mm=False):
    """Read only GT tiles, after inference, and restore original full XYZ grid."""
    full = np.empty((case.record["depth"], 512, 512), dtype=np.float32)
    for tile in case.record["tiles"]:
        img = inspect_tile(case, tile["tile_id"], "gt", assume_mm)
        values = img.get_fdata(dtype=np.float64, caching="unchanged")
        if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError(f"{case.case_id}: GT must be finite binary 0/1")
        c = tile["xyz_crop"]
        full[:, c["y0"]:c["y1"], c["x0"]:c["x1"]] = values.transpose(2, 1, 0)
    return full, restore_volume(full, case.metadata)
