"""Process one original CT/mask pair through the shared read_arrays/read_volume API.

NIfTI input uses X,Y,Z; returned tensors use Z,Y,X (Z,H,W). Operations create
arrays in memory and never overwrite original NIfTI files.
"""

from __future__ import annotations

import numpy as np
import warnings

from .cases import Case
from .config import PreprocessConfig
from .geometry import geometry_metadata, inspect_pair, reorient_to_lps


def validate_voxels(image_xyz: np.ndarray, mask_xyz: np.ndarray | None, case_id: str) -> None:
    """Validate full scaled CT/mask arrays before windowing or resizing."""
    if not np.isfinite(image_xyz).all() or (mask_xyz is not None and not np.isfinite(mask_xyz).all()):
        raise ValueError(f"{case_id}: CT/mask contains NaN or Inf")
    if mask_xyz is not None and not np.isin(mask_xyz, [0, 1]).all():
        raise ValueError(
            f"{case_id}: mask labels must be exactly 0/1; "
            f"found {np.unique(mask_xyz)}"
        )


def normalize_ct(
    image_xyz: np.ndarray,
    config: PreprocessConfig,
    case_id: str,
) -> np.ndarray:
    """Preserve scaled intensities by default; window only under explicit policy."""
    if config.intensity_encoding == "native":
        return image_xyz
    if config.intensity_provenance is None:
        warnings.warn(f"{case_id}: {config.intensity_encoding} is an explicit intensity policy, "
                      "not verified HU/normalization provenance; supply intensity_provenance from source evidence",
                      UserWarning, stacklevel=2)

    minimum = float(image_xyz.min())
    maximum = float(image_xyz.max())

    if config.intensity_encoding == "auto_hu" and (
        minimum >= -300 or maximum <= -300 or minimum < -32768 or maximum > 65535
    ):
        raise ValueError(
            f"{case_id}: cannot infer HU from scaled range [{minimum}, {maximum}]; "
            "inspect encoding and explicitly configure it"
        )

    if config.intensity_encoding == "normalized_0_1":
        if minimum < 0 or maximum > 1:
            raise ValueError(f"{case_id}: normalized_0_1 requires intensities in [0, 1]")
        return image_xyz

    low, high = config.hu_window
    return (np.clip(image_xyz, low, high) - low) / (high - low)


def to_slice_tensor(array_xyz: np.ndarray):
    """Reorder XYZ to ZYX storage without performing another reorientation."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(array_xyz.transpose(2, 1, 0)))


def resize_in_plane(
    image_zyx,
    mask_zyx,
    size_hw: tuple[int, int] | None,
):
    """Resize XY planes, treating Z as a batch rather than interpolating it.

    CT uses bilinear with align_corners=False; masks use nearest-exact for 0/1
    labels. size_hw=None preserves native H,W.
    """
    if size_hw is None or tuple(image_zyx.shape[1:]) == size_hw:
        return image_zyx, mask_zyx
    from torch.nn import functional as F

    resized_image = F.interpolate(
        image_zyx[:, None],
        size=size_hw,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    resized_mask = None if mask_zyx is None else F.interpolate(
        mask_zyx[:, None],
        size=size_hw,
        mode="nearest-exact",
    )[:, 0]
    return resized_image, resized_mask


def lesion_retention(mask_zyx, size_hw):
    """Report completely lost 2D connected components under nearest-exact XY.

    Mapping is the same pixel-centre nearest rule as resize_in_plane, without
    allocating a resized integer-labelled volume. Never used to select samples.
    """
    from scipy.ndimage import label
    height, width = mask_zyx.shape[1:]
    target = tuple(size_hw or (height, width))
    yy = np.minimum(((np.arange(target[0]) + 0.5) * height / target[0]).astype(int), height - 1)
    xx = np.minimum(((np.arange(target[1]) + 0.5) * width / target[1]).astype(int), width - 1)
    lost_slices, lost_components = [], []
    for z, plane in enumerate(mask_zyx):
        if not plane.any():
            continue
        components, count = label(plane > 0)  # 4-connected in-plane components.
        survived = np.unique(components[np.ix_(yy, xx)])
        survived = survived[survived > 0]
        missing = int(count - len(survived))
        if missing:
            lost_components.append({"slice": z, "lost": missing, "native_components": int(count)})
        if not len(survived):
            lost_slices.append(z)
    return {"lost_positive_slices": lost_slices, "lost_components_by_slice": lost_components,
            "connectivity": "2D 4-connected", "method": "nearest-exact XY pixel-centre mapping"}


def read_arrays(
    case: Case,
    config: PreprocessConfig,
    *, require_mask: bool = True,
):
    """Read a case as (CT[Z,H,W], optional mask[Z,H,W], metadata).

    This orchestrates shared preprocessing for lazy Dataset volume loads and
    audits of real CT/mask pairs.
    """
    # 1. Validate CT/mask headers before transforming any voxels.
    original_ct, original_mask = inspect_pair(case, config, require_mask=require_mask)

    # 2. Standardize to LPS by axis permutation/flips, without interpolation.
    canonical_ct = reorient_to_lps(original_ct)
    canonical_mask = reorient_to_lps(original_mask) if original_mask is not None else None

    # 3. NiBabel applies slope/intercept; validate values/labels on the native grid.
    image_xyz = canonical_ct.get_fdata(dtype=np.float32, caching="unchanged")
    mask_xyz = canonical_mask.get_fdata(dtype=np.float32, caching="unchanged") if canonical_mask is not None else None
    validate_voxels(image_xyz, mask_xyz, case.case_id)
    intensity_range = [float(image_xyz.min()), float(image_xyz.max())]
    label_values = np.unique(mask_xyz).astype(int).tolist() if mask_xyz is not None else []
    native_lesion_voxels = int(np.count_nonzero(mask_xyz)) if mask_xyz is not None else 0

    # 4. Default to native NIfTI-scaled intensity, without HU inference.
    processed_xyz = normalize_ct(image_xyz, config, case.case_id)
    del image_xyz

    # 5. Convert storage order; resize H,W only when explicitly requested.
    image_zyx = np.ascontiguousarray(processed_xyz.transpose(2, 1, 0))
    mask_zyx = np.ascontiguousarray(mask_xyz.transpose(2, 1, 0)) if mask_xyz is not None else None
    del processed_xyz, mask_xyz
    native_size_hw = list(image_zyx.shape[1:])
    retention = {"lost_positive_slices": [], "lost_components_by_slice": [], "resize_applied": False}
    if config.size_hw is not None and tuple(image_zyx.shape[1:]) != config.size_hw:
        if mask_zyx is not None:
            retention = {**lesion_retention(mask_zyx, config.size_hw), "resize_applied": True}
            if retention["lost_components_by_slice"]:
                warnings.warn(f"{case.case_id}: XY resize lost lesion components: {retention}", UserWarning, stacklevel=2)
        import torch
        image_zyx, mask_zyx = resize_in_plane(
            torch.from_numpy(image_zyx), torch.from_numpy(mask_zyx) if mask_zyx is not None else None,
            config.size_hw,
        )
        image_zyx = image_zyx.numpy()
        mask_zyx = mask_zyx.numpy() if mask_zyx is not None else None

    # 6. Retain native geometry and processing policy for provenance/restoration.
    metadata = geometry_metadata(original_ct, canonical_ct, config.assume_mm)
    metadata.update(
        {
            "processed_size_hw": list(image_zyx.shape[1:]),
            "native_size_hw": native_size_hw,
            "scaled_intensity_range": intensity_range,
            "intensity_encoding": config.intensity_encoding,
            "label_values": label_values,
            "native_lesion_voxels": native_lesion_voxels,
            "nifti_scaling": {
                "slope": float(original_ct.dataobj.slope),
                "intercept": float(original_ct.dataobj.inter),
            },
            "hu_window": list(config.hu_window) if config.hu_window is not None else None,
            "padding": "replicate_boundary",
            "intensity_provenance": config.intensity_provenance,
            "lesion_retention": retention,
        }
    )
    return image_zyx, mask_zyx, metadata


def read_volume(case: Case, config: PreprocessConfig, *, require_mask=True):
    """Torch API from Duong, backed by the same NumPy core as main's API."""
    import torch
    image, mask, metadata = read_arrays(case, config, require_mask=require_mask)
    return torch.from_numpy(image), torch.from_numpy(mask) if mask is not None else None, metadata
