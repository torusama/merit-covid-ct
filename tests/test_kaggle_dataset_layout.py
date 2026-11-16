"""Kaggle extracts the three unknown-unit pairs into a different layout."""
from pathlib import Path
import pytest
from scripts.kaggle.dataset_layout import resolve_case_files


@pytest.mark.parametrize('case', ['volume-covid19-A-0161', 'volume-covid19-A-0585', 'volume-covid19-A-0599'])
def test_unknown_unit_cases_resolve_nested_ct_and_mask(case):
    names = {f'{case}_ct.nii/{case}_ct.nii', f'{case}_seg.nii/{case}_seg.nii'}
    result = resolve_case_files(Path('Train'), case, names)
    assert {p.as_posix() for p in result.values()} == {'Train/' + n for n in names}


def test_standard_case_and_original_gzip_layouts():
    case = 'volume-covid19-A-0025'
    for names in ({f'{case}_ct.nii/{case}.nii', f'{case}_seg.nii'},
                  {f'{case}_ct.nii.gz', f'{case}_seg.nii.gz'}):
        assert len(resolve_case_files(Path('Train'), case, names)) == 2


def test_missing_and_ambiguous_pairs_are_rejected():
    case = 'volume-covid19-A-0025'
    with pytest.raises(ValueError, match='mask'):
        resolve_case_files(Path('Train'), case, {f'{case}_ct.nii.gz'})
    with pytest.raises(ValueError, match='image'):
        resolve_case_files(Path('Train'), case, {f'{case}_ct.nii.gz', f'{case}_ct.nii'})
