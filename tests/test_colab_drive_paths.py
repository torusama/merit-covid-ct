from pathlib import Path

import pytest

from tools.colab_drive_paths import drive_folder_id, resolve_drive_project, rebase_drive_input, ensure_drive_mount
from tools.prepare_colab_alignment_notebook import drive_project_setup_source


FOLDER='1wW7uej60ZBLOL0_WVv-uBDQsLPkr6ar8'
LINK='https://drive.google.com/drive/folders/'+FOLDER+'?usp=drive_link'


def test_existing_owner_drive_is_used_without_mount_auth_or_shortcut(tmp_path):
    root=tmp_path/'drive/MyDrive'
    project=root/'MERIT_alignment_test'
    (project/'inputs').mkdir(parents=True)
    class Drive:
        def mount(self,*args,**kwargs):
            raise AssertionError('Existing owner mount must not be remounted')
    assert ensure_drive_mount(Drive(),tmp_path/'drive')==root
    assert resolve_drive_project(project,my_drive=root)==str(project)


def test_missing_mount_requests_auth_and_explicit_force_remount_is_respected(tmp_path):
    calls=[]
    class Drive:
        def mount(self,path,**kwargs):
            calls.append(kwargs['force_remount'])
            (Path(path)/'MyDrive').mkdir(parents=True,exist_ok=True)
    mount=tmp_path/'drive'
    ensure_drive_mount(Drive(),mount)
    ensure_drive_mount(Drive(),mount,force=True)
    assert calls==[False,True]


def test_shared_shortcut_uses_existing_project_without_copying_data(tmp_path):
    project=tmp_path/'MyDrive/MERIT_alignment_test'
    (project/'inputs').mkdir(parents=True)
    (project/'runs/alignment_pair_300_v1').mkdir(parents=True)
    assert drive_folder_id(LINK)==FOLDER
    assert resolve_drive_project(project,shared_url=LINK,my_drive=tmp_path/'MyDrive')==str(project)


def test_custom_shared_path_rebases_default_inputs_and_preserves_run(tmp_path):
    project=tmp_path/'MyDrive/shared project'
    (project/'inputs').mkdir(parents=True)
    old=Path('/content/drive/MyDrive/MERIT_alignment_test')
    namespace=dict(DRIVE_ROOT=str(old),SHARED_PROJECT_PATH=str(project),SHARED_FOLDER_URL=LINK,
        DATASET_SOURCE=str(old/'inputs/Train'),EPOCH0_CHECKPOINT=str(old/'inputs/epoch0/last.pt'),
        EPOCH0_HISTORY=str(old/'inputs/epoch0/history.jsonl'),RUN_NAME='alignment_pair_300_v1',
        mounted_my_drive=tmp_path/'MyDrive')
    exec(drive_project_setup_source(),namespace)
    assert namespace['DRIVE_ROOT']==str(project)
    assert namespace['DATASET_SOURCE']==str(project/'inputs/Train')
    assert namespace['EPOCH0_CHECKPOINT']==str(project/'inputs/epoch0/last.pt')
    assert namespace['EPOCH0_HISTORY']==str(project/'inputs/epoch0/history.jsonl')
    assert namespace['RUN_NAME']=='alignment_pair_300_v1'
    external=tmp_path/'other/last.pt'
    assert rebase_drive_input(external,old,project)==str(external)


def test_missing_shortcut_explains_required_sharing_and_does_not_create_project(tmp_path):
    my_drive=tmp_path/'MyDrive';my_drive.mkdir()
    (my_drive/'visible-folder').mkdir()
    project=my_drive/'MERIT_alignment_test'
    with pytest.raises(FileNotFoundError) as error:
        resolve_drive_project(project,shared_url=LINK,my_drive=my_drive)
    text=str(error.value)
    assert 'Editor' in text and 'Add shortcut' in text and 'visible-folder' in text
    assert LINK in text and not project.exists()


def test_wrong_folder_and_known_different_shortcut_target_are_rejected(tmp_path):
    wrong=tmp_path/'wrong';wrong.mkdir()
    with pytest.raises(FileNotFoundError,match='inputs'):
        resolve_drive_project(wrong,shared_url=LINK)
    target=tmp_path/'.shortcut-targets-by-id/another-id/project'
    (target/'inputs').mkdir(parents=True)
    with pytest.raises(ValueError,match='different Drive folder ID'):
        resolve_drive_project(target,shared_url=LINK)


def test_web_url_is_not_treated_as_a_filesystem_path(tmp_path):
    with pytest.raises(ValueError,match='SHARED_FOLDER_URL'):
        resolve_drive_project(LINK)
    with pytest.raises(ValueError,match='Google Drive folder link'):
        drive_folder_id('https://drive.google.com.example.org/drive/folders/'+FOLDER)
