"""Use an existing Drive mount, either the owner's Drive or a shared folder."""
from pathlib import Path
import re
from urllib.parse import urlparse, parse_qs


def ensure_drive_mount(drive, mountpoint='/content/drive', force=False):
    """Preserve an existing mount without triggering another auth request."""
    mountpoint = Path(mountpoint)
    roots = (mountpoint / 'MyDrive', mountpoint / 'My Drive')
    if not force:
        for root in roots:
            if root.is_dir():
                print('Using existing Drive mount:', root)
                return root
    drive.mount(str(mountpoint), force_remount=force)
    for root in roots:
        if root.is_dir():
            return root
    raise RuntimeError('Drive is not mounted. Authorize the account owning the folder, or an account with access.')


def drive_folder_id(url):
    if not url:
        return None
    parsed = urlparse(str(url).strip())
    if parsed.scheme != 'https' or parsed.hostname != 'drive.google.com':
        raise ValueError('SHARED_FOLDER_URL must be a Google Drive folder link (https://drive.google.com/drive/folders/...).')
    match = re.search(r'/folders/([A-Za-z0-9_-]+)(?:/|$)', parsed.path)
    folder_id = match.group(1) if match else parse_qs(parsed.query).get('id', [None])[0]
    if not folder_id or not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id):
        raise ValueError('Cannot extract the shared folder ID from SHARED_FOLDER_URL.')
    return folder_id


def resolve_drive_project(configured_root, shared_path='', shared_url='', my_drive='/content/drive/MyDrive'):
    """Use an already mounted owner folder or an accessible shared shortcut."""
    folder_id = drive_folder_id(shared_url)
    root = Path(shared_path or configured_root).expanduser()
    if str(shared_path or configured_root).startswith(('http://', 'https://')):
        raise ValueError('Put the folder web link in SHARED_FOLDER_URL; SHARED_PROJECT_PATH is a mounted folder path.')
    if not root.is_absolute():
        root = Path(my_drive) / root
    guidance = (
        ' Neu truoc day da mount truc tiep Drive account chu folder, dung lai cach do; khong can shortcut. '
        'Kiem tra account Drive dang mount va duong dan SHARED_PROJECT_PATH. '
        'Phuong an khac khi dung Drive account moi: account cu share folder voi quyen Editor, '
        'account moi vao Shared with me -> Organize -> Add shortcut -> My Drive, '
        'dat ten shortcut MERIT_alignment_test. Neu ten/vi tri khac, sua SHARED_PROJECT_PATH o Cell 1. '
        'Chi dan link khong tu cap quyen truy cap. Folder: ' + str(shared_url or '(not configured)')
    )
    if not root.is_dir():
        visible = sorted(p.name for p in Path(my_drive).iterdir())[:25] if Path(my_drive).is_dir() else []
        raise FileNotFoundError('Khong thay folder/shortcut: ' + str(root)
                                + '; My Drive dang co: ' + repr(visible) + '.' + guidance)
    if not (root / 'inputs').is_dir():
        visible = sorted(p.name for p in root.iterdir())[:25]
        raise FileNotFoundError('Folder nay chua co inputs: ' + str(root)
                                + '; cac muc: ' + repr(visible) + '.' + guidance)
    # If Drive exposes a shortcut target in its resolved path, reject a known
    # different target instead of silently using a same-named project.
    parts = root.resolve().parts
    marker = '.shortcut-targets-by-id'
    if folder_id and marker in parts:
        index = parts.index(marker)
        if len(parts) > index + 1 and parts[index + 1] != folder_id:
            raise ValueError('Shortcut points to a different Drive folder ID.' + guidance)
    return str(root)


def rebase_drive_input(path, previous_root, resolved_root):
    """Rebase default inputs while preserving explicitly external input paths."""
    path = Path(path)
    try:
        relative = path.relative_to(Path(previous_root))
    except ValueError:
        return str(path)
    return str(Path(resolved_root) / relative)
