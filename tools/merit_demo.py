"""Local interactive-demo backend using the actual completed epoch-0 checkpoint."""
from pathlib import Path
import hashlib
import json
import threading

import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'outputs/kaggle_review_20261116/epoch0_complete/MERIT/fold-00/checkpoints/last.pt'
EXPECTED_CHECKPOINT_SHA='22596971dc0f3ff7b6796756a61cf61187ded97472a9bdf6d754766165cd6934'
CACHE=ROOT/'outputs/demo_epoch0'
REGISTRY=ROOT/'configs/demo_models.json'
_LOCK=threading.RLock()


def models():
    entries=json.loads(REGISTRY.read_text(encoding='utf-8'))['models']
    if not entries or len({x['id'] for x in entries})!=len(entries):raise ValueError('Invalid model registry')
    return entries


def model_report(selection):
    history=selection.get('history')
    if not history:return None
    path=ROOT/history
    if not path.is_file():return None
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    matches=[row for row in rows if row['epoch']==selection['epoch']]
    if len(matches)!=1:raise ValueError('History must contain exactly one entry for the selected model epoch')
    return matches[0]['validation']


def register_model(checkpoint,history=None,label=None):
    """Register a trusted local completed checkpoint; preserve all older entries."""
    import torch
    from merit_covid.engine.training import protocol_fingerprint
    checkpoint=Path(checkpoint).expanduser().resolve(strict=True)
    if not checkpoint.is_file():raise ValueError('Checkpoint must be a file')
    h=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    sha=h.hexdigest()
    cp=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if cp.get('schema_version')!=2 or cp.get('in_progress') or not isinstance(cp.get('epoch'),int) or cp['epoch']<0:
        raise ValueError('Choose a completed epoch checkpoint, not in_progress.pt')
    if protocol_fingerprint(cp['config'],cp['provenance'])!=cp['protocol_fingerprint']:
        raise ValueError('Checkpoint protocol metadata is inconsistent')
    def portable(path):
        path=Path(path).resolve()
        return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    if history:
        history=Path(history).expanduser().resolve(strict=True)
    elif (checkpoint.parent/'history.jsonl').exists():history=checkpoint.parent/'history.jsonl'
    entry=dict(id=sha[:16],label=label or 'Epoch '+str(cp['epoch']),epoch=cp['epoch'],sha256=sha,
        checkpoint=portable(checkpoint),history=portable(history) if history else None,
        validation_case_ids=cp['provenance']['partitions']['validation']['case_ids'])
    # Check that the displayed metrics refer to this epoch and these volumes.
    report=model_report(entry)
    if report is not None and {r['case_id'] for r in report['per_case']}!=set(entry['validation_case_ids']):
        raise ValueError('Validation report case identities differ from checkpoint')
    with _LOCK:
        registry=json.loads(REGISTRY.read_text(encoding='utf-8'))
        for old in registry['models']:
            if (ROOT/old['checkpoint']).resolve()==checkpoint and old['sha256']!=sha:
                raise ValueError('Keep the old checkpoint in its own file; register the new model from a separate path')
        registry['models']=[old for old in registry['models'] if old['id']!=entry['id']]+[entry]
        temp=REGISTRY.with_suffix('.tmp');temp.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(REGISTRY)
    return entry


def cases(selection=None):
    selection=selection or models()[0]
    records=json.loads((ROOT/'data/covid19_20/cases.json').read_text(encoding='utf-8-sig'))
    report=model_report(selection)
    metrics={r['case_id']:r for r in report['per_case']} if report else {}
    by_id={r['id']:r for r in records}
    return [dict(case_id=case,ct=ROOT/'data/covid19_20'/by_id[case]['image'],
        mask=ROOT/'data/covid19_20'/by_id[case]['mask'],dice=metrics.get(case,{}).get('dice'))
        for case in sorted(selection['validation_case_ids']) if case in by_id]


def view_case(ct_path,mask_path):
    from merit_covid.data.pipeline.geometry import reorient_to_lps
    ct=reorient_to_lps(nib.load(ct_path)).get_fdata(dtype=np.float32).transpose(2,1,0)
    gt=reorient_to_lps(nib.load(mask_path)).get_fdata(dtype=np.float32).transpose(2,1,0)>0
    if ct.shape!=gt.shape:raise ValueError('CT/mask geometry differs')
    # Display-only percentiles: inference always receives original scaled pixels.
    low,high=np.percentile(ct[::max(1,len(ct)//20)],(1,99))
    if not np.isfinite(low+high) or high<=low:raise ValueError('Invalid display intensity range')
    return ct,gt,float(low),float(high)


def render(image,mask=None,*,low,high,alpha=.45,color=(30,200,255)):
    grey=np.clip((image.astype(np.float32)-low)/(high-low),0,1)
    rgb=np.repeat(grey[:,:,None],3,axis=2)*255
    if mask is not None:
        region=np.asarray(mask,dtype=bool)
        rgb[region]=(1-alpha)*rgb[region]+alpha*np.asarray(color)
    return np.clip(rgb,0,255).astype(np.uint8)


def load_model(selection=None):
    import torch
    from merit_covid.engine.training import protocol_fingerprint
    from merit_covid.engine.activation_checkpointing import configure_model_execution
    from merit_covid.models.merit_unetrpp import MERITUNETRPP
    from tools.merit_factorized_attention import enable_factorized_mer
    selection=selection or models()[0]
    checkpoint=ROOT/selection['checkpoint']
    if not torch.cuda.is_available():raise RuntimeError('Demo inference needs the local NVIDIA GPU. The CT viewer still works without inference.')
    h=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=selection['sha256']:raise ValueError('Checkpoint changed after registration; choose the correct model file')
    cp=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if cp['epoch']!=selection['epoch'] or cp.get('in_progress'):raise ValueError('Checkpoint epoch differs from registered model')
    if protocol_fingerprint(cp['config'],cp['provenance'])!=cp['protocol_fingerprint']:raise ValueError('Invalid checkpoint metadata')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    # Avoid allocating an extra CPU model copy on the 8-GB local machine.
    with torch.device('meta'):model=MERITUNETRPP(**cp['config']['model'])
    model.load_state_dict(cp['model'],strict=True,assign=True)
    model=model.cuda().eval()
    configure_model_execution(model,cp['config']['training'])
    enable_factorized_mer(model)
    # Smaller chunk bounds memory even if this Windows Torch build uses math SDPA.
    model.mer.query_chunk_size=128
    return model,cp['config']['preprocess']


def cache_file(case_id,master,selection=None):
    selection=selection or models()[0]
    name=case_id+'_'+str(master)+'_fp32.npy'
    # Retain verified existing epoch-0 demo predictions without mixing models.
    legacy=CACHE/name
    if selection['sha256']==EXPECTED_CHECKPOINT_SHA and legacy.exists():return legacy
    return ROOT/'outputs/demo_predictions'/selection['sha256']/'fp32-factorized-q128-v1'/name


def predict(model,preprocess,case_id,ct_path,master,selection=None):
    import torch
    from merit_covid.data.pipeline.cases import Case
    from merit_covid.data.pipeline.config import PreprocessConfig
    from merit_covid.data.pipeline.dataset import CTSlabDataset
    from merit_covid.data.pipeline.dataloader import collate_slabs
    from merit_covid.models.merit_unetrpp import MERITUNETRPPAdapter
    # No ground-truth path is supplied to the model input pipeline.
    case=Case(case_id,Path(ct_path),None,case_id)
    dataset=CTSlabDataset([case],PreprocessConfig(**preprocess),require_mask=False)
    if not 0<=master<len(dataset):raise ValueError('Invalid master slice')
    batch=collate_slabs([dataset[master]])
    batch['image_slab']=batch['image_slab'].cuda()
    with _LOCK,torch.inference_mode():
        logits=MERITUNETRPPAdapter(model).predict_logits(batch)
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite model prediction')
        probability=logits.float().sigmoid()[0,0].cpu().numpy()
    path=cache_file(case_id,master,selection);path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,probability)
    return probability
