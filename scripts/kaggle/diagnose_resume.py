"""Bounded replay of the failed Kaggle continuation with numerical tracing."""
import traceback
from scripts import train_colab
from scripts.kaggle.resume_checkpoint import verify_dataset_content, rebind_verified_timestamps

checkpoints = list(input_root.glob('**/in_progress.pt'))
if len(checkpoints) != 1:
    raise RuntimeError(f'Expected saved checkpoint, found {checkpoints}')
checkpoint = checkpoints[0]
torch.set_num_threads(2)
saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
print('Saved progress:', {k:v for k,v in saved['in_progress'].items() if k != 'epoch_generator'}, flush=True)
bad_weights = [n for n,t in saved['model'].items() if t.is_floating_point() and not torch.isfinite(t).all()]
print('Nonfinite saved model tensors:', bad_weights, flush=True)
del saved
original_fit = train_colab.Trainer.fit

def fit_with_diagnostics(self, **kwargs):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if saved['protocol_fingerprint'] != self.fingerprint:
        print('Checking all dataset bytes before timestamp migration', flush=True)
        verify_dataset_content(cases, DATA_HASHES)
        migrated = rebind_verified_timestamps(saved, self.config, self.provenance, self.fingerprint)
        resume_path = Path('/tmp/verified-resume.pt')
        torch.save(migrated, resume_path)
        print('Verified timestamp migration completed', flush=True)
        del migrated
    else:
        resume_path = checkpoint
    del saved
    def trace(name):
        def hook(module, args, result):
            values = result if isinstance(result, (tuple, list)) else [result]
            for value in values:
                if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                    message = f'FIRST NONFINITE: {name} ({type(module).__name__}), output={value.dtype}, shape={list(value.shape)}'
                    print(message, flush=True)
                    raise FloatingPointError(message)
        return hook
    for name, module in self.model.named_modules():
        if not list(module.children()):
            module.register_forward_hook(trace(name))
    original_collate = self.train_loader.collate_fn
    calls = 0
    def collate(samples):
        nonlocal calls
        calls += 1
        batch = original_collate(samples)
        if calls > 1950:
            image = batch['image_slab']
            print('BATCH', calls, batch['case_id'], 'master', batch['master_index'].tolist(),
                  'range', float(image.min()), float(image.max()), flush=True)
        return batch
    self.train_loader.collate_fn = collate
    return original_fit(self, resume=resume_path, checkpoint_every=5, max_seconds=900)

train_colab.Trainer.fit = fit_with_diagnostics
try:
    train_colab.train(config, 0, output / 'diagnostic-checkpoints', epochs=1)
    result = {'status':'PASS_BOUNDED_REPLAY'}
except Exception as error:
    result = {'status':'REPRODUCED_ERROR', 'error':str(error), 'traceback':traceback.format_exc()}
    print(result['traceback'], flush=True)
(output / 'diagnosis.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
