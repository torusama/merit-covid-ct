# B3B — bàn giao Input/Output MERIT với V3 Tiling128 fold 0

## 1. Bản source và phạm vi

Thực hiện trên **`train/merit-unetrpp-colab`**, HEAD ban đầu
`bb48650cb418147cfac48d6b110c0b3ba1798dd2`. Đã fetch origin ngày 24/11/2026:
local/upstream trùng nhau, ahead/behind 0/0. Không đổi nhánh, không reset/stash.
Hồ sơ kiểm thử ghi nhận giai đoạn trước commit; sau đó người dùng yêu cầu công
bố B3B lên chính nhánh train. Năm file demo có sẵn được giữ nguyên.

B3B bổ sung đường đọc tile đã xuất, train/validation/resume, inference ghép
xác suất và evaluation volume gốc. **Chưa train chính thức, chưa chạy CV,
chưa benchmark GPU/VRAM, chưa có bằng chứng Dice tốt hơn.** Không triển khai B4/B5.

## 2. Dataset cần gắn

Dùng **Notebook Output đã Save Version** chứa thư mục
`TILING128_V3_FOLD0_NIFTI`. Không dùng RAW, V3 full512 hoặc NPY pilot B3A.
Không cần tải 199 raw CT. Không ghi gì vào thư mục input.

Thư mục gốc phải chứa `READY_ALL_199.json`, `TILE_MANIFEST.json`,
`source_metadata/` và `cases/<original_case_id>/case_export.json` cùng 16 cặp
NIfTI CT/GT cho mỗi ca. Reader đối chiếu schema với mã exporter nhúng trong
notebook `KAGGLE_NOTEBOOK_XUAT_DATASET_V3_DA_CHUAN_HOA_VA_CAT_TILING128_FOLD0.ipynb`
của người dùng; không dùng schema NPY của B3A.

Trong Kaggle, chọn **Add Input → Notebook Outputs**, sau đó tìm root:

```python
from pathlib import Path
roots = [p.parent for p in Path('/kaggle/input').rglob('READY_ALL_199.json')
         if (p.parent/'TILE_MANIFEST.json').is_file()]
if len(roots) != 1:
    raise RuntimeError(f'Chọn chính xác tile_root; tìm được: {roots}')
tile_root = roots[0]
import os
os.environ['MERIT_TILE_ROOT'] = str(tile_root)
print(tile_root)
```

Không hardcode account hoặc slug. Cũng có thể truyền `--tile-root /đường/dẫn`
cho preflight, io-smoke, check-split và train. `tile_root` được resolve vào config
và checkpoint; các lần resume/evaluate phải truy cập cùng mount/path đã khóa.

## 3. Dữ liệu đi vào mô hình như thế nào

- Vẫn **199 volume gốc**, split fold0: **127 TRAIN / 32 VAL / 40 TEST**.
  Fingerprint được kiểm trên tên CT/GT **volume nguồn** trong manifest gốc,
  trước khi mở rộng index. Không tạo split theo 3.184 tile.
- Chỉ chạy **fold 0**. Fold 1–4 và lệnh `cv` bị chặn vì V3 đã fit bằng TRAIN
  fold0; dùng các fold khác sẽ rò thống kê preprocessing.
- Tổng index: **13.705 master × 16 tile = 219.280 mẫu**, gồm cả tile nền.
  Mỗi mẫu là `(original_case_id, tile_id, master_z)`; `tile_id=row*4+col`,
  row là Y, col là X. Trong mỗi block volume, index theo tile rồi master Z.
- CT đã V3: **native pass-through**, `size_hw: null`, không clip/Z-score lại,
  không resize, không cắt lại XY, không resample Z. CT file `float32`; GT file
  `uint16` nhị phân, được chuyển sang `float32` khi tính loss như pipeline cũ.
- Chọn đúng 11 lát từ **cùng tile**, edge padding và unique duplicate policy.
  GT phụ trợ là GT thật của 11 lát tile, không lặp GT master.
  Trước batch `[1,11,128,128]`; sau batch `[B,1,11,128,128]`.
- `ends` vẫn có một block cho mỗi volume, dài `16×Z`; volume shuffle và
  volume-balanced sampler tiếp tục nhóm theo **volume gốc**, không theo tile.
  Config giữ `samples_per_volume=0`, đi đủ mọi mẫu, seed 42, volume shuffle.

Cache mới giữ **một cặp tile 128×128×Z mỗi worker**, không mở 6.368 file vào RAM
cùng lúc. Volume shuffle vẫn xáo mọi mẫu trong volume nên có thể giải nén lại
tile thường xuyên. Đây là giới hạn hiệu năng đã công bố; B3B không tự đổi sampler
để tăng tốc và không tuyên bố nhanh hơn Native512. RAM còn bao gồm slab/batch,
prefetch và một volume full512 khi inference/evaluate.

API DataLoader riêng, không đổi API Native512:

```python
from merit_covid.data.pipeline.config import PreprocessConfig
from merit_covid.data.pipeline.dataloader import build_tiled_fold_loaders
loaders = build_tiled_fold_loaders(
    tile_root, fold=0, preprocess=PreprocessConfig(assume_mm=True),
    batch_size=1, num_workers=0, seed=42)
batch = next(iter(loaders.train))
```

`assume_mm: true` giữ giả định đơn vị đã có của dự án; không xác minh lại đơn vị
thiếu và không sửa NIfTI. Provenance và giới hạn danh tính bệnh nhân vẫn được giữ.

## 4. Mô hình và checkpoint

Config mới: `configs/merit_tiling128_v3_fold0.yaml`. So với Native512, khác
nguồn input và `model.img_size=[11,128,128]`. Giữ feature_size 16,
depths `[3,3,3,3]`, TEP/MER/PCG, loss weights, optimizer, augmentation,
sampling policy và các setting train khác của config Native512.

**Không sửa file kiến trúc hay loss.** Code hiện có đã dựng grid phù hợp từ
`img_size`. Positional embeddings và ma trận E/F của EPA phụ thuộc số token,
nên hình dạng một số weights và số tham số của grid128 khác grid512.
Checkpoint Native512 **không được partial-load/resume** sang protocol tile.
B3B không chuyển đổi weights cũ và không khẳng định đã khởi tạo từ epoch0 cũ.

Resume khóa config, fold, split, SHA manifest, SHA metadata từng case, danh tính
CT/GT **của toàn bộ 16 tile**, file size/mtime và SHA code model/loss/reader/trainer.
Không tính SHA 6.368 file mỗi `__getitem__` hay epoch. Muốn xác minh bytes, chạy
preflight `--hash-files` một lần trước train. Size/mtime không thay thế hash byte.

Giữ cùng source và dataset/mount khi resume. Nếu remount làm timestamp khác,
reader sẽ từ chối để tránh tiếp tục sai dữ liệu; chưa có cơ chế tự rebind mount
trong B3B. `--verified-resume-hashes` của đường RAW cũ bị chặn cho tile vì nó
chỉ xác minh một file mỗi volume, không đủ cho 16 tile. Không sửa checkpoint để
lách guard. Giữ mount ổn định hoặc bàn riêng quy trình rebind có xác minh đầy đủ.

Checkpoint tile nạp không dùng mmap để tránh khóa file đang cần thay/xóa trên
Windows; cách load Native512 cũ giữ nguyên. Mọi thử nghiệm B3B dùng output mới.

## 5. Inference và evaluation

Inference chỉ mở CT. Với mỗi master Z, ghép đúng 16 probability `[128,128]`
thành `[512,512]`, rồi thành `[Z,512,512]`. Không threshold trước ghép, không
resize ô lên 512, không lấy trung bình 16 tile Dice. Thiếu/trùng tile, sai case,
crop, Z, shape hoặc NaN/Inf đều bị từ chối.

`VolumePrediction` nhận **geometry full-volume**, kiểm quan hệ affine giữa
full LPS và original CT, rồi `to_original_xyz()` khôi phục grid gốc bằng phép
hoán vị/lật trục. Tile affine dịch chuyển chỉ dùng cho sample tile.

Evaluation mới đọc 16 GT tile, ghép GT về full512 và khôi phục grid gốc.
Dice/IoU/HD95, threshold 0,5, empty-mask policy và group macro aggregation dùng
evaluator hiện có. Mỗi hàng kết quả là **một original_case_id**, không phải tile.
VAL có 32 hàng, TEST có 40 hàng; nếu dùng API diagnostic cho toàn cohort thì
có 199 hàng. Không dùng TEST chọn threshold/hyperparameter.

Presence/salience trong loss vẫn dùng GT tile cục bộ. **Whole-slice presence và
calibration chưa hỗ trợ**: không cộng/max tile presence để giả làm xác suất
whole-slice. Báo cáo ghi `UNSUPPORTED`, không xuất metric presence sai nghĩa.

## 6. Các lệnh bàn giao

Chạy từ root repository, trong môi trường đã cài dependencies của dự án:

```bash
python -m pip install -e ".[unetrpp,test]"
python -m scripts.merit_pipeline check-split --config configs/merit_tiling128_v3_fold0.yaml
```

Preflight đọc header toàn bộ dataset; tùy chọn hash và kiểm voxel đầy đủ sẽ tốn
I/O hơn. Không tự tải dữ liệu. Các file report dùng chế độ tạo mới, không ghi đè:

```bash
python -m scripts.merit_pipeline preflight --config configs/merit_tiling128_v3_fold0.yaml --hash-files --voxels --output /kaggle/working/b3b_preflight_new.json
python -m scripts.merit_pipeline io-smoke --config configs/merit_tiling128_v3_fold0.yaml --device cuda --case-id volume-covid19-A-0003 --output /kaggle/working/b3b_smoke_new.json
```

`io-smoke` chỉ một forward/loss/backward của **đúng config**, không optimizer
update. Nếu CUDA/RAM không đủ, báo lỗi; không giảm ảnh/model để chạy tiếp.
`--case-id` ở preflight cho phép kiểm một ca, nhưng report ghi rõ QC subset.

**Các lệnh train sau chỉ để thành viên phụ trách thực thi; Codex chưa chạy:**

```bash
python -m scripts.merit_pipeline train --config configs/merit_tiling128_v3_fold0.yaml --fold 0 --output /kaggle/working/merit_tile_fold0_run_NEW --checkpoint-every 100
python -m scripts.merit_pipeline train --config configs/merit_tiling128_v3_fold0.yaml --fold 0 --output /kaggle/working/merit_tile_fold0_run_NEW --checkpoint-every 100 --resume /kaggle/working/merit_tile_fold0_run_NEW/in_progress.pt
python -m scripts.merit_pipeline evaluate --checkpoint /kaggle/working/merit_tile_fold0_run_NEW/best.pt --partition validation --device cuda --output /kaggle/working/tile_val_NEW.json
python -m scripts.merit_pipeline evaluate --checkpoint /kaggle/working/merit_tile_fold0_run_NEW/best.pt --partition test --device cuda --output /kaggle/working/tile_test_NEW.json
```

Giữ `num_workers=0` nếu cần exact mid-epoch resume. RNG Python/NumPy/Torch/CUDA,
loader generator, vị trí batch, optimizer/scheduler/scaler được lưu. Epoch-boundary
resume dùng `last.pt`. Output trên `/kaggle/working` cần Save Version/copy ra nơi
lưu bền vững; B3B không thêm cơ chế cloud backup hoặc cam kết phục hồi file đã mất.

## 7. Kiểm thử và điều kiện trước training

Đọc `B3B_TILING128_TEST_REPORT.json` để biết kết quả thực cuối cùng, XML, thời
gian và lỗi regression. Dataset NIfTI tile thật của Kaggle **chưa có trên máy**;
không tự cắt dataset full512 ở máy để giả làm kiểm thử tile thật.

Synthetic có full512, Z=1/3/13, flip/permutation, affine tịnh tiến/shear, CT-only,
GT local, sampler/worker/resume, negative cases và forward/backward. Test counts
199/3184/3184/13705/219280 chỉ kiểm **metadata synthetic**; không phải audit voxel
của 199 CT Kaggle. Unit test dùng feature_size2/depths1 để gọn. Ngoài ra đã chạy
**một forward/loss/backward CPU với đúng model cấu hình bàn giao**, feature_size16,
depths3 và đủ TEP/MER/PCG, input `[1,1,11,128,128]`: loss/gradient finite,
0 optimizer update, khoảng 5,43 giây. Đây là smoke synthetic, không phải benchmark
hiệu năng hoặc bằng chứng train tốt. CUDA và tile A-0003 thật chưa kiểm chứng.

Riêng **44 kiểm thử B3B đạt**: 43 unit/integration và một smoke CLI chạy process
con thật (1 update synthetic, dừng trước hết epoch, rồi evaluate đầy đủ 2 volume
VAL synthetic). Process con chỉ nới bộ đếm cohort trong test fixture 10 ca;
config production vẫn bắt buộc 199/127/32/40. Không phải đánh giá 32 CT thật.
Toàn repository chưa xanh: có lỗi có sẵn ở notebook,
fixture/tài liệu và Native resume Windows. Đã tái hiện các lỗi này ở commit nền;
xem từng lần chạy XML trong report thay vì coi regression toàn bộ là PASS.

Người nhận chạy preflight đầy đủ và một io-smoke đúng config trên input đã mount
trước training. Nếu chúng FAIL, giữ log và dừng; không sửa normalization, loss,
split hoặc tự giảm model. Không có yêu cầu chạy QUICK100/full epoch trong B3B.

## 8. Đọc code theo thứ tự

1. `configs/merit_tiling128_v3_fold0.yaml`: protocol riêng.
2. `data/pipeline/tiled.py`: bundle/geometry/split/reader/index/cache/GT reconstruction.
3. `data/pipeline/dataloader.py`: loader riêng, giữ sampler original-volume.
4. `engine/training.py`: dispatch/guards/source binding/checkpoint.
5. `engine/tiled_inference.py`: CT-only → probability stitch → volume metrics.
6. `engine/tiled_smoke.py`, `scripts/merit_pipeline.py`, `tests/test_tiled_pipeline.py`.

Các đường `data/pipeline/` và `engine/` ở trên nằm dưới `src/merit_covid/`.
Diff B3B và file kiểm thử được đóng gói riêng, không chứa dữ liệu hoặc checkpoint.

## 9. Gói source để nhận bàn giao

ZIP nằm tại `outputs/b3b_20261124/B3B_TILING128_HANDOFF_20261124.zip`.
`source/` là snapshot commit nền cộng B3B, có thể cài dependencies và chạy từ đó;
`changes/` chỉ chứa file B3B; `B3B_TRACKED.patch` là diff của các file đã có.
Không có CT, GT, slab, model weights hoặc checkpoint trong ZIP.

Để nhận B3B qua GitHub, dùng nhánh **`train/merit-unetrpp-colab`** và kiểm tra có
`configs/merit_tiling128_v3_fold0.yaml`, file hướng dẫn này và báo cáo kiểm thử.
ZIP là snapshot trước công bố, được giữ nguyên để đối chiếu. Các trường
`head_after` và `commit_or_push_performed` trong báo cáo mô tả lúc kiểm thử,
không phải trạng thái GitHub hiện tại. Các sửa demo cũ chưa commit được giữ
trên máy nhưng không trộn vào bản công bố B3B.
Không giải nén `source/` đè lên một checkout đang có sửa riêng. Nếu cần áp vào
checkout nền, kiểm `git apply --check B3B_TRACKED.patch` trước khi áp patch và
đưa các file mới tương ứng vào; không discard thay đổi chưa commit.
