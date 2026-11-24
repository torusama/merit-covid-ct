# B2 — QC Tiling128 trên bộ CT/GT đã precompute V3

## 1. Phạm vi

Công cụ độc lập: `tools/qc_tiling128.py`. Chỉ chạy CPU để kiểm tra dữ liệu và
chia–ghép tile. Không tạo model, không train, không sửa Dataset/DataLoader,
loss, checkpoint hoặc pipeline production. B3 chưa được thực hiện.

Đầu vào là **bản xuất precomputed V3**, không phải CT gốc chưa chuẩn hóa.
`intensity_encoding: native` trong YAML của bộ xuất có nghĩa **giữ nguyên giá
trị CT đã V3**. Công cụ dùng `read_arrays` hiện tại nhưng khóa `size_hw: null`,
không clipping/Z-score lần nữa, không resize và không nội suy.

## 2. Cách chia và khôi phục

- Ảnh `512×512` → 4 hàng × 4 cột → 16 tile `128×128`, không overlap/halo.
- Array do reader trả về là `[Z,Y,X]`; **row đi theo Y, col đi theo X**.
- Tọa độ bắt đầu từ 0 và là khoảng nửa mở: tile `(row=2,col=1)` có
  `y0=256,y1=384,x0=128,x1=256`.
- CT và GT cùng `master_z` được crop bằng chính các tọa độ đó. Stitch đặt tile
  về vị trí cũ; không thay giá trị hoặc ép kiểu khi chia–ghép.
- Với mọi lát của mọi ca, công cụ ghép lại ảnh đầy đủ và so sánh **từng voxel**
  CT/GT; CT float32 được kiểm cả bit pattern. Không thay phép so equality bằng
  shape hoặc checksum.
- `tiles_lesion` là số **tile của master slice** chứa ít nhất một pixel GT=1;
  `tiles_empty` là số tile master không chứa lesion. Đây chưa phải phân bố sau
  sampling/balancing, và không đếm lặp toàn bộ 11 lát context.

## 3. Slab 11 lát ở biên Z

Dùng đúng `pipeline/slab.py:sample_arrays`, `index`, `edge`, `unique`.
Kiểm các master đầu/cuối, gần biên 5 lát và giữa volume; trùng master được bỏ.
`edge` vẫn tạo đủ 11 vị trí input bằng clamp ở biên. `unique` đánh dấu mỗi lát
nguồn chỉ có một đại diện hợp lệ; **không có nghĩa luôn có 11 lát nguồn khác nhau**.

Kiểm độc lập chỉ số Z, centre index, validity, từng crop CT/GT của từng context
slot và ghép lại slab. GT của slot lấy từ đúng lát nguồn đó; không dùng GT master
lặp cho các lát khác. Việc lặp GT tại các slot cùng chỉ số Z do edge là hợp lệ.

## 4. Dữ liệu và các cổng kiểm tra

Thư mục dữ liệu cần có:

```text
<DATA_ROOT>/
  EXPORT_COMPLETE.json
  cases.json
  splits_covid19_20_5fold.json
  merit_precomputed_v3.yaml
  <các file CT/GT được manifest tham chiếu>
```

Mọi đường dẫn CLI tương đối được tìm từ `--data-root`; đường dẫn CT/GT trong
manifest giữ quy tắc production là tương đối với chính file manifest.
Không cứng hóa tên dataset/account Kaggle. `--output-dir` phải là thư mục **mới**,
nằm ngoài cây dữ liệu. Công cụ không ghi đè báo cáo cũ.

Kiểm tra:

1. Manifest/export đủ 199 ID; filename, SHA-256 và manifest fingerprint khớp.
2. Split qua validator production: đủ 5 fold, không trùng volume/group giữa các
   partition; fold0 đúng TRAIN127/VAL32/TEST40. Không chia lại split.
3. NIfTI shape, affine, orientation, spacing, CT hữu hạn, GT nhị phân chính xác
   trước khi reader ép float32. CT precomputed phải lưu float32, slope=1/intercept=0.
4. Mọi voxel CT/GT sau tiling roundtrip, lesion count và slab như mục 2–3.
5. Hash CT/GT trước/sau giống nhau và khớp bộ xuất; hash metadata đầu vào cũng
   không đổi. Không sửa hoặc “sửa hộ” một ca lỗi.

Giới hạn cần giữ: split theo volume/group không xác minh danh tính bệnh nhân.
Nếu exporter ghi `COMPLETE_WITH_PROVENANCE_LIMITATION`, báo cáo B2 giữ nguyên
giới hạn đó. B2 không chứng minh bản xuất V3 giống voxel/byte tuyệt đối với dữ
liệu gốc chưa được đối chiếu.

## 5. Lệnh local

Tại repo, dùng môi trường có `numpy`, `nibabel`, `PyYAML`; chạy test cần thêm
`pytest`. Không cần Torch/CUDA cho công cụ B2.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_tiling128_qc.py -q --basetemp outputs/b2_test_tmp_NEW

# Giới hạn một ca: báo 1/199 tested, chưa qua B2 gate.
.\.venv\Scripts\python.exe tools/qc_tiling128.py --data-root C:\1_HDD\paper\material\merit_v3_precomputed_nii --manifest cases.json --split splits_covid19_20_5fold.json --output-dir outputs/b2_A0003_NEW --case-id volume-covid19-A-0003

# Đủ 199 ca; đổi tên output nếu đã từng chạy.
.\.venv\Scripts\python.exe tools/qc_tiling128.py --data-root C:\1_HDD\paper\material\merit_v3_precomputed_nii --manifest cases.json --split splits_covid19_20_5fold.json --output-dir outputs/b2_all199_NEW
```

Nếu Python hệ thống thiếu dependency, dùng `.venv` đang có thay vì coi đó là lỗi
dữ liệu. Với bản bundle tách biệt, chạy `python -m pip install -r requirements_b2.txt`
trong `project/` khi môi trường chưa có các thư viện này.

## 6. Chạy trên Kaggle

Thêm dataset V3 bằng **Add Input**. Đưa source repo hoặc giải nén bundle B2 vào
`/kaggle/working/`; không cần bật GPU. Đặt hai biến đường dẫn theo Input thực tế:

```python
from pathlib import Path
import subprocess, sys

PROJECT = Path('/kaggle/working/b2_bundle/project')  # nơi đã giải nén source B2
DATA_ROOT = Path('/kaggle/input/<dataset-v3-cua-ban>/<thu-muc-chua-cases.json>')
OUTPUT = Path('/kaggle/working/b2_qc199_run01')  # tên mới, chưa tồn tại

required = ['cases.json', 'splits_covid19_20_5fold.json',
            'merit_precomputed_v3.yaml', 'EXPORT_COMPLETE.json']
missing = [str(DATA_ROOT / p) for p in required if not (DATA_ROOT / p).is_file()]
if missing:
    raise FileNotFoundError('Chon dung thu muc V3; thieu: ' + ', '.join(missing))

subprocess.run([sys.executable, str(PROJECT / 'tools/qc_tiling128.py'),
    '--data-root', str(DATA_ROOT), '--manifest', 'cases.json',
    '--split', 'splits_covid19_20_5fold.json', '--output-dir', str(OUTPUT)],
    cwd=PROJECT, check=True)
```

Đây là lệnh QC, không chạy nội dung `training.epochs:100` trong YAML.
Không tải dataset/checkpoint trong script. Đọc báo cáo và tải output từ Kaggle
sau khi hoàn tất; không ghi kết quả vào `/kaggle/input`.

## 7. Đọc kết quả

| File | Nội dung |
| --- | --- |
| `tiling128_qc_199.json` | Trạng thái/gate, số ca thật, số voxel so sánh, lỗi từng ca, tổng theo partition, geometry/slab details, input và source hashes/phiên bản code |
| `tiling128_qc_199.csv` | Một hàng cho mỗi ca đã chọn kiểm, trạng thái/lý do và số tile/lesion/mismatch/hash/thời gian |
| `tiling128_tile_metadata.jsonl.gz` | Mỗi dòng một tile: `case_id,partition,master_z,tile_row,tile_col,x0,x1,y0,y1,lesion_pixels`; không chứa tensor CT/GT |

Tọa độ tile thuộc canonical LPS `[Z,Y,X]`. Geometry trong hàng JSON của từng
volume giữ affine/orientation và phép quay về hệ ảnh nguồn; B3 phải dùng chúng
khi ghép/xuất inference. Metadata tile riêng không thay thế affine.

- **PASS + `b2_gate_pass:true`:** đủ 199 ca và không ca lỗi.
- **PARTIAL_PASS:** tập con đã chọn đạt; ví dụ 1/199 tested vẫn chưa qua gate.
- **FAIL:** có ca hoặc kiểm tra toàn cục lỗi; không kết luận đủ 199 PASS.
- **BLOCKED_INPUT:** chưa đủ điều kiện đọc dữ liệu, báo rõ input cần bổ sung.

Số tile/lesion trong tổng hợp chỉ lấy các ca PASS, ghi rõ ở tên trường. Khi có
lỗi, không dùng các dòng metadata đã ghi dở để nối model. `ct_mismatched_bits`
là số pixel có **bit pattern float32 khác**, không phải số bit riêng lẻ bị đổi.
Thời gian báo cáo gồm kiểm input, đọc/hash, tiling, slab và hash sau; không gồm
thời gian đóng gói ZIP hoặc người dùng giải nén/cài dependency.

Có thể đọc metadata mà không giải nén hết:

```python
import gzip, json
with gzip.open(OUTPUT / 'tiling128_tile_metadata.jsonl.gz', 'rt') as f:
    first_tile = json.loads(next(f))
print(first_tile)
```

## 8. Sau B2: chỉ đề xuất B3, chưa thực hiện

Sau khi xem báo cáo và được bạn xác nhận, B3 cần thiết kế thay đổi tại:

- `data/pipeline/dataset.py`, `dataloader.py`: index `(volume,master_z,tile)` và
  batch metadata, giữ nguyên split volume.
- Cấu hình model/input và `models/merit_unetrpp/`: kiểm khả năng nhận grid128,
  position/feature shapes và tương thích checkpoint; không tự coi model512 dùng
  nguyên trọng số là chạy đúng trên grid128.
- `models/merit_unetrpp/loss.py` cùng nhãn phụ: presence/salience theo đúng GT
  của tile/context; không gán nhãn tổn thương toàn volume cho mọi tile.
- `engine/` inference/reconstruction: phủ đủ 16 tile của mọi master rồi ghép
  về512/native geometry trước metric; kiểm lại GT leakage.
- Tests và config mới cho train/inference tiling; quyết định sampling/balancing,
  augmentation/overlap/halo là công việc sau, chưa thuộc B2.

**B2 đạt chỉ chứng minh thao tác dữ liệu bảo toàn, chưa chứng minh model có thể
train grid128, checkpoint tương thích hoặc Dice/IoU tốt hơn.**
