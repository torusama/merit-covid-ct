# Đọc source theo luồng một batch CT

Nếu chỉ muốn lấy dữ liệu, bắt đầu từ [ví dụ chạy được](../examples/read_training_batch.py).
API dùng ở code train sau này:

```python
from ct_pipeline import PreprocessConfig, build_fold_loaders
```

`__init__.py` chỉ export các tên đó cùng `CTSlabDataset`, `make_dataloader` và
`FoldDataLoaders`. Chức năng thật nằm ở các file dưới, không viết training loop
trong Dataset hoặc trong tools audit.

## 1. Đọc file theo thứ tự nào?

| Thứ tự | File / hàm nên đọc | Câu hỏi được trả lời |
| --- | --- | --- |
| 1 | [config.py](config.py): `PreprocessConfig` | Dùng bao nhiêu lát? H,W bao nhiêu? Giữ cường độ gốc và giả định đơn vị thế nào? |
| 2 | [cases.py](cases.py): `Case`, `load_cases()` | CT nào đi cùng mask nào? Có patient/group ID hay chỉ volume ID? |
| 3 | [splits.py](splits.py): `validate_splits()` | Danh sách train/validation/test đã lưu có khớp manifest, có trùng nhóm không? |
| 4 | [geometry.py](geometry.py): `inspect_pair()`, `canonical_shape()` | CT/mask có cùng geometry không? Có bao nhiêu lát canonical nếu chỉ đọc header? |
| 5 | [preprocessing.py](preprocessing.py): `read_volume()` | Một volume đi qua kiểm tra, LPS, giữ cường độ và H,W gốc ra sao? |
| 6 | [dataset.py](dataset.py): `CTSlabDataset.__getitem__()` | Một mẫu lấy master nào? Context và mask tương ứng ra sao? |
| 7 | [dataloader.py](dataloader.py): `collate_slabs()`, `build_fold_loaders()` | Ghép B mẫu và lấy đúng partition của một fold thế nào? |
| 8 | [metrics.py](metrics.py): `VolumeEvaluator` | Sau khi có prediction thực, ghép volume và đánh giá ra sao? |

[reporting.py](reporting.py) chỉ phục vụ CSV và biểu đồ mô tả. Các code train sau
này không cần đọc CSV hoặc gọi file này để tạo input.

## 2. Khi gọi build_fold_loaders(), điều gì xảy ra?

```text
build_fold_loaders(manifest, split_path, fold=0, ...)
    ├── CTSlabDataset.from_split(..., partition="train")
    ├── CTSlabDataset.from_split(..., partition="validation")
    └── CTSlabDataset.from_split(..., partition="test")
            ↓ mỗi Dataset
        load_cases() → validate_splits() → chọn case IDs
            ↓
        __init__() đọc headers, tính số master slices, chưa giải mã cả cohort
            ↓
        make_dataloader() gắn batching, seed và collate_slabs
```

Ở đây chỉ đọc split đã có. `create_splits()` chỉ chạy khi người dùng chủ động gọi
lệnh `tools/data_pipeline.py split`, không nằm trong vòng đọc batch.

Hiện `group_id` dùng volume ID. `validate_splits()` có thể ngăn trùng nhóm được
cung cấp, nhưng chưa thể tự biết hai volume có cùng bệnh nhân nếu metadata thiếu
ánh xạ. Đơn vị mm unknown cũng cần giải quyết/chấp nhận giả định trước khi đọc.

## 3. Khi yêu cầu batch đầu, điều gì xảy ra?

```text
next(iter(loaders.train))
    ↓ DataLoader yêu cầu B chỉ số mẫu
CTSlabDataset.__getitem__(global_index)
    1. Tìm case_index và master_slice_index
    2. Nếu đổi ca: gọi read_volume(); nếu cùng ca: dùng cache
    3. Chọn D chỉ số lát, clip chỉ số để lặp lát biên
    4. Tính vị trí/offset từ affine native
    5. Trả một sample dictionary
    ↓
collate_slabs(B samples)
    ↓
Batch dictionary: image_slab, master_mask, z_offsets_mm, IDs, metadata...
```

`global_index` đánh số các master trên toàn Dataset. Ví dụ ca đầu có Z lát thì
index 0…Z−1 thuộc ca đầu; index Z là master 0 của ca kế tiếp. `ends` trong Dataset
là các mốc cộng dồn để làm phép đổi này, không phải danh sách pixel.

Tất cả master đều được lấy, gồm cả GT mask rỗng. Không chọn ROI hoặc bỏ lát dựa
vào GT khi tạo input.

## 4. read_volume() được chia thành sáu bước rõ ràng

| Bước | Hàm/thao tác | Input → output |
| --- | --- | --- |
| 1 | `geometry.inspect_pair()` | Paths → hai ảnh NiBabel; shape/affine/spacing/orientation khớp |
| 2 | `geometry.reorient_to_lps()` cho cả hai ảnh | Giữ ảnh LPS; ảnh LPI đảo Z, cập nhật affine; không nội suy |
| 3 | `get_fdata()` + `validate_voxels()` | Scaling NIfTI → mảng float32 X,Y,Z, finite và mask 0/1 |
| 4 | `normalize_ct()` | Mặc định `native`: trả nguyên cường độ sau NIfTI scaling, không window/normalize |
| 5 | `to_slice_tensor()` + `resize_in_plane()` | X,Y,Z → Z,Y,X; `size_hw=None` bỏ qua nội suy, giữ H,W |
| 6 | `geometry_metadata()` và thông tin intensity | Metadata đủ để truy vết và restore về ảnh gốc |

Ví dụ A-0025: original LPS, 512×512×61 → standardized LPS, giữ nguyên voxel/affine.
`to_slice_tensor()` đưa mảng thành [61,512,512]; đây chỉ là quy ước thứ tự tensor.
Không resize: tensor giữ [61,512,512]. Dataset chọn 11 lát, thêm channel, thành
[1,11,512,512]. Master mask có shape [1,512,512].

`intensity_encoding="native"`, `hu_window=None` giữ cường độ CT sau scaling
NIfTI. Tên helper `normalize_ct()` giữ để tương thích code, nhưng nhánh native
trả nguyên mảng, không chuẩn hóa hay suy đoán HU. CT không bắt buộc ở [0,1].
Chỉ khi caller chọn `hu`/`auto_hu` và truyền window cụ thể mới clip/normalize;
window không được âm thầm áp dụng vào chế độ native. PNG dùng thang xám tự động
chỉ để hiển thị; không làm thay đổi batch.

Hai thao tác khác nhau cần nhớ:

- **Reorient:** thống nhất hướng và thứ tự trục, CT/mask và affine cùng thay đổi.
  Không tự làm thẳng ảnh oblique; không tạo thêm lát.
- **Resize XY:** mặc định tắt (`size_hw=None`). Chỉ chạy khi caller truyền H,W
  cụ thể; khi đó CT bilinear, mask nearest-exact. Giữ helper này để tùy chọn
  resize được khai báo rõ, không có bước resize ngầm trong Dataset.

Các tên `canonical_*` nay chỉ grid LPS đã thống nhất. `slice_positions_mm` vẫn
là tọa độ world RAS+ của NiBabel; không được đổi dấu tọa độ chỉ vì mảng là LPS.

## 5. Đọc batch trong code train sau này

```python
batch = next(iter(loaders.train))
inputs = batch["image_slab"]       # [B,1,D,H,W], float32
targets = batch["master_mask"]     # [B,1,H,W], float32 0/1
offsets_mm = batch["z_offsets_mm"] # [B,D], float32
```

Đây là vị trí nối model ở giai đoạn sau. `targets` chỉ tương ứng master ở tâm
slab, không có chiều D. Một batch có thể chứa các master cùng hoặc khác volume.
`case_id` và `master_slice_index` giúp biết chính xác mỗi mẫu thuộc đâu.

Metadata là list B dictionary vì geometry gốc giữa các ca có thể khác. CT/mask
gốc không bị ghi đè. `num_workers=0` dùng tiến trình chính; khi tăng workers cần
guard `if __name__ == "__main__":` như trong ví dụ để chạy Windows đúng cách.

## 6. Sau khi có prediction, nó quay về volume gốc thế nào?

`metrics.VolumeEvaluator` nhận cặp `(master_slice_index, probability_mask)` cho
từng ca. `SliceReconstructor` báo lỗi nếu thiếu/trùng/sai shape. Sau khi đủ lát,
`geometry.restore_volume()` đưa về X,Y,Z gốc bằng inverse orientation; mặc định
không cần nội suy vì H,W được giữ nguyên. Nếu chủ động bật resize lúc đọc, hàm
mới resize prediction về native trước khi đảo orientation.
Evaluator threshold sau khi restore probabilities và tính metric trên toàn volume.
Không dùng mean Dice từng lát thay cho Dice volume.

## 7. Chỗ sửa cấu hình và tương thích code cũ

- Đổi một lần cấu hình ở `PreprocessConfig` hoặc đối tượng truyền vào
  `build_fold_loaders()`, dùng cùng cấu hình cho mọi partition.
- Đổi phép biến đổi CT ở `preprocessing.py`; đổi cách cắt context ở `dataset.py`.
- Đổi batching/worker/shuffle ở `dataloader.py`; validation/test không shuffle.
- Các import cũ từ `ct_pipeline.dataset` vẫn chạy cho `PreprocessConfig`,
  `read_volume`, `restore_volume`, `affine_mm`, `inspect_pair`, `make_dataloader`,
  `collate_slabs`, `seed_worker`. Đây là alias sang module mới, không phải hai
  bản implementation khác nhau.

Quay lại [README chính](../README.md) để có lệnh PowerShell và hợp đồng toàn bộ
trường của batch. [Hướng dẫn chi tiết](../docs/data_pipeline.md) giữ quy ước
intensity, empty-mask metrics và định dạng prediction cho CLI evaluate.
