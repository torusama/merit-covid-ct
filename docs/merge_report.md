# Tích hợp main và Duong trên nhánh MERGE

Ngày: 07/11/2026. Nhánh mới bắt đầu từ `main` tại `4fa8753`, tích hợp `Duong`
tại `5127e52`. Hai nhánh nguồn không bị sửa. Merge commit giữ cả hai lịch sử.

## Phạm vi đã thống nhất

Giữ cấu trúc package `src/merit_covid/`, cách cài editable, cấu hình YAML,
scripts, downloader, viewer, model adapter và tài liệu nền của main. Chuyển
logic xử lý dữ liệu đã kiểm chứng của Duong vào package đó:

- Hướng mảng LPS; không đổi giá trị voxel khi đảo/hoán vị trục.
- Mặc định giữ H,W gốc và cường độ CT sau scaling NIfTI; không window/normalize.
- Giữ số lát Z; slab mặc định 11 lát, master ở vị trí 5; lặp lát biên.
- Đơn vị unknown bị từ chối mặc định. `--assume-mm`/`assumed_unit="mm"` chỉ là
  giả định đã thống nhất cho ba ca thiếu đơn vị, không phải xác minh nguồn.
- Giữ nguyên file split Duong và danh sách ID trong mọi fold.
- Giữ quy ước evaluator Duong: metric toàn volume, trung bình trong group rồi
  macro các group; không trung bình điểm từng lát.

Không có model, training, checkpoint hoặc thí nghiệm mới. Dataset gốc không bị
ghi lại; không tải hoặc phân tích thêm external dataset.

## Một nguồn xử lý, các giao diện tương thích

`src/merit_covid/data/pipeline/` chứa implementation chuyển từ Duong. Các module
main ở `data/nifti_io.py`, `preprocessing.py`, `slab.py`, `splitting.py` và
`engine/` gọi các hàm chung đó. Các file `.py` trong `ct_pipeline/` chỉ chuyển
tiếp import để ví dụ/notebook cũ vẫn chạy; không chứa thuật toán xử lý thứ hai.

| Giao diện | Input/target | Chỉ số master |
| --- | --- | --- |
| `from merit_covid import build_fold_loaders` | `image_slab [B,1,D,H,W]`, `master_mask [B,1,H,W]` | `master_slice_index`: chỉ số lát trong volume |
| `merit_covid.data.dataset.SlabDataset` | `image [1,D,H,W]`, `mask [1,H,W]` trước batching | `master_index`: vị trí trong slab; `slice_index`: lát trong volume |

Hai giao diện gọi cùng core; không đổi tên `master_index` thành
`master_slice_index` vì chúng có ý nghĩa khác nhau. NIfTI/PreparedVolume dùng
X,Y,Z; tensor dùng Z,Y,X, H=Y, W=X. Đường ghép volume của main được cập nhật để
nhận mask lát theo H,W rồi trả về X,Y,Z trước khi đảo orientation.

API NumPy của main giữ inference không đọc GT và giữ exporter về NIfTI gốc.
PyTorch được import khi dùng Tensor/DataLoader hoặc chủ động resize. Worker
Windows không nhận cache volume đã tải. Cache cũ được giải phóng trước ca mới.

## Split duy nhất

File đang dùng: `splits/covid19_20_5fold.json`, giữ nguyên từng byte từ Duong.
File main cũ `splits/covid19_20/folds_seed42.json` được bỏ để tránh hai protocol.

| Fold | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 0–3, mỗi fold | 127 | 32 | 40 |
| 4 | 128 | 32 | 39 |

API `data.splitting.load_splits()` của main trả view có các list `train`,
`validation`, `internal_test`, nhưng đọc/ghi cùng JSON canonical của Duong.
View và canonical được đối chiếu khi validate/save. Không tạo split 60/20/20 mới.

Patient identity hiện chưa xác minh. `group_id` không được coi là patient ID.
Giữ group trong manifest; khi có patient mapping đáng tin cậy phải tạo split mới
theo mapping đó. Manifest versioned do main xuất có thể dùng với API batch Duong.

## Giải quyết xung đột và kiểm thử

Git báo xung đột ban đầu ở README, requirements và ba file tests trùng tên.
Giữ tests main tại `tests/`; giữ nguyên suite Duong tại `tests/duong/`, với fixture
riêng. Tests main được cập nhật theo các quyết định LPS, H/W, window tắt và quy
ước metric Duong; các kiểm tra hình học, nhãn, inference và lỗi vẫn được giữ.

Các kiểm chứng tích hợp:

1. Trước merge lưu sáu slab thật của A0025 (LPS) và A0161 (LPI), tại master
   đầu/giữa/cuối. Sau tích hợp đối chiếu cả hai API: CT, mask, indices, offsets,
   vị trí world và metadata khớp chính xác bản Duong trước merge.
2. Tests dùng ảnh không vuông để phát hiện đảo H/W, kiểm tra reconstruction,
   restore, chế độ inference và DataLoader workers 0/1.
3. Audit qua API main trên 199 volume: đối chiếu từng metadata record, range CT
   và số voxel mask với audit Duong. Báo cáo thực tế nằm ở
   [verification.json](../analysis/merge/verification.json).
4. Giữ hash split, kiểm tra không có file chưa giải quyết xung đột hoặc marker,
   chạy tests, scripts và kiểm tra cú pháp trước commit.

Các báo cáo primary và ảnh cũ được giữ từ Duong vì dữ liệu/processing không đổi.
Báo cáo kiểm tra main ngày trước trong `implementation_report.md` được giữ làm
lịch sử; báo cáo này và README mô tả trạng thái hiện hành.

## Chạy sau tích hợp

```powershell
python -m pip install -e ".[test,training,tools]"
python -m pytest -q
python scripts/smoke_test.py
python examples/read_training_batch.py --assume-mm --workers 1
```

Không chạy lại lệnh tạo split để bắt đầu train; đọc file đã có. Muốn nối model,
dùng API ở README. Các tham số epochs/device trong YAML vẫn là cấu hình dự kiến,
chưa phải trainer. GPU/training và chất lượng phân đoạn chưa được kiểm chứng.

## Final validation

- 135 tests passed, 0 skipped on Windows/CPU; exact environment is in verification.json.
- The synthetic main smoke script reconstructs and exports the original geometry.
- Both public APIs match all six pre-merge real CT slab references exactly.
- Main reader/preprocessor matched all 199 Duong audit records, native ranges and mask counts.
- Original source file sizes/timestamps were unchanged during the full check.
- Main-only zscore/windowed_255 normalization policies are not retained in the active pipeline;
  native is the default, and optional windowing requires explicit encoding/window.
- Main sample H/W and reconstruction layout follow Duong (H=Y, W=X).
- Main empty-mask precision/recall and linked-volume aggregation now follow Duong,
  rather than the former main pooling convention.
- Main model adapter and existing downloader/viewer regression checks pass.
- No GPU/training or real external evaluation was performed.

Main model adapter accepts both batch key conventions and never treats a volume slice index as the master position in the slab. A complete NumPy read/sample/reconstruct/restore passed with torch imports blocked.
