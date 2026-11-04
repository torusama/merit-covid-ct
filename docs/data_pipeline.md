# Pipeline dữ liệu: chỉ giai đoạn 1–3

Code nằm trong `ct_pipeline/`; CLI duy nhất là `tools/data_pipeline.py`.
Không có model, optimizer, training loop hoặc checkpoint. Không sử dụng Zenodo20.

## Chạy lại trên Windows PowerShell

```powershell
Set-Location C:\1_HDD\paper\merit-covid-ct

# Khi cần tạo môi trường mới. Có thể dùng Python 3.11 trở lên.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Chỉ chạy nếu bộ COVID-19-20 chưa có. Script dùng cache khi tải gián đoạn.
.\.venv\Scripts\python.exe tools/download_covid19_20.py

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/data_pipeline.py split --seed 42

# Chế độ nghiêm ngặt: báo lỗi nếu header không xác định đơn vị không gian.
.\.venv\Scripts\python.exe tools/data_pipeline.py audit --output analysis/covid19_20/audit_strict.json

# Có 3 ca thiếu đơn vị trong header (xem báo cáo). Lệnh này CHỦ ĐỘNG giả định
# các tọa độ/spacing của các ca đó cũng là mm. Đây chưa phải đơn vị đã xác minh.
.\.venv\Scripts\python.exe tools/data_pipeline.py audit --assume-mm
.\.venv\Scripts\python.exe tools/data_pipeline.py smoke --assume-mm --workers 1
.\.venv\Scripts\python.exe tools/data_pipeline.py smoke --assume-mm --case-id volume-covid19-A-0025 --master 30 --output outputs/covid19_20_slab_5mm.png --report analysis/covid19_20/smoke_5mm.json
```

Không cần activate môi trường. Môi trường dùng trong phiên triển khai được tạo với
`--system-site-packages` để tái sử dụng NumPy/NiBabel/Matplotlib đã cài sẵn;
các lệnh trên tạo môi trường độc lập khi chạy từ đầu. CPU được dùng xuyên suốt.

## Manifest, patient identity và split

Manifest hiện có là danh sách `{id, image, mask, shape, spacing_mm}`. Đường dẫn
CT/mask được resolve theo thư mục chứa manifest; kiểm tra trùng ID, trùng đường dẫn,
tồn tại file và số lượng ca (CLI mặc định yêu cầu đúng 199). Không suy diễn patient
ID bằng cách xóa hậu tố `_0`/`_1` của tên volume.

Ưu tiên `patient_id` trong manifest; nếu thiếu thì dùng `group_id`; nếu cả hai thiếu,
dùng volume ID và lưu `patient_ids_available: false`. Metadata hiện tại không đủ
để kiểm chứng bệnh nhân có nhiều series hay không. Dataset card tuyên bố các bệnh
nhân là duy nhất, nhưng đó không phải bảng ánh xạ identity có thể kiểm tra tại đây.
**Split hiện tại bảo đảm độc lập theo volume/group, chưa bảo đảm patient-level
independence.**

Khi có ánh xạ bệnh nhân được xác minh, tạo JSON chứa đủ mọi case ID:

```json
{"volume-covid19-A-0003": "patient_001", "another_case_id": "patient_001"}
```

Đây chỉ là minh họa schema, không phải identity thực của các ca. Truyền cùng
`--patient-map path/to/patients.json` khi chạy split, smoke, audit và evaluate.
`patient_id` có một phần nhưng thiếu ở các ca khác sẽ gây lỗi. Danh tính được cung
cấp vẫn cần được xác minh ở nguồn; code chỉ bảo đảm không tách nhóm đó giữa các tập.

Split seed 42 được cân bằng theo số volume, giữ nguyên nhóm. Mỗi outer test group
xuất hiện đúng một lần trong 5 fold. Validation lấy khoảng 20% phần còn lại; tất cả
partition không rỗng. Không stratify lesion burden vì chưa có thống kê tin cậy
được cung cấp tại thời điểm tạo split. Không dùng test mask để chọn split.

Với manifest hiện tại: fold 0–3 có train/validation/test = 127/32/40;
fold 4 = 128/32/39. JSON chứa case IDs, group IDs, seed, provenance và fingerprint
của ID/group/tên file. Fingerprint không phải checksum nội dung ảnh. Dataset kiểm
tra lại toàn bộ partition trước khi chọn dữ liệu của một fold.

## Geometry, intensity và tensor

`inspect_pair()` kiểm tra 3D shape, affine hữu hạn/khả nghịch, spacing header khớp
affine, orientation, đơn vị không gian và geometry CT/mask trước preprocessing.
Không sửa hoặc resample mask để che mismatch. mm/meter/micron được đổi về mm;
đơn vị unknown bị từ chối trừ khi chủ động bật `assume_mm` và ghi vào metadata.

`read_volume()` đọc một ca bằng NiBabel, áp dụng scaling NIfTI, kiểm tra toàn bộ
NaN/Inf và nhãn đúng 0/1. Chuẩn hóa thành orientation gần RAS bằng hoán vị/đảo trục;
không deoblique, không nội suy Z. CT được window mặc định [-1000, 400] rồi đưa về
[0,1]. `auto_hu` đòi hỏi range có cả giá trị dưới và trên -300, trong giới hạn
[-32768,65535]; đây là sanity check, không chứng minh HU từ DICOM. Ảnh encode khác
bị từ chối; `hu` là khai báo HU chủ động, `normalized_0_1` kiểm tra [0,1] và bỏ HU
windowing. Không âm thầm coi ảnh 0–255 là HU.

Resize mặc định H=W=256: CT bilinear (`align_corners=False`), mask nearest neighbor
(`nearest-exact`, cùng quy ước tâm pixel). Không crop bằng ground truth. Số lát và
spacing dọc trục lát được giữ nguyên. Các kích thước trong metadata NIfTI là X,Y,Z;
tensor volume nội bộ là Z,Y,X, nên tensor H=Y, W=X.

`CTSlabDataset` trả về:

| Trường | Shape / ý nghĩa |
| --- | --- |
| `image_slab` | float32 `[1, D, H, W]`, D thuộc 1/3/5/11 |
| `master_mask` | float32 binary `[1, H, W]` |
| `master_slice_index` | Chỉ số canonical RAS, bắt đầu từ 0 |
| `case_id`, `group_id` | Identity từ manifest/split |
| `z_offsets_mm` | `[D]`, khoảng cách có dấu tới master |
| `slice_indices` | `[D]`, chỉ số thực sau padding |
| `slice_positions_mm` | `[D,3]`, tâm lát trong tọa độ RAS world |
| `metadata` | Affine/shape/orientation gốc, inverse orientation, cấu hình và intensity |

Master ở vị trí D//2. Padding lặp lát biên; lát lặp có offset thực trùng nhau,
không giả tạo vị trí ngoài volume. Offset được chiếu từ chênh lệch tâm lát affine
lên vector đơn vị dọc trục lát: với affine shear, đây là khoảng cách tâm có dấu,
không phải khoảng cách vuông góc giữa hai mặt phẳng. Với Z=5 mm, master 30 lấy
25..35 và offsets -25,-20,...,0,...,20,25 mm.

Khởi tạo Dataset chỉ đọc headers. Mỗi worker cache tối đa một ca, giải phóng cache
cũ trước ca tiếp theo. Random slice shuffle có thể làm tăng số lần giải nén NIfTI;
không preload cả cohort để tránh điều đó. Windows default `num_workers=0`;
`num_workers=1` cũng đã được kiểm thử. Chạy code DataLoader dưới `if __name__ ==
"__main__":` khi dùng workers. Metadata trong batch là danh sách vì geometry gốc
khác nhau giữa các ca. Đặt `size_hw=None` chỉ khi các ảnh trong batch có cùng H,W.

```python
from ct_pipeline.dataset import CTSlabDataset, PreprocessConfig, make_dataloader

if __name__ == "__main__":
    dataset = CTSlabDataset.from_split(
        "data/covid19_20/cases.json", "splits/covid19_20_5fold.json",
        fold=0, partition="train", config=PreprocessConfig(assume_mm=True),
    )
    loader = make_dataloader(dataset, batch_size=2, shuffle=True, seed=42)
    batch = next(iter(loader))  # [2, 1, 11, 256, 256], CPU
```

`restore_volume()` đưa tensor prediction Z,Y,X về H,W native bằng nội suy 2D,
sau đó đảo orientation để trả về mảng X,Y,Z gốc. Binary dùng nearest; probability
dùng bilinear. Evaluator khôi phục probability trước rồi mới threshold ở native
grid. Chỉ số canonical không nhất thiết là chỉ số lát gốc khi affine đảo/hoán vị
trục. Affine/shape gốc được giữ để lưu prediction NIfTI về đúng geometry.

## Evaluator độc lập model

`SliceReconstructor` nhận từng master slice theo thứ tự bất kỳ, báo lỗi nếu trùng,
thiếu, sai shape, chỉ số ngoài volume, NaN/Inf hoặc probability ngoài [0,1]. Không
nhận logits; caller phải sigmoid trước. Mỗi ca được đánh giá đúng một lần.

Tính TP/FP/FN trên **toàn volume native**, rồi Foreground Dice, Foreground IoU
(không bao gồm background), precision và recall. Không trung bình Dice từng lát.
Nhiều series cùng group: trung bình metric các volume trong group, sau đó macro
trung bình các group, mỗi group có trọng số như nhau. Báo từng ca, từng group và
mean/std population (`ddof=0`). Khi patient identity chưa xác minh, báo group/volume
macro thay vì tuyên bố patient-level macro đã xác minh.

| Trường hợp | Dice / IoU / precision / recall | HD95 |
| --- | --- | --- |
| Cả hai rỗng | 1 / 1 / 1 / 1 | null |
| Chỉ một mask rỗng | 0 / 0 / 0 / 0 | null |
| Khớp hoàn toàn, không rỗng | 1 / 1 / 1 / 1 | 0 mm |

HD95 dùng bề mặt voxel 6-connected, tọa độ world từ đầy đủ affine và percentile 95
của hai tập khoảng cách nearest-surface gộp lại. Undefined được ghi null và loại
khỏi mean của metric đó; số ca/group hợp lệ và undefined được báo rõ. Sai số thể
tích tổn thương dùng `abs(det(affine[:3,:3]))/1000` mL/voxel; ghi sai số có dấu,
giá trị tuyệt đối và sai số tương đối (null khi GT rỗng).

Để dùng khi đã có prediction thực sau này, chuẩn bị một file `case_id.npz` cho mỗi
ca trong partition: `indices` là mảng số nguyên `[Z]`, `probabilities` là float
`[Z,H,W]` hoặc `[Z,1,H,W]` trên canonical processed grid. Cấu hình preprocessing
phải trùng với lúc tạo prediction. Threshold được ấn định bằng validation trước;
CLI không tìm threshold từ test GT.

```powershell
# Chỉ chạy khi đã có các file prediction thực. Task hiện tại không tạo prediction.
.\.venv\Scripts\python.exe tools/data_pipeline.py evaluate --assume-mm --fold 0 --partition test --predictions path/to/predictions --threshold 0.5 --hd95 --output outputs/metrics_fold0.json
```

Mọi mask/prediction tổng hợp trong `tests/` chỉ dùng kiểm tra phần mềm. Các giá trị
Dice/IoU trong unit tests không phải kết quả thí nghiệm COVID-19-20.

## CSV và biểu đồ mô tả 199 volume

Sau khi có manifest và audit đầy đủ, chạy:

```powershell
.\.venv\Scripts\python.exe tools/summarize_covid19_20.py --assume-mm
```

Lệnh tạo `case_statistics.csv`, `summary.json`, `dataset_overview.png`,
`lesion_trajectories.png`, `ct_mask_examples.png` và `REPORT.md` trong
`analysis/covid19_20/`. CSV dùng UTF-8 BOM và mỗi dòng tương ứng một volume.
Patient ID chưa biết được để trống; không coi volume ID là patient ID.

Thống kê mask dùng native-resolution canonical RAS, trước resize; không nội suy Z.
Script kiểm tra lại headers CT/mask và toàn bộ mask từng ca, đối chiếu geometry và
voxel count với audit cũ. Min/max CT tái sử dụng kết quả audit cũ, không đọc lại
toàn bộ intensity CT của 199 ca. Chỉ các ca minh họa mới được đọc CT để xuất PNG.
Spacing/diện tích/thể tích của ba ca thiếu đơn vị phụ thuộc `--assume-mm`.

Tỷ lệ lesion lấy mẫu số là toàn bộ volume CT, không phải volume phổi vì không có
lung mask. GT Dice giữa các lát liên tiếp mô tả tính liên tục nhãn, không phải
chất lượng mô hình. Xem [hướng dẫn đọc báo cáo](dataset_reports.md) để đối chiếu
với các báo cáo Zenodo20 đã có sẵn.
