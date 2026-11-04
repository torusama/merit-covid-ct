# Những thay đổi trên nhánh Duong: pipeline dữ liệu giai đoạn 1–3

Tài liệu này dành cho người mới đọc repository, giúp hiểu phần nào đã được triển
khai, các file dùng để làm gì và cách chạy lại. Phần triển khai nằm trong commit
`a863fad`, phát triển từ commit `aec40ed`. Kết quả kiểm tra bên dưới được ghi nhận
ngày **03/11/2026**.

**Trạng thái hiện tại:** đã có code chuẩn bị dữ liệu, chia fold, Dataset/DataLoader
và evaluator độc lập model. Chưa có model segmentation hoặc kết quả training.
Patient identity và đơn vị không gian của ba ca vẫn còn giới hạn xác minh.

## 1. Mục đích và các khái niệm cần biết

Bài toán là phân đoạn vùng tổn thương COVID-19 trên CT phổi. Dữ liệu primary là
COVID-19-20, gồm 199 CT volumes và 199 lesion masks tương ứng.

| Khái niệm | Ý nghĩa trong pipeline |
| --- | --- |
| Volume | Một ảnh CT 3D, chứa nhiều lát cắt. |
| Case / case ID | Một cặp CT volume và mask, cùng mã nhận diện trong manifest. |
| Patient | Một bệnh nhân; về nguyên tắc có thể có nhiều volume/series hoặc nhiều lần chụp. |
| Mask / Ground Truth (GT) | Nhãn tham chiếu: 0 là background, 1 là tổn thương. |
| Slab | Một nhóm lát CT liên tiếp, mặc định 11 lát. |
| Master slice | Lát cần dự đoán mask; nằm giữa slab, vị trí 5 trong slab 11 lát nếu đếm từ 0. |
| Fold | Một lần chia train/validation/test trong quy trình cross-validation. |

Luồng sử dụng chính:

```text
CT + mask NIfTI → cases.json → chia 5-fold
                                     ↓
                         chọn fold và partition
                                     ↓
                     kiểm tra dữ liệu, preprocessing
                                     ↓
                       CT slab + mask của master

Prediction từng master slice, khi có model sau này
    → ghép đủ volume → khôi phục geometry gốc → tính metrics → tổng hợp theo nhóm
```

## 2. Trước và sau thay đổi

Trước thay đổi, repository có các script tải, chuẩn bị, audit và xem dữ liệu, cùng
tài liệu định hướng nghiên cứu. Chưa có code split, CT slab Dataset/DataLoader
hoặc evaluator dùng chung.

Sau thay đổi, repository có đủ ba thành phần nền tảng đó, tests và báo cáo chạy
trên dữ liệu thật. Các script download có sẵn được tái sử dụng, không chỉnh sửa.
Dữ liệu CT/mask gốc được giữ nguyên.

## 3. File và folder được thêm

### Code pipeline

| File | Chức năng |
| --- | --- |
| [ct_pipeline/__init__.py](ct_pipeline/__init__.py) | Khai báo package Python của pipeline. |
| [ct_pipeline/cases.py](ct_pipeline/cases.py) | Đọc manifest; kiểm tra ID duy nhất, CT/mask tồn tại và đường dẫn không bị dùng lặp; xác định nguồn patient/group ID; tạo fingerprint nhận diện danh sách ca/nhóm. |
| [ct_pipeline/splits.py](ct_pipeline/splits.py) | Chia 5-fold có seed cố định và giữ nguyên nhóm; kiểm tra các partition không trùng nhóm, không thiếu/thừa ca và mỗi nhóm test xuất hiện đúng một lần. |
| [ct_pipeline/dataset.py](ct_pipeline/dataset.py) | Đọc và kiểm tra NIfTI; preprocessing; tạo slab; trả metadata vật lý; cung cấp DataLoader; khôi phục prediction về shape và orientation gốc. |
| [ct_pipeline/metrics.py](ct_pipeline/metrics.py) | Ghép prediction theo chỉ số master; tính metrics toàn volume; báo cáo từng ca và macro-average theo nhóm. |
| [tools/data_pipeline.py](tools/data_pipeline.py) | CLI chạy các lệnh `split`, `audit`, `smoke`, `evaluate`; xuất PNG CT/mask/overlay bằng Matplotlib. |

### Tests và cấu hình

| File | Chức năng |
| --- | --- |
| [tests/conftest.py](tests/conftest.py) | Tạo NIfTI nhỏ dùng làm fixtures cho unit tests. |
| [tests/test_splits.py](tests/test_splits.py) | Kiểm tra chia nhóm, tái lập seed, coverage và phát hiện split sai. |
| [tests/test_dataset.py](tests/test_dataset.py) | Kiểm tra master/mask, padding, offsets mm, orientation, resize, intensity, lỗi dữ liệu, cache và DataLoader trên Windows. |
| [tests/test_metrics.py](tests/test_metrics.py) | Kiểm tra metric bằng phép tính tay, trường hợp rỗng, threshold, reconstruction, HD95 và macro-average. |
| [pytest.ini](pytest.ini) | Cấu hình thư mục tests và đường dẫn import package. |

Các dữ liệu nhỏ tổng hợp trong tests chỉ dùng kiểm tra phần mềm. Chúng không thay
thế bộ 199 ca và không phải kết quả thí nghiệm của model.

### Split, tài liệu và kết quả kiểm tra

| File | Nội dung |
| --- | --- |
| [splits/covid19_20_5fold.json](splits/covid19_20_5fold.json) | Split thực của 199 ca, seed 42; chứa case IDs và group IDs cho từng partition, fingerprint và provenance identity. |
| [docs/data_pipeline.md](docs/data_pipeline.md) | Hướng dẫn PowerShell và quy ước chi tiết về dữ liệu, tensor, geometry, preprocessing và metrics. |
| [analysis/covid19_20/README.md](analysis/covid19_20/README.md) | Báo cáo triển khai và kiểm tra dữ liệu thực. |
| [analysis/covid19_20/audit_strict.json](analysis/covid19_20/audit_strict.json) | Audit nghiêm ngặt: 196 ca đạt, 3 ca bị từ chối do thiếu đơn vị không gian. |
| [analysis/covid19_20/audit.json](analysis/covid19_20/audit.json) | Audit có bật giả định mm: 199 ca đạt kiểm tra geometry, finite và binary labels; lưu các trường metadata từng ca. |
| [analysis/covid19_20/smoke.json](analysis/covid19_20/smoke.json) | Smoke test CT thật có spacing Z=1 mm, Windows worker=1. |
| [analysis/covid19_20/smoke_5mm.json](analysis/covid19_20/smoke_5mm.json) | Smoke test CT thật có spacing Z=5 mm, master 30, worker=0. |
| [outputs/covid19_20_slab.png](outputs/covid19_20_slab.png) | Ba panel CT master, GT và overlay của ca Z=1 mm. |
| [outputs/covid19_20_slab_5mm.png](outputs/covid19_20_slab_5mm.png) | Ba panel tương tự cho ca Z=5 mm. |

Một số đường dẫn trong JSON smoke test là đường dẫn của máy đã chạy kiểm tra.
Người dùng khác chạy lại lệnh CLI từ thư mục repository; không cần tạo đường dẫn
máy cũ.

## 4. File có sẵn được chỉnh sửa

| File | Thay đổi |
| --- | --- |
| [requirements.txt](requirements.txt) | Thêm `torch>=2.2`, `scipy>=1.11`, `pytest>=8` phục vụ Dataset, metrics và tests. |
| [.gitignore](.gitignore) | Bỏ qua `.pytest_cache/`; cho phép commit riêng hai PNG minh họa đã nêu, tiếp tục bỏ qua các output khác. |
| [README.md](README.md) | Thêm trạng thái triển khai giai đoạn 1–3, liên kết hướng dẫn/báo cáo và giới hạn patient identity. |

## 5. Chính sách xử lý dữ liệu

### Split và giới hạn patient-level independence

Seed mặc định là 42. Validation lấy khoảng 20% phần còn lại sau khi tách test.
Với manifest hiện tại, số ca là:

| Fold | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 0–3, mỗi fold | 127 | 32 | 40 |
| 4 | 128 | 32 | 39 |

**Tất cả các lát/slab của cùng một volume được giữ trong cùng partition.** Không
chia các lát của một volume sang cả train và test.

Điều chưa xác minh là hai volume khác nhau có thuộc cùng một bệnh nhân hay không.
Manifest chỉ có volume ID; bảng XLSX liệt kê filenames; các trường metadata NIfTI
đã kiểm tra không có ánh xạ volume → patient. Dataset card của nguồn tuyên bố bệnh
nhân là duy nhất, nhưng chưa có bảng identity để tự đối chiếu tuyên bố này.

Vì vậy, split hiện dùng `group_id = volume_id` và ghi
`patient_ids_available=false`. Split bảo đảm độc lập theo volume/group, chưa được
xác minh độc lập theo bệnh nhân. Ví dụ, nếu một người có hai volume A và B, cần biết
chúng cùng người để giữ A và B trong cùng partition. Đây là giới hạn bằng chứng,
chưa phải phát hiện rằng dữ liệu đã bị leakage.

Code hỗ trợ `patient_id` trong manifest hoặc `--patient-map` khi có ánh xạ đáng
tin cậy. Khi đó phải tạo lại split và dùng cùng ánh xạ cho Dataset/evaluator.

### Preprocessing và slab

- Kiểm tra shape, affine, orientation, spacing và geometry CT/mask trước xử lý;
  báo lỗi khi lệch, không resize để che mismatch.
- Đọc scaling NIfTI trước kiểm tra intensity; từ chối NaN/Inf và mask ngoài 0/1.
- Chuẩn hóa orientation gần RAS bằng hoán vị/đảo trục. Giữ obliquity gốc.
- Window mặc định `[-1000,400]`, normalize về `[0,1]`; có cấu hình encoding/window.
  Kiểm tra `auto_hu` là sanity check theo range, không chứng minh calibration HU.
- Resize trong mặt phẳng về 256×256: CT dùng bilinear, mask dùng nearest neighbor.
  Không dùng GT để chọn ROI đầu vào.
- Không nội suy Z. Slab hỗ trợ 1/3/5/11 lát, padding bằng lặp lát biên.
- Tính vị trí tâm lát từ affine và offsets có dấu theo vector trục lát, đơn vị mm.
  Lát được lặp ở biên có cùng vị trí và offset thực, không tạo khoảng cách giả.
- Khởi tạo Dataset chỉ đọc headers; cache tối đa một volume đã giải mã mỗi worker.

Với slab 11 lát: `image_slab` có shape `[1,11,256,256]`, `master_mask` có shape
`[1,256,256]`. Batch size 2 cho input `[2,1,11,256,256]`. Metadata giữ thông tin
để khôi phục prediction về grid NIfTI gốc. Chỉ số master là chỉ số canonical,
bắt đầu từ 0; có thể khác thứ tự lát gốc nếu orientation bị đảo/hoán vị.

### Evaluator

Evaluator nhận probability từng master slice; kiểm tra đủ lát, không trùng chỉ số
và đúng shape. Prediction được ghép thành volume, khôi phục về grid gốc rồi mới
threshold, mặc định `>=0.5`. Không tính Dice trung bình từng lát thay cho Dice 3D.

Metrics gồm Foreground Dice, Foreground IoU, precision, recall, HD95 tùy chọn và
sai số thể tích tổn thương. Nếu cả prediction và GT rỗng, bốn metric overlap/
precision/recall bằng 1; nếu chỉ một bên rỗng thì bằng 0. HD95 là `null` nếu một
trong hai mask rỗng. Các giá trị undefined được loại khỏi mean của metric tương
ứng và báo số lượng hợp lệ/undefined.

Khi một nhóm có nhiều volume, lấy trung bình metric trong nhóm trước, rồi lấy
macro-average giữa các nhóm; báo mean và population std (`ddof=0`). Với identity
hiện tại, đây là group/volume macro-average, chưa phải patient macro đã xác minh.

## 6. Kết quả kiểm tra thực tế và các điểm cần lưu ý

- **48 unit tests đạt**; kiểm tra compile, dependencies và `git diff --check` đạt
  trong phiên triển khai. Đây là kết quả kiểm tra code, không phải chất lượng model.
- Đã tải 199 cặp CT/mask, khoảng 3,77 GB nén, bằng script có sẵn. Dữ liệu gốc giữ
  nguyên và không đưa vào Git.
- Audit nghiêm ngặt đạt 196/199 ca. Ba ca `volume-covid19-A-0161`,
  `volume-covid19-A-0585`, `volume-covid19-A-0599` có đơn vị không gian `unknown`.
- Khi bật rõ `--assume-mm`, 199/199 ca khớp geometry, finite và mask chỉ có 0/1.
  Giả định mm của ba ca vẫn cần kiểm chứng từ nguồn; không sửa header để che thiếu sót.
- Sau quy đổi/giả định mm: 188 ca có spacing Z=5 mm, 6 ca Z=2 mm, 4 ca Z=1 mm,
  1 ca Z≈0,799988 mm. Không tạo thêm lát trong preprocessing.
- Intensity sau scaling có range toàn cohort `[-10380,32737]`. Các cực trị được
  lưu trong audit; không tự sửa voxel hoặc chứng nhận lại calibration HU.
- Smoke test CPU thành công với batch size 1/2 và Windows worker 0/1. Ca Z=5 mm,
  master 30 dùng lát 25–35, offsets `[-25,-20,-15,-10,-5,0,5,10,15,20,25]` mm.

## 7. Chạy lại trên máy khác bằng Windows PowerShell

```powershell
# Clone đúng nhánh. Nếu đã có repository và dữ liệu, dùng thư mục hiện có.
git clone --branch Duong https://github.com/torusama/merit-covid-ct.git
Set-Location merit-covid-ct

# Tạo môi trường mới khi cần; không cần activate.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Chỉ tải nếu chưa có bộ primary; không tải thêm Zenodo20 cho giai đoạn này.
if (-not (Test-Path data/covid19_20/cases.json)) {
    .\.venv\Scripts\python.exe tools/download_covid19_20.py
}

.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/data_pipeline.py split --seed 42

# Mặc định nghiêm ngặt: hiện sẽ trả exit code 1 do ba ca thiếu đơn vị.
.\.venv\Scripts\python.exe tools/data_pipeline.py audit --output analysis/covid19_20/audit_strict.json

# Chỉ dùng khi chủ động chấp nhận giả định mm cho ba ca đã nêu.
.\.venv\Scripts\python.exe tools/data_pipeline.py audit --assume-mm
.\.venv\Scripts\python.exe tools/data_pipeline.py smoke --assume-mm --workers 1
.\.venv\Scripts\python.exe tools/data_pipeline.py smoke --assume-mm --case-id volume-covid19-A-0025 --master 30 --output outputs/covid19_20_slab_5mm.png --report analysis/covid19_20/smoke_5mm.json
```

Các lệnh audit/smoke ghi lại report và PNG tương ứng. Máy kiểm tra ban đầu dùng
Python 3.13 và PyTorch bản CPU. Xem [hướng dẫn chi tiết](docs/data_pipeline.md)
để dùng Dataset trong Python hoặc evaluator khi đã có prediction thực.

## 8. Phần có trên GitHub và phần chỉ có ở máy local

GitHub chứa source, tests, cấu hình, split, tài liệu, audit/smoke reports, hai
PNG minh họa trong `outputs/` và bộ CSV/summary/biểu đồ bổ sung trong
`analysis/covid19_20/` (xem mục 10). Các file có sẵn từ repository gốc cũng được giữ lại.

Các thành phần sau không được push theo `.gitignore`:

| Thành phần local | Nội dung / cách tái tạo |
| --- | --- |
| `data/covid19_20/Train/` | 199 CT và 199 mask; tải lại bằng script có sẵn nếu chưa có. |
| `data/covid19_20/cases.json` | Manifest tạo bởi script download. |
| `data/covid19_20/README.md`, `COVID-19-20_TrainValidation.xlsx` | Dataset card và danh sách filenames tải kèm. |
| `data/covid19_20/.cache/` | Cache tải Hugging Face. |
| `.venv/` | Môi trường Python và dependencies; tạo lại bằng lệnh phía trên. |
| `.pytest_cache/`, các `__pycache__/` | Cache tự sinh khi chạy tests/Python. |

## 9. Những phần chưa triển khai

Chưa implement hoặc train U-Net, UNETR++, MERIT-UNETR++ hay MADGNet. Chưa có
backbone, TEP/MER/PCG, Reliability-Guided Routing hoặc module nghiên cứu mới.
Chưa có checkpoint, model prediction, Dice/IoU thí nghiệm, ablation, 5-fold
training hoặc external evaluation. Các file/script Zenodo20 có sẵn thuộc nội dung
repository gốc; lần triển khai này không tải hoặc xử lý thêm dataset đó.

**Phạm vi thay đổi dừng ở nền tảng dữ liệu và evaluator giai đoạn 1–3.**

## 10. Bổ sung CSV và biểu đồ ngày 04/11/2026

Để người đọc dễ xem thông tin 199 volume như bộ báo cáo Zenodo20 có sẵn, bổ sung:

| File | Chức năng |
| --- | --- |
| [ct_pipeline/reporting.py](ct_pipeline/reporting.py) | Thống kê GT ở độ phân giải gốc, đối chiếu audit và tổng hợp cohort. |
| [tools/summarize_covid19_20.py](tools/summarize_covid19_20.py) | Xuất CSV, summary JSON, ba PNG và REPORT.md. |
| [tests/test_reporting.py](tests/test_reporting.py) | 8 tests mới về số lát/voxel/diện tích/thể tích, mask rỗng, orientation, dữ liệu không hợp lệ, audit cũ không khớp và đơn vị thiếu. |
| [docs/dataset_reports.md](docs/dataset_reports.md) | Giải thích bằng tiếng Việt các file và chỉ số của Zenodo20 và COVID-19-20. |
| [analysis/covid19_20/case_statistics.csv](analysis/covid19_20/case_statistics.csv) | Mỗi dòng là một volume; lưu metadata, GT statistics và partition của từng fold. |
| [analysis/covid19_20/summary.json](analysis/covid19_20/summary.json) | Các con số tổng hợp và số voxel lesion trên từng lát. |
| [analysis/covid19_20/dataset_overview.png](analysis/covid19_20/dataset_overview.png) | Tổng quan spacing, số lát, lesion burden và lát có/không có tổn thương. |
| [analysis/covid19_20/lesion_trajectories.png](analysis/covid19_20/lesion_trajectories.png) | Diện tích GT qua các lát ở các ca ví dụ. |
| [analysis/covid19_20/ct_mask_examples.png](analysis/covid19_20/ct_mask_examples.png) | Các hàng CT/GT/overlay ở lát giữa các ca ví dụ. |
| [analysis/covid19_20/REPORT.md](analysis/covid19_20/REPORT.md) | Tổng kết dễ đọc, ý nghĩa cột CSV, số liệu và giới hạn. |

`analysis/covid19_20/README.md` và `docs/data_pipeline.md` được thêm liên kết và
lệnh chạy lại. Sau bổ sung, **56 tests đạt** (48 tests cũ và 8 tests mới).

Kết quả đọc lại dữ liệu thật:

- CSV có **199 dòng dữ liệu và 71 cột**, mỗi dòng là một volume, không phải một bệnh nhân đã xác minh.
- Tổng **13.705 lát**, gồm **4.981 lát có tổn thương GT** và **8.724 lát có mask rỗng**.
- Số voxel tổn thương từng ca khớp với audit; các ma trận affine và metadata gốc
  được giữ trong các cột `audit_*`. CSV dùng UTF-8 BOM để mở bằng Excel.
- Patient ID chưa biết được để trống. Ba ca thiếu đơn vị được đánh dấu riêng bằng
  `spatial_units_assumed_mm=True`; các ca có đơn vị mm không bị đánh dấu giả định.
- Đã đối chiếu số dòng, case ID duy nhất, tổng số lát, partition trong 5 fold,
  các trường audit được xuất và liên kết trong tài liệu; đã xem cả ba PNG mới.

Người mới nên đọc [REPORT.md](analysis/covid19_20/REPORT.md) trước, mở CSV bằng
Excel, rồi xem các PNG. Muốn hiểu báo cáo Zenodo20 có sẵn và các khác biệt giữa
hai dataset, xem [hướng dẫn đọc báo cáo](docs/dataset_reports.md).

Tạo lại CSV, summary và hình ảnh bằng PowerShell từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe tools/summarize_covid19_20.py --assume-mm
```

Lệnh yêu cầu dataset, audit đầy đủ và split đã có. `--assume-mm` là khai báo chấp
nhận giả định đơn vị mm cho ba ca đã nêu, không phải xác minh đơn vị từ nguồn.

Min/max intensity được tái sử dụng từ audit đầy đủ trước đây; headers và mask của
199 ca được đọc lại. Không tính tỷ lệ lesion/phổi vì thiếu lung mask; không làm
components/resize-loss trong bản báo cáo này. Patient identity và đơn vị mm giả
định của ba ca giữ nguyên trạng thái chưa xác minh. Không xử lý thêm Zenodo20,
không thay đổi split, không training.
