# Báo cáo triển khai giai đoạn 1–3

**Bổ sung ngày 04/11/2026:** xem [bản tổng quan dễ đọc](REPORT.md),
[CSV mỗi dòng một volume](case_statistics.csv) và [hướng dẫn đọc báo cáo hai dataset](../../docs/dataset_reports.md).
Bộ báo cáo mới có summary JSON, biểu đồ tổng quan, đường diễn tiến GT qua các lát
và ảnh CT/GT/overlay. Các kết quả ngày 03/11 bên dưới được giữ làm lịch sử kiểm tra.

Ngày kiểm tra: 2026-11-03 (Asia/Bangkok).
Repository được clone vào `C:\1_HDD\paper\merit-covid-ct` từ commit
`aec40edd18de94cf6d0ce94dbf7c21080eafbbf8`. Báo cáo ghi nhận kết quả kiểm tra
giai đoạn 1–3; source, báo cáo và hai PNG được chuẩn bị trên nhánh `Duong`
theo yêu cầu xuất bản tiếp theo. CT gốc, môi trường Python và cache vẫn ở máy local.

## Các thành phần đã triển khai

| File | Chức năng |
| --- | --- |
| `ct_pipeline/__init__.py` | Package pipeline, không có model/training |
| `ct_pipeline/cases.py` | Đọc manifest, kiểm tra ID/path, provenance patient/group |
| `ct_pipeline/splits.py` | 5-fold deterministic, group disjoint, validation nội bộ, kiểm tra leakage |
| `ct_pipeline/dataset.py` | Kiểm tra NIfTI, RAS, HU, resize XY, slab, metadata vật lý, DataLoader và inverse geometry |
| `ct_pipeline/metrics.py` | Reconstruct volume, Dice/IoU/precision/recall, HD95 mm, sai số thể tích và group macro |
| `tools/data_pipeline.py` | CLI split/audit/smoke/evaluate và ảnh PNG CT/mask/overlay bằng Matplotlib |
| `tests/conftest.py`, `tests/test_splits.py`, `tests/test_dataset.py`, `tests/test_metrics.py` | Fixtures nhỏ và 48 unit tests |
| `pytest.ini` | Cấu hình test paths/imports |
| `requirements.txt`, `.gitignore`, `README.md` | Thêm dependencies, bỏ cache pytest khỏi Git, liên kết hướng dẫn |
| `docs/data_pipeline.md` | Lệnh PowerShell và quy ước geometry/intensity/metrics |
| `splits/covid19_20_5fold.json` | Split thực 199 ca, seed 42 |

## Dataset và split

Workspace ban đầu không có repository hoặc bộ COVID-19-20. Đã dùng nguyên script
`tools/download_covid19_20.py` để tải riêng bộ primary. Script xác nhận 199 cặp,
khoảng 3.77 GB nén. Dữ liệu ở `data/covid19_20/`, không đưa vào Git. Không đọc/xử lý
CT mẫu Zenodo có sẵn ngoài repository, không tải external dataset.

Manifest chỉ có volume ID; bảng XLSX chỉ liệt kê filenames. Đã kiểm tra header cả
199 CT: description là `Computed Tomography` ở 196 ca, trống ở 3 ca; `aux_file`
và `intent_name` trống, không có NIfTI extension ở cả 199 ca. Không có patient
mapping trong các trường này. Dataset card có tuyên bố patient là duy nhất, nhưng thiếu bằng
chứng identity để tự kiểm chứng. Không tự nhóm bằng hậu tố tên file. Do đó split
hiện tại là **volume/group-level**, `patient_ids_available=false`; chưa xác minh
patient-level independence. Code hỗ trợ patient ID/ánh xạ khi có nguồn tin cậy.

| Fold | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 0 | 127 | 32 | 40 |
| 1 | 127 | 32 | 40 |
| 2 | 127 | 32 | 40 |
| 3 | 127 | 32 | 40 |
| 4 | 128 | 32 | 39 |

Đã đọc lại manifest thật 199 ca, kiểm tra không trùng nhóm giữa ba partition,
không thiếu/thừa ca, mỗi ca/nhóm nằm đúng một outer test fold, và so sánh toàn bộ
JSON với split tạo lại seed 42: giống nhau. SHA256 của split:
`d9567a68c33c74b560de70cecda4a462f8c17dca3aea2e0933836bad65c01896`.

Audit nghiêm ngặt trong [audit_strict.json](audit_strict.json) xác nhận 196/199 ca;
ba ca bị từ chối do **đơn vị không gian unknown**, không phải geometry bị lệch:

| Case | Spacing XYZ trong affine/header, đơn vị chưa được xác minh | Scaled intensity min/max | Mask |
| --- | --- | --- | --- |
| `volume-covid19-A-0161` | 0.740, 0.740, 5 | -2048 / 2348 | 0/1 |
| `volume-covid19-A-0585` | 0.428, 0.428, 2 | -2048 / 2253 | 0/1 |
| `volume-covid19-A-0599` | 0.586, 0.586, 5 | -2048 / 6269 | 0/1 |

Cả CT và mask của ba ca đều thiếu đơn vị; 196 ca còn lại ghi mm. Khi bật rõ
`--assume-mm`, các số trên được coi là mm và kiểm tra đầy đủ trong
[audit.json](audit.json). Giả định này không sửa header hoặc voxel gốc, và không
được coi là xác minh đơn vị từ nguồn DICOM. CLI mặc định vẫn từ chối unknown.

Kết quả audit có giả định: **199/199 cặp khớp geometry, finite, nhãn 0/1; không
có lỗi còn lại**. Orientation gốc: 193 LPS và 6 LPI, đều reorient về RAS mà không
nội suy. Phân bố Z spacing sau quy đổi/giả định mm: 188 ca 5 mm, 6 ca 2 mm,
4 ca 1 mm và 1 ca khoảng 0.799988 mm. Không tạo thêm lát.

Cường độ sau NIfTI scaling có range toàn bộ cohort từ -10380 đến 32737:
`volume-covid19-A-0660` có min -10380 và `volume-covid19-A-0295` có max 32737.
Đây là các giá trị cực trị cần giữ trong audit để đối chiếu với nguồn, không tự
sửa dữ liệu hoặc chứng nhận lại HU. Window [-1000,400] chỉ là preprocessing đã
cấu hình. Kiểm tra range `auto_hu` không chứng minh calibration từ DICOM.

## Smoke test thật và kiểm thử phần mềm

- [smoke.json](smoke.json): CT `volume-covid19-A-0003`, native 512×512×291,
  spacing 0.74×0.74×1 mm, master 145, offsets -5..5 mm. Windows worker=1.
- [smoke_5mm.json](smoke_5mm.json): CT `volume-covid19-A-0025`, native
  512×512×61, spacing 0.846×0.846×5 mm, master 30, lát 25..35,
  offsets `[-25,-20,-15,-10,-5,0,5,10,15,20,25]` mm. Worker=0.
- Tensor mẫu `[1,11,256,256]`, mask `[1,256,256]`; batch size 1/2 trả về
  `[1,1,11,256,256]` / `[2,1,11,256,256]`; mask chỉ 0/1.
- PNG có CT master, GT và overlay: `outputs/covid19_20_slab.png`,
  `outputs/covid19_20_slab_5mm.png`. Đã xem trực tiếp ảnh thứ nhất để kiểm tra
  xuất ảnh, ba panel và chú thích spacing/index. Các PNG là ảnh dữ liệu thật,
  không sử dụng image generation.
- Unit tests: **48 passed**. Kiểm tra split, master/mask/offset, padding, resize,
  orientation inverse, mm/meter, oblique affine, HU scaling, lỗi geometry/label/
  NaN/Inf/encoding, cache, Windows workers và seed; metrics có trường hợp rỗng,
  khớp, không giao, giao một phần, threshold, reconstruction và patient macro.
- Ví dụ phép tính tay trong unit tests tổng hợp: TP=1, FP=1, FN=1 cho Dice=0.5,
  IoU=1/3. Đây chỉ là **kiểm thử evaluator**, không phải chất lượng mô hình.
- `git diff --check`, compile Python và `pip check` đều đạt.

Phiên kiểm tra dùng Python 3.13, torch 2.15.0+cpu, nibabel 5.4.2,
numpy 2.5.4, scipy 1.18.1, pytest 9.2.0. Xem [hướng dẫn PowerShell](../../docs/data_pipeline.md)
để chạy lại hoặc dùng evaluator khi đã có prediction thực.

## Giới hạn và phạm vi dừng

Code giai đoạn 1–3 đã được triển khai, nhưng patient identity và đơn vị không gian
của ba ca trên còn cần xác minh từ nguồn. Không tuyên bố đã hoàn tất xác minh
patient-level protocol. `auto_hu` chỉ là kiểm tra encoding theo range sau scaling.

Chưa train U-Net, UNETR++, MERIT-UNETR++ hoặc MADGNet; chưa implement backbone,
TEP/MER/PCG, Reliability-Guided Routing hoặc module nghiên cứu. Không có checkpoint,
model prediction, Dice/IoU thí nghiệm, external evaluation, ablation hay 5-fold
training. Dừng tại giai đoạn 3.
