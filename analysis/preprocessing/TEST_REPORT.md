# Kiểm thử preprocessing

21/11/2026 — kiểm code/fit/QC trên CPU, không thử nghiệm huấn luyện.

| Hạng mục | Kết quả | Bằng chứng |
| --- | --- | --- |
| Unit/regression CPU cuối | PASS76 tests,4 warnings legacy,15.54s | `final_cpu_tests.xml` |
| CPU-fit đúng127 TRAIN | PASS, hai lượt,150,08s | `fit_stats.json`, `read_scope.json`, `cpu_fit.log` |
| VAL/TEST/GT vào FIT | Không có; guard từ chối ngoài whitelist | `read_scope.json`:127 IDs, mỗi ID2 lần |
| Padding có neo góc/stable4-connected | PASS120 ca;7 UNKNOWN fallback | `case_padding_audit.csv` |
| Source CT127 trước/sau FIT | PASS SHA bằng nhau | `fit_stats.json`:source_ct_sha256/source_bytes_unchanged |
| Source CT/GT398 khớp cache audit cũ | PASS | `REAL_DATA_CPU_CHECK.json`, manifest checksum audit trước |
| Native compatibility | PASS effective config/fingerprint giữ nguyên | `LEGACY_COMPATIBILITY.json` |
|6 ca TRAIN thật: GT/formula/LPS | PASS tọa độ/count/nhãn và original GT roundtrip | `REAL_DATA_CPU_CHECK.json` |
| Batch qua CLI chung | PASS CPU,float32,[1,1,11,512,512] | `CANONICAL_BATCH.json` |
| Inference không có mask | PASS | unit test + real A-0599 |
| Model/loss/BN/optimizer/sampler/evaluator | Không thay | kiểm hash/AST trong `VERIFICATION.json` |
| GPU/model forward cho CT thật/train/resume | NOT_RUN, ngoài scope | không có checkpoint mới |
| Dice/IoU trên CT thật, A/B, full epoch | NOT_RUN | không tạo báo cáo chất lượng mô hình |

## Tests cuối đã chạy

```text
python -m pytest tests/test_intensity_fit.py tests/test_research_metrics_normalization.py::test_train_fit_is_frozen_and_rejects_fold_or_case_leakage tests/test_preprocessing.py tests/test_dataset.py tests/test_slab.py tests/duong/test_duong_dataset.py tests/duong/test_duong_dataloader.py -q --junitxml=analysis/preprocessing/final_cpu_tests.xml
```

Bao gồm rule biên/isolated sentinel/noncircular/−1024/biến đổi theo lát/ăn sâu,
fallback toàn voxel và widespread-fallback BLOCKED; oracle quantile/moments trên
array nhỏ; transform chung có clipping; stats NOT_FITTED/tamper/invalid std/bounds;
TRAIN-vs-VAL/fold/manifest mismatch; config load một stats file, sai grid bị từ chối;
GT float64 không binary trước cast/NaN/affine sai bị chặn; GT LPS/LPI giữ nguyên và
khôi phục gốc; sample11×512×512; inference CT-only; Native và global train_zscore cũ.

Test dùng array/NIfTI giả không có model/optimizer mới. Oracle concatenate chỉ ở
test array nhỏ, production fit không concatenate cohort.

## Ngoại lệ phạm vi ở lượt regression đầu

Lượt regression đầu chạy53 tests và đạt, nhưng bộ lọc tên `not native_evaluator`
không loại đúng hai test:

- `tests/duong/test_duong_pipeline_cli.py::test_evaluate_cli_reads_only_selected_partition_with_unordered_slices`
- `tests/test_research_metrics_normalization.py::test_evaluation_reports_native_slice_and_presence_metrics`

Hai test này đã tính metric trên **mẫu giả**, trong đó một test gọi evaluator của
model toy CPU. Đây là sai lệch so với yêu cầu không chạy Dice/IoU; không dùng nó làm
bằng chứng chất lượng V3. Không đọc CT thật để chạy metric, không GPU hoặc optimizer
updates. Lượt cuối chỉ chọn các test preprocessing/loader/fit; không lặp evaluator.
JUnit của lượt đầu vẫn giữ ở `legacy_cpu_tests.xml` để không che giấu việc đã chạy.
Các JUnit có test trùng giữa các lượt; **không cộng53+76 thành129 test độc lập**.

## QC CT thật

Sau khi stats đã đóng băng, đọc6 TRAIN CT/GT: A-0074_1 (noncircular), A-0120 (tail
âm), A-0295 (tail dương), A-0402 (UNKNOWN), A-0599 (LPI/unknown-unit), A-0698 (−1024).
So reader Native/V3, binary mask giữ nguyên từng tọa độ, lesion count bằng nhau;
đảo mapping về GT gốc đúng hoàn toàn; CT/GT bytes trước/sau đọc không đổi.
Tạo loader và CT-only Dataset cho A-0599, không tạo model hoặc optimizer và không
chạy forward/loss. Khởi tạo saved split đọc paired headers199 ca, không dùng held-out
voxel để fit hoặc đánh giá. Kiểm SHA cuối398 file là đọc bytes, không tính metric.

Stats fitting chỉ đọc TRAIN CT theo guard, không dùng GT chọn foreground/FOV hoặc
thống kê. Population accounting và before/after hashes khớp. Nguồn HU và patient
linkage vẫn chưa xác minh; metadata giả định mm không được nâng thành verified.

## Những điều chưa chứng minh

Không đo Dice của V3; không chứng minh cải thiện/collapse repair hoặc hội tụ.
Không GPU/profile VRAM, không train/checkpoint resume với V3 thật. Không xác minh
thời lượng/Drive quota/cloud I/O trên máy người nhận. Không có giả định mask padding
là giải phẫu đã được DICOM xác nhận. Không thử mọi CT/GT với V3 sau fit: QC đầy đủ
giá trị/công thức/GT roundtrip mới thực hiện trên6 ca đại diện, không gọi là199 ca.

Phần fit và config/reader/loader CPU đạt đủ điều kiện `READY_TO_USE` trong phạm vi
preprocessing. Các bước train/evaluate cần một công việc riêng do người dùng yêu cầu.
