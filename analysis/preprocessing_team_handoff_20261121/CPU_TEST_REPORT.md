# Kiểm tra chốt bàn giao

Ngày21/11/2026. Đây là lượt QC/bàn giao, **không nghiên cứu hay sửa thuật toán V3**.

## Đã chạy thật

1. `scripts/qc_preprocessing.py`: gọi đúng `read_arrays` production trên từng cặp,
   đủ199 ca;127 TRAIN/32 VAL/40 TEST đều PASS. CT finite sau transform; header,
   affine/orientation/spacing theo kiểm tra hiện hữu; native512/LPS/slab11 giữ nguyên.
   GT scaled exact0/1 trước cast, số lesion trước/canonical/inverse bằng nhau và
   tổng mismatch tọa độ bằng0. Hash CT/GT trước/sau không đổi.516,87 giây CPU.
2.7 tests QC mới đạt: synthetic LPI inverse/hash, GT float64 không binary, affine
   sai, case filename sai, file thiếu/BLOCKED và từ chối ghi đè báo cáo. Các tests
   dùng stats V3 thật đã cố định, không fit hoặc đánh giá model.
3. Source snapshot đã giải ra thư mục riêng:31 tests CPU preprocessing/QC đạt trong
   8,76s; import được xác nhận từ chính `project/src`, không fallback sang workspace.
   Một số unit fixtures kiểm thuật toán thống kê bằng array giả, không refit stats
   thật hoặc ghi đè fit_stats. Không chạy test metric, optimizer hoặc training.
4. Build wheel thành công bằng `pip wheel --no-index --no-deps --no-build-isolation`:
   chỉ kiểm đóng gói, không tải/cài dependency, không chạy model/GPU.
5. Bản source riêng đọc batch CT thật trên CPU thành công qua CLI hiện có với YAML
   chỉ đổi paths: shape`[1,1,11,512,512]`, float32, đúng127/32/40. Không sao chép CT.
6. Cuối cùng kiểm protected hashes và manifest bundle; raw stats và provenance
   checksum giữ nguyên. Chi tiết machine-readable trong `TEAM_HANDOFF_VERIFICATION.json`.

## Không chạy

Không fit lại127 TRAIN, không normalization stats mới từ VAL/TEST. Không GPU/model
forward/training, không validation Dice/IoU hoặc A/B; không checkpoint mới, không
thay model/loss/BN/optimizer/sampler/evaluator hoặc spacing. Không commit/push/merge.
Không chạy lại hai test metric trên mẫu giả từng bị chọn nhầm ở lượt trước.

## Tái lập và giới hạn

Gói là source snapshot cần thiết cho MERIT từ working tree dirty đã ghi rõ base
commit, không phải commit được tạo giả hoặc overlay thiếu dependency source.
`SOURCE_PROVENANCE.json` liệt kê byte hashes và khác biệt nội dung/line endings;
`pyproject.toml` cung cấp dependencies, các vendored UNETR++ resources/license được
giữ. Không đóng kèm nghiên cứu MADGNet, notebook/demo, CT/GT hoặc checkpoint.

Huấn luyện/model trên GPU và cloud I/O chưa kiểm. Người nhận phải dùng cache khớp
hash/manifest/split, giữ stats cố định và chọn output mới cho training V3. HU/patient
identity/ý nghĩa DICOM của padding chưa xác minh. Trạng thái READY chỉ cho bàn giao
preprocessing, không cho chất lượng mô hình hay clinical readiness.
