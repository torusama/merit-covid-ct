# Thay đổi preprocessing CT + GT

Ngày21/11/2026. **READY_TO_USE cho preprocessing**, đã fit thật; không có kết luận
training/GPU/Dice. Thực hiện prompt V3 ngày20/11, thay cho đề xuất clipped-Z-score
toàn voxel trước đây. File kế hoạch riêng được nhắc trong prompt không có ở những
vị trí local đã tìm; yêu cầu triển khai chi tiết trong prompt là phạm vi thực hiện.

## Các file production đã sửa/thêm

| File | Thay đổi |
| --- | --- |
| `src/merit_covid/data/pipeline/intensity_fit.py` — thêm | Một module CPU-fit: rule boundary4-connected, confidence/fallback/gate, histogram pooled, clipped moments, provenance guards |
| `src/merit_covid/data/pipeline/config.py` | Cho phép một policy mới; dùng lại zscore_fit/mean/std, không thêm field mặc định làm đổi fingerprint Native |
| `src/merit_covid/data/pipeline/preprocessing.py` | Transform CT chung theo stats; GT binary kiểm float64 trước cast; giữ toàn bộ hình học/GT; metadata V3 rõ ràng |
| `src/merit_covid/data/pipeline/geometry.py` | Bỏ nhánh unreachable khi mask thiếu; không đổi quy tắc LPS/affine/spacing |
| `src/merit_covid/data/pipeline/dataset.py` | Check stats/split khi from_split; bỏ hai private alias import không có người gọi; không thêm Dataset |
| `src/merit_covid/engine/training.py` | Wiring stats file→config, binding fold/manifest/source CT trước tạo model; giữ optimizer/sampler/evaluator/checkpoint methods |
| `scripts/fit_train_zscore.py` | Tận dụng CLI fit đang có, thêm --policy/--output-dir cho V3; global train_zscore giữ chức năng/format cũ |
| `scripts/read_training_batch.py` | Thêm --config dùng chính config/reader/loader chung, vẫn giữ đường gọi Native cũ |
| `configs/merit_preprocessing.yaml` — thêm | Cấu hình chuẩn fold0: trỏ bộ stats đã fit, native512/slab11; model/loss/training settings bằng Native config |
| `tests/test_intensity_fit.py` — thêm | Test CPU cho domain FIT, clipping/moments, GT, binding, loader và missing-mask inference |
| `docs/PREPROCESSING_HANDOFF.md` — thêm | Hướng dẫn tiếng Việt, luồng dữ liệu và lệnh chạy |
| `README.md`, `docs/data_pipeline.md`, `docs/merit_pipeline.md` | Đặt V3 là khuyến nghị cho lần train mới; Native được ghi rõ là đường lịch sử/default tương thích |

Không đổi model/loss/BatchNorm/optimizer/sampler hoặc phép tính Dice/IoU. Không đổi
default CLI train/cv sang policy khác một cách âm thầm: muốn V3 phải truyền config
chuẩn. Không có migration checkpoint Native→V3, không sửa checkpoint cũ.

## Đã xóa gì, và vì sao?

**Không xóa nguyên file nào.** Đã tìm references trong src/scripts/tools/tests/docs
và cả source snapshot/notebook local. Không thấy script/config preprocessing nguyên
vẹn nào có thể chứng minh bỏ đi mà không phá API, test hoặc tái lập run quan trọng.
Không xóa chỉ để đạt một con số dọn dẹp.

Đã xóa code dư trong hai module:

- `dataset.py`: alias `_slab_indices`, `_slice_positions`; không có caller trong repo,
  chỉ có các định nghĩa cũ ở snapshot/build. Dùng trực tiếp `sample_arrays`; exports
  public và các định nghĩa trong slab.py giữ nguyên.
- `geometry.py`: điều kiện `if require_mask` và return không thể chạy sau nhánh
  CT-only đã return trước đó. Thiếu mask ở labeled operation vẫn báo cùng lỗi.

Giữ `src/merit_covid/data/preprocessing.py`: bridge đang được các Dataset/tests/API
khác import và đã delegate normalize/resize về shared pipeline, không phải bản V3
song song. Giữ Native/global train_zscore, các preflight diagnostic, source bundle,
report và notebook lịch sử. Không đụng build/outputs/dataset/checkpoints để dọn rác.
Không tạo notebook train hoặc thư mục copy source mới trong production.

## Fit và artifact thật

- `fit_stats.json`: L=-1241; U=1032; mean=-583.0622212292245; std=488.0394718302036.
- `case_padding_audit.csv`:127 ca;120 CONFIDENT,7 UNKNOWN,0 AMBIGUOUS.
- Đếm2.309.488.640 voxel, loại473.812.900 khỏi FIT; accepted1.835.675.740.
  Tổng sentinel473.812.909; giữ9 sentinel rời rạc bên trong. Không xóa voxel khỏi input.
-7 UNKNOWN giữ trọn105.644.032 voxel, chiếm khoảng5,76% population accepted;
  `unverified_background_included=true`. Không suy−1024 thành padding.
- CPU-fit150,08s, hai lượt CT; guard bảo đảm không đọc GT/VAL/TEST để fit. Phân vị có
  sai số tối đa1/32 đơn vị scaled, mean/std tính lại trên voxel thật sau clipping.
- QC6 ca thật và kiểm hash cuối35,38s; loader/CLI shape1×1×11×512×512; no-mask inference PASS.
- `REAL_DATA_CPU_CHECK.json`, `CANONICAL_BATCH.json`, `read_scope.json`,
  `LEGACY_COMPATIBILITY.json`, `fit_status.json`, log và JUnit giữ bằng chứng riêng.

## Test và giới hạn

76 test CPU chọn lọc cuối đạt,4 warning provenance legacy đã dự kiến. Không train hoặc
dùng GPU. Có hai test metric trên dữ liệu giả chạy kèm regression đầu do lọc tên chưa
đúng; ghi rõ trong TEST_REPORT. Không có Dice/IoU mô hình trên CT thật hoặc A/B mới.

Clipping làm mất phân biệt ở tails. Padding là heuristic có criteria, không phải
verified anatomy/HU. Patient identity chưa xác minh; đơn vị unknown vẫn là giả định
mm cũ. READY_TO_USE chỉ cho phần preprocessing đã kiểm, không khẳng định training
GPU/resume/model quality đã sẵn sàng hoặc collapse epoch1 đã được chữa.

Working tree vốn dirty trước lượt này; giữ nguyên các thay đổi ngoài scope. Không
commit/push/merge hoặc chuyển nhánh. `VERIFICATION.json` ghi kiểm tra bảo vệ cuối.
