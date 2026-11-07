# Tổng quan dữ liệu COVID-19-20

Ngày tạo: 2026-11-07T10:28:45+07:00. Báo cáo mô tả dữ liệu thật; chưa có model prediction.

## Cấu hình hiện hành

- Hướng mảng thống nhất **LPS**: 193 volume LPS giữ nguyên thứ tự; 6 volume LPI đảo Z.
- **Không resize mặc định**: CT/mask giữ H,W gốc; không nội suy không gian.
- CT vẫn áp dụng scaling NIfTI, HU window và normalize; mask giữ nhãn 0/1.
- Các trường `canonical_*` nay chỉ grid LPS. Tọa độ world của NiBabel vẫn là RAS+.
- Báo cáo, audit và ảnh minh họa đã được tạo lại theo cấu hình này; không dùng
  prediction RAS/256×256 của pipeline cũ với metadata mới.

## Nên mở file nào?

| File | Cách đọc |
| --- | --- |
| [case_statistics.csv](case_statistics.csv) | Mở bằng Excel: **mỗi dòng là một CT volume**, tổng 199 dòng dữ liệu. |
| [summary.json](summary.json) | Các con số tổng hợp và số voxel tổn thương trên từng lát của từng volume. |
| [dataset_overview.png](dataset_overview.png) | Bốn biểu đồ: spacing Z, số lát mỗi volume, tỷ lệ tổn thương và số lát có/không có tổn thương. |
| [lesion_trajectories.png](lesion_trajectories.png) | Diện tích GT tổn thương thay đổi qua các lát của các volume ví dụ. |
| [ct_mask_examples.png](ct_mask_examples.png) | CT, GT mask và ảnh chồng mask lên CT; lấy lát giữa volume, không chọn lát theo GT. |

## Các con số chính

- 199 CT volumes; chưa xác minh tương ứng với bao nhiêu bệnh nhân khác nhau.
- 13,705 lát canonical: 4,981 lát có ít nhất một voxel tổn thương,
  8,724 lát có mask rỗng.
- Mỗi volume có từ 39 đến 361 lát;
  median là 63 lát.
- 14,818,707 voxel tổn thương trên tổng 3,592,683,520 voxel CT.
- Tỷ lệ gộp lesion/toàn bộ CT: 0.4125%.
  Trung bình tỷ lệ riêng từng volume: 0.4378%.
  Hai cách tổng hợp có trọng số khác nhau, nên có thể cho kết quả khác nhau.
- Tất cả headers CT/mask được kiểm tra lại geometry; toàn bộ mask được đọc lại,
  kiểm tra finite và nhãn 0/1. Tổng voxel tổn thương từng ca khớp audit cũ.

| Spacing Z gốc (mm, có giả định cho các ca thiếu đơn vị) | Số volume |
| --- | ---: |
| 0.799988 | 1 |
| 1 | 4 |
| 2 | 6 |
| 5 | 188 |

## Các cột CSV thường dùng

| Cột | Giải thích |
| --- | --- |
| `volume_order`, `case_id` | Số thứ tự tương ứng biểu đồ và mã volume. |
| `group_id`, `patient_id`, `patient_ids_available` | Nhóm dùng chia dữ liệu; patient ID để trống khi chưa biết. Group hiện là volume ID. |
| `image`, `mask` | Đường dẫn file CT và mask theo thư mục chứa manifest. |
| `original_shape_x/y/z` | Kích thước NIfTI gốc; trục thứ ba gốc có thể bị đảo/hoán vị khi reorient. |
| `slices` | Số lát axial sau thống nhất LPS, trước padding; giữ nguyên độ phân giải. |
| `spacing_x/y/z_mm`, `original_orientation` | Khoảng cách voxel và hướng của ảnh gốc. |
| `spatial_unit_in_header`, `spatial_units_assumed_mm` | Header có ghi đơn vị không, và ca đó có phải giả định mm không. |
| `ct_min`, `ct_max`, `ct_dtype`, `ct_scaling_slope/intercept` | Thông tin intensity CT. Min/max lấy lại từ audit đầy đủ trước đây; dtype/scaling đọc header hiện tại. |
| `positive_slices`, `empty_slices` | Số lát có và không có tổn thương GT. |
| `lesion_voxels`, `total_voxels` | Số voxel tổn thương và số voxel của toàn volume. |
| `lesion_fraction_volume` | Tỷ lệ 0–1 của lesion trên toàn bộ CT, **không phải tỷ lệ trên phổi**. |
| `lesion_volume_ml` | Thể tích GT tổn thương, đơn vị mL, dựa trên determinant của affine. |
| `first/last_positive_slice` | Chỉ số canonical 0-based đầu/cuối có tổn thương; trống nếu không có. |
| `positive_run_count/lengths`, `max_positive_run_length` | Số đoạn lát liên tiếp có tổn thương và độ dài các đoạn. |
| `adjacent_positive_pairs`, `adjacent_gt_dice_median` | Số cặp GT liên tiếp cùng có tổn thương và median Dice giữa các GT đó; **không phải Dice model**. |
| `fold_0` đến `fold_4` | Ca đó thuộc train, validation hay test trong từng fold của split đã lưu. |
| `geometry_match`, `mask_finite_binary`, `z_resampled` | Kết quả kiểm tra geometry/mask và việc có nội suy Z hay không. |
| Các cột `audit_*` | Giữ toàn bộ trường của audit gốc; list/matrix được lưu dạng JSON trong một ô CSV. |

## Giới hạn cần đọc cùng số liệu

- **Patient-level independence chưa xác minh**: chưa biết hai volume khác nhau có cùng người hay không.
  Không suy diễn patient ID từ tên file. Báo cáo không thay đổi split hoặc chọn dữ liệu training.
- Các ca sau thiếu đơn vị trong header, đang chủ động giả định mm: `volume-covid19-A-0161`, `volume-covid19-A-0585`, `volume-covid19-A-0599`.
  Spacing, diện tích và thể tích của chúng phụ thuộc giả định này.
- Không có lung mask nên không tính phần trăm phổi bị tổn thương hoặc tổn thương ngoài phổi.
  `lesion_fraction_volume` lấy mẫu số là toàn bộ khối CT, gồm cả vùng không khí bên ngoài cơ thể.
- Không tính connected components hoặc mất thành phần do resize trong bản báo cáo này;
  các chỉ số đó có ở báo cáo Zenodo20. Báo cáo hiện dùng độ phân giải mask gốc.
- Không đọc lại toàn bộ intensity CT của 199 ca: dùng min/max và kết quả kiểm tra intensity từ
  audit cũ; headers và mask được kiểm tra lại. Không có checksum để xác minh byte identity CT.
- Intensity gộp trong audit: [-10380.0, 32737.0]. Đây không phải chứng nhận calibration HU.
- Ví dụ CT được chọn theo case ID đầu tiên trong từng nhóm spacing, rồi lấy lát giữa;
  mask ở lát minh họa có thể rỗng. Không dùng GT để chọn ROI đầu vào.

## Chạy lại bằng PowerShell

Chạy từ thư mục gốc repository khi đã có dataset và audit:

```powershell
.\.venv\Scripts\python.exe tools/summarize_covid19_20.py --assume-mm
```

CSV dùng UTF-8 BOM để hỗ trợ Excel. Tổng hợp từng mask một; không nạp cả cohort vào RAM.
Script không train model, không sửa voxel gốc, không tải hoặc xử lý thêm Zenodo20.
