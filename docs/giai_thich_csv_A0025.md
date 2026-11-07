# Giải thích đủ 71 trường CSV bằng một volume thật

Volume xuyên suốt: **`volume-covid19-A-0025`**, dòng `volume_order = 6` trong
[case_statistics.csv](../analysis/covid19_20/case_statistics.csv).
Ngày đối chiếu: **07/11/2026**, cấu hình LPS và giữ kích thước gốc.

Các giá trị dưới đây lấy từ CSV thật. Số thập phân được làm tròn để dễ đọc;
CSV giữ số đầy đủ. Trong văn bản, dấu phẩy biểu thị phần thập phân, dấu chấm
phân cách hàng nghìn. CSV dùng dấu chấm cho phần thập phân.

## 1. Trước khi đọc các cột: hình dung volume này

Đây là một file CT 3D gồm **61 lát**. Mỗi lát có **512 × 512 pixel**. Xếp 61 lát
lại thành một khối, mỗi ô nhỏ của khối là một **voxel**. File mask đi kèm đánh dấu
voxel nào thuộc tổn thương bằng giá trị 1; background là 0.

Luồng xử lý của ví dụ:

```text
CT gốc và mask gốc
  Mỗi lát 512 × 512, tổng 61 lát, orientation LPS
                       ↓
Thống nhất LPS: volume này đã LPS nên giữ nguyên thứ tự CT/mask
  Mỗi lát vẫn 512 × 512, vẫn 61 lát, affine không đổi
                       ↓
Preprocessing đầu vào: chỉ window/normalize cường độ CT
  Mỗi lát vẫn 512 × 512, vẫn 61 lát; không resize hoặc nội suy Z
                       ↓
Lấy 11 lát quanh một master để tạo slab
  Ví dụ master 30 → lấy các lát có chỉ số 25–35
```

**Thống kê số lát và voxel tổn thương trong CSV được tính từ mask ở độ phân giải
gốc.** Các trường `audit_*` còn lưu cấu hình preprocessing; kích thước gốc và
kích thước đầu vào của ca này đều là 512×512 vì resize mặc định đã tắt.
Trong mọi ví dụ chỉ số lát bắt đầu từ **0**.

## 2. Nhận diện: đang xem volume nào? — trường 1–7

| STT | Trường | Giá trị của volume này | Ý nghĩa |
| ---: | --- | --- | --- |
| 1 | `volume_order` | 6 | Dòng thứ 6 trong danh sách dữ liệu CSV, đồng thời là số thứ tự trên biểu đồ có trục volume order. Không phải mức độ bệnh hay patient ID. |
| 2 | `case_id` | `volume-covid19-A-0025` | Mã của cặp CT/mask đang xem. |
| 3 | `patient_id` | Trống | Chưa có patient ID được xác minh. Ô trống không có nghĩa ảnh không thuộc bệnh nhân nào. |
| 4 | `patient_ids_available` | `False` | Manifest/ánh xạ hiện tại chưa cung cấp patient ID cho pipeline. |
| 5 | `image` | `Train/volume-covid19-A-0025_ct.nii.gz` | Đường dẫn CT, tính từ thư mục chứa manifest. |
| 6 | `mask` | `Train/volume-covid19-A-0025_seg.nii.gz` | Đường dẫn mask GT tương ứng. |
| 7 | `group_id` | `volume-covid19-A-0025` | Nhãn nhóm phải được giữ chung khi chia tập. Hiện bằng case ID vì chưa có ánh xạ bệnh nhân. |

`group_id` trả lời: **những volume nào phải đi cùng nhau khi chia tập?** Nếu sau
này xác minh hai lần chụp đều thuộc bệnh nhân P, hai volume cần cùng group P.
Hiện mỗi volume là một group riêng, nên chỉ bảo đảm các lát/slab của volume này
ở cùng một partition trong một fold. Chưa bảo đảm gom các volume khác của cùng
bệnh nhân. Group ID không phải fold ID.

## 3. Khối ảnh có hình học ra sao? — trường 8–17

### 3.1. Kích thước đếm bằng số voxel

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 8 | `original_shape_x` | 512 | Số voxel theo trục thứ nhất của mảng NIfTI gốc. |
| 9 | `original_shape_y` | 512 | Số voxel theo trục thứ hai của mảng NIfTI gốc. |
| 10 | `original_shape_z` | 61 | Số voxel theo trục thứ ba của mảng NIfTI gốc; ở volume này chính là số lát axial. |

Kết hợp ba trường: **512 × 512 × 61**, không có đơn vị mm. Chúng đếm số ô.

### 3.2. Orientation nói các trục hướng về đâu trong cơ thể

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 11 | `original_orientation` | `LPS` | Khi tăng chỉ số của ba trục gốc, hướng gần nhất lần lượt về trái (L), sau (P), phía đầu (S) của người bệnh. |
| 12 | `canonical_orientation` | `LPS` | Hướng đã thống nhất: trái (L), sau (P), phía đầu (S); ca này giữ nguyên hướng gốc. |

L/R là trái/phải của **người bệnh**, A/P là trước/sau, S/I là phía đầu/phía chân.
Đây không phải chỉ dẫn trái/phải của người đang nhìn màn hình.

Với volume này, hướng gốc đã là LPS nên không đảo trục; mảng CT/mask và affine
được giữ nguyên ở bước thống nhất hướng. Trong cohort có 193 volume LPS và 6
volume LPI; chỉ nhóm LPI cần đảo Z để về LPS. Shape ca này vẫn **512×512×61**.
Nếu một volume khác cần hoán vị trục thì các số shape có thể đổi vị trí, nhưng
đó vẫn là sắp lại các voxel, không phải resize. Bước này cũng không tự deoblique
một ảnh đang nghiêng so với các trục cơ thể.

### 3.3. Spacing đo khoảng cách thật giữa các tâm voxel

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 13 | `spacing_x_mm` | ≈0,846 mm | Khoảng cách tâm hai voxel liền kề theo trục thứ nhất gốc. |
| 14 | `spacing_y_mm` | ≈0,846 mm | Khoảng cách tâm hai voxel liền kề theo trục thứ hai gốc. |
| 15 | `spacing_z_mm` | 5 mm | Khoảng cách tâm hai voxel liền kề theo trục thứ ba gốc; ở đây là bước qua hai lát. |
| 16 | `spatial_unit_in_header` | `mm` | Header của volume có ghi rõ đơn vị millimet. |
| 17 | `spatial_units_assumed_mm` | `False` | Riêng volume này không cần giả định đơn vị. |

Bạn có thể hình dung lưới như các viên gạch dài theo Z: bước X/Y là 0,846 mm,
bước Z là 5 mm. Các bước trên cùng một trục đều nhau; ba trục không bắt buộc cùng
spacing. Không nên đồng nhất spacing Z với độ dày lát chụp khi chưa có thông tin
độ dày từ nguồn.

Ba volume khác của cohort có header đơn vị `unknown`; chỉ những ca đó được đánh
dấu `spatial_units_assumed_mm=True` khi bật tùy chọn giả định mm. Ca A-0025 có
đơn vị mm thật trong header nên cờ này là False.

## 4. Các số trong CT/mask được lưu và đọc thế nào? — trường 18–27

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 18 | `ct_dtype` | `int16` | CT lưu bằng số nguyên có dấu 16-bit; kiểu này có thể biểu diễn cả số âm. |
| 19 | `mask_dtype` | `uint16` | Mask lưu bằng số nguyên không dấu 16-bit. Kiểu lưu không cho biết có bao nhiêu nhãn thực tế. |
| 20 | `ct_min` | −2048 | Giá trị CT nhỏ nhất sau scaling NIfTI, trước clipping/normalize. Lấy từ audit đầy đủ trước đó. |
| 21 | `ct_max` | 1672 | Giá trị CT lớn nhất ở cùng giai đoạn với ct_min. |
| 22 | `ct_scaling_slope` | 1 | Hệ số nhân khi đọc số CT từ file. |
| 23 | `ct_scaling_intercept` | 0 | Hằng số cộng khi đọc số CT từ file. |
| 24 | `mask_label_values` | `[0,1]` | Hai giá trị thực sự có trong mask: background 0 và lesion 1. |
| 25 | `geometry_match` | `True` | Headers CT/mask hiện tại khớp shape, affine, orientation và spacing theo kiểm tra pipeline. |
| 26 | `mask_finite_binary` | `True` | Đã đọc toàn bộ mask hiện tại: không có NaN/Inf, chỉ có 0/1. |
| 27 | `z_resampled` | `False` | Không nội suy trục Z để tạo lại hoặc tăng số lát. |

Scaling NIfTI được hiểu là:

```text
Cường độ sau scaling = số lưu trong file × slope + intercept
                     = số lưu trong file × 1 + 0
```

Ở ca này scaling không đổi các số. Sau đó mới có bước window/normalize đầu vào
(xem trường 64). Vì thế CT đầu vào Dataset có thể nằm trong [0,1] trong khi CSV
vẫn ghi ct_min = −2048 và ct_max = 1672: chúng mô tả hai giai đoạn khác nhau.

## 5. Tổn thương xuất hiện ở những lát nào? — trường 28–36

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 28 | `slices` | 61 | Số lát axial sau thống nhất LPS, trước padding. Ở ca này bằng original_shape_z. |
| 29 | `positive_slices` | 40 | Có 40 lát chứa ít nhất một voxel GT bằng 1. |
| 30 | `empty_slices` | 21 | Có 21 lát mà toàn bộ GT trên lát bằng 0. Không có nghĩa ảnh CT bị rỗng/hỏng. |
| 31 | `positive_slice_fraction` | ≈0,655738 | 40/61 ≈65,57% **số lát** có tổn thương. |
| 32 | `first_positive_slice` | 10 | Chỉ số canonical đầu tiên có lesion, tức lát thứ 11 nếu đếm từ 1. |
| 33 | `last_positive_slice` | 49 | Chỉ số canonical cuối có lesion, tức lát thứ 50 nếu đếm từ 1. |
| 34 | `positive_run_count` | 1 | Chỉ có một đoạn liên tiếp gồm các lát có lesion. |
| 35 | `max_positive_run_length` | 40 | Đoạn liên tiếp dài nhất có 40 lát. |
| 36 | `positive_run_lengths` | `[40]` | Danh sách độ dài các đoạn; ca này có duy nhất đoạn dài 40. |

Đọc các trường này cùng nhau sẽ có hình dung:

```text
Chỉ số lát:  0 ........ 9 | 10 ...................... 49 | 50 ........ 60
GT:            10 lát 0  |        40 lát có lesion       |    11 lát 0
```

40 + 21 = 61; 49 − 10 + 1 = 40. Tỷ lệ 65,57% nói về **số lát**, không có nghĩa
65,57% thể tích CT hoặc 65,57% phổi bị tổn thương.
Các chỉ số đầu/cuối theo orientation canonical; với ca đảo Z, chúng có thể khác
chỉ số của file gốc. Riêng A-0025 không đảo Z.

## 6. Vùng GT tổn thương lớn bao nhiêu? — trường 37–43

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 37 | `lesion_voxels` | 402.334 | Số voxel GT bằng 1 trong toàn volume gốc. |
| 38 | `total_voxels` | 15.990.784 | Tổng số voxel = 512×512×61. |
| 39 | `lesion_fraction_volume` | ≈0,0251604 | 402.334/15.990.784 ≈2,516% số voxel của toàn bộ CT. |
| 40 | `voxel_volume_mm3` | ≈3,578580 mm³ | Thể tích một voxel, tính từ determinant affine. |
| 41 | `lesion_volume_ml` | ≈1.439,784 mL | Tổng thể tích GT lesion: số voxel lesion × thể tích một voxel, đổi sang mL. |
| 42 | `pixel_area_mm2` | ≈0,715716 mm² | Diện tích một pixel trên mặt phẳng lát canonical, tính từ hai vector affine trong mặt phẳng. |
| 43 | `slice_center_step_mm` | 5 mm | Độ dài vector dịch tâm từ một lát canonical sang lát kế tiếp. |

Với lưới thẳng trục của ca này, có thể tự kiểm tra:

```text
Diện tích pixel ≈ 0,846 × 0,846 = 0,715716 mm²
Thể tích voxel ≈ 0,846 × 0,846 × 5 = 3,578580 mm³
Thể tích lesion ≈ 402.334 × 3,578580 / 1.000 = 1.439,784 mL
```

1 mL = 1.000 mm³. Code dùng đầy đủ affine để xử lý cả trường hợp ảnh oblique/shear.
Spacing Z gốc và slice_center_step_mm có thể ứng với trục khác nhau nếu cần hoán
vị trục; ở ca này chúng đều bằng 5 mm.

**Mẫu số của lesion_fraction_volume là toàn bộ khối CT**, bao gồm vùng ngoài phổi
và không khí ngoài cơ thể trong trường ảnh. Bộ primary hiện không có lung mask,
nên không được đọc con số 2,516% thành “2,516% phổi bị tổn thương”.

## 7. Nhãn có thay đổi nhiều giữa hai lát liền nhau không? — trường 44–45

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 44 | `adjacent_positive_pairs` | 39 | Số cặp liền kề mà cả hai lát đều có GT lesion. Đoạn 40 lát liên tục tạo 39 cặp: (10,11)…(48,49). |
| 45 | `adjacent_gt_dice_median` | ≈0,833942 | Median của Dice giữa hai GT mask trong 39 cặp đó. |

Dice ở đây so **GT lát k với GT lát k+1 tại cùng tọa độ pixel**, không có model,
không đăng ký chuyển động để căn hai lát. Nó mô tả mức chồng lấp nhãn qua lát.
Giá trị ≈0,834 không phải điểm dự đoán của mô hình và không có nghĩa “mô hình đúng
83,4%”. Median là giá trị nằm giữa khi sắp 39 điểm Dice theo thứ tự.

## 8. Vì sao có các cột audit_*? — trường 46–66

Chúng giữ bản sao metadata/cấu hình của audit trước đó, giúp kiểm tra nguồn gốc
và khôi phục geometry. Một số lặp ý nghĩa với cột cơ bản nhưng được giữ nguyên
dạng list/matrix/dictionary trong một ô CSV.

### 8.1. Shape và tọa độ

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 46 | `audit_original_shape_xyz` | `[512,512,61]` | Shape gốc viết gộp thành một danh sách theo thứ tự X,Y,Z. |
| 47 | `audit_canonical_shape_xyz` | `[512,512,61]` | Shape sau sắp orientation, trước resize; ca này không đổi kích thước. |
| 48 | `audit_original_affine` | Ma trận A bên dưới | Chuyển chỉ số voxel gốc thành tọa độ vật lý; đơn vị theo header. |
| 49 | `audit_original_affine_mm` | Cũng là A | Affine gốc sau quy đổi đơn vị về mm. Header đã là mm nên bằng trường 48. |
| 50 | `audit_canonical_affine_mm` | Ma trận B bên dưới, bằng A | Chuyển chỉ số voxel LPS thành tọa độ world RAS+ theo mm; ca này không đổi affine. |
| 51 | `audit_canonical_to_original_orientation` | `[[0,1],[1,1],[2,1]]` | Phép đồng nhất: giữ cả ba trục vì ảnh gốc đã LPS. |
| 52 | `audit_original_orientation` | `["L","P","S"]` | Orientation gốc lưu dưới dạng danh sách. |
| 53 | `audit_canonical_orientation` | `["L","P","S"]` | Orientation đã thống nhất LPS lưu dưới dạng danh sách. |
| 54 | `audit_native_spacing_mm` | `[0.846…,0.846…,5]` | Spacing native canonical, trước resize XY, đã đổi/giả định đơn vị mm theo cấu hình audit. |

Các affine sau được làm tròn để minh họa; số chính xác nằm trong CSV:

```text
A — original affine, đồng thời là original affine mm:
[[-0.846,  0,      0,   216.373001],
 [ 0,     -0.846,  0,   216.373398],
 [ 0,      0,      5,  -252.5     ],
 [ 0,      0,      0,     1       ]]

B — canonical affine mm:
[[ -0.846, 0,      0,   216.373001],
 [ 0,     -0.846,  0,   216.373398],
 [ 0,      0,      5,  -252.5     ],
 [ 0,      0,      0,     1       ]]
```

Bạn không cần học phép nhân ma trận ngay để đọc CSV. Vai trò của affine là trả
lời: **ô có chỉ số (i,j,k) nằm ở tọa độ vật lý nào?** Trong A, hệ số −0,846 nghĩa
là tăng chỉ số trục đó một đơn vị thì tọa độ tương ứng giảm khoảng 0,846 mm.
Số ở cột cuối xác định vị trí của voxel có chỉ số (0,0,0); không phải kích thước
ảnh hoặc spacing.

Ví dụ cùng một voxel:

```text
Trong mảng gốc:      (0, 0, 0)
Trong mảng standardized LPS: (0, 0, 0)
```

Ca này không cần đổi chỉ số hoặc affine. Với 6 ca LPI phải đảo Z, chỉ số lát thay
đổi nhưng affine được cập nhật để voxel vẫn ở cùng vị trí vật lý. Hướng mảng LPS
khác với hệ tọa độ world RAS+ mà affine của NiBabel sử dụng.

### 8.2. Kích thước xử lý và đơn vị

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 55 | `audit_processed_size_hw` | `[512,512]` | Kích thước H,W đầu vào; bằng kích thước gốc vì không resize. |
| 56 | `audit_native_size_hw` | `[512,512]` | Kích thước H,W native canonical, trước resize. |
| 57 | `audit_spatial_unit_in_header` | `mm` | Bản sao đơn vị thật trong header của ca này. |
| 58 | `audit_assume_mm` | `True` | Khi audit cả cohort, tùy chọn cho phép coi các ca thiếu đơn vị là mm đã được bật. |

**audit_assume_mm=True không mâu thuẫn với spatial_units_assumed_mm=False.** Cột
audit nói tùy chọn được bật cho chương trình; cột spatial_units_assumed_mm nói
**ca cụ thể này có cần sử dụng giả định không**. A-0025 có đơn vị mm sẵn nên không
cần. Chỉ ba ca thiếu đơn vị mới thực sự sử dụng giả định.

Mặc định `size_hw=None` nên không thay đổi lưới lấy mẫu. Chỉ khi caller chủ động
truyền kích thước H,W mới, pipeline mới resize; khi đó spacing native trong CSV
vẫn mô tả ảnh gốc, không được dùng như spacing của ảnh đã resize.

### 8.3. Intensity, mask và cách tạo slab

| STT | Trường | Giá trị | Ý nghĩa |
| ---: | --- | --- | --- |
| 59 | `audit_scaled_intensity_range` | `[-2048,1672]` | Min/max CT sau scaling NIfTI, trước window/normalize; tương ứng trường 20–21. |
| 60 | `audit_intensity_encoding` | `auto_hu` | Dùng kiểm tra range để phát hiện ảnh không có encoding phù hợp trước khi xử lý như HU. Đây là sanity check, không chứng minh calibration từ DICOM. |
| 61 | `audit_label_values` | `[0,1]` | Các nhãn mask đã gặp ở audit trước. |
| 62 | `audit_native_lesion_voxels` | 402.334 | Số voxel lesion native ghi ở audit; lần tổng hợp mới đọc lại mask và đối chiếu số này. |
| 63 | `audit_nifti_scaling` | `{"slope":1,"intercept":0}` | Hai thông số scaling CT được lưu gộp thành dictionary. |
| 64 | `audit_hu_window` | `[-1000,400]` | Cấu hình clip cường độ vào khoảng này, sau đó normalize về [0,1]. |
| 65 | `audit_padding` | `replicate_boundary` | Khi lấy slab sát biên, lặp lát biên để đủ 11 lát. Không ghi thêm lát vào CT gốc. |
| 66 | `audit_z_resampled` | `False` | Cấu hình/báo cáo xác nhận không nội suy trục Z. |

Window và normalize của ca này:

```text
Sau clipping:      giới hạn cường độ trong [-1000,400]
Sau normalize:    (giá trị đã clip + 1000) / 1400

Ví dụ: -2048 → -1000 → 0
        -300 →  -300 → 0,5
        1672 →   400 → 1
```

Min/max CT được tái sử dụng từ audit đầy đủ trước đây. Script báo cáo mới kiểm
tra lại headers và toàn bộ mask; không đọc lại mọi voxel CT của 199 ca hoặc xác
minh byte identity CT bằng checksum. Vì thế audit cũ vẫn là nguồn của thống kê
intensity, không nên mô tả min/max như một phép đo CT mới trong lần xuất CSV.

Ví dụ padding, nếu lấy master có chỉ số 0:

```text
Các chỉ số slab 11 lát: [0,0,0,0,0,0,1,2,3,4,5]
```

Các số 0 lặp lại là cùng lát biên. Volume gốc vẫn chỉ có 61 lát; không tạo những
lát giả nằm bên ngoài volume. Khi master là 30, đủ context nên lấy 25–35 và không
cần lặp biên.

## 9. Volume này nằm ở tập nào qua 5 fold? — trường 67–71

Mỗi cột fold là **một lần chia độc lập**. Chọn fold nào thì chỉ đọc vai trò của ca
trong cột đó, không gộp vai trò năm cột vào một lần huấn luyện.

| STT | Trường | Giá trị | Nghĩa khi sử dụng fold đó |
| ---: | --- | --- | --- |
| 67 | `fold_0` | `train` | Ca A-0025 dùng để học trong fold 0. |
| 68 | `fold_1` | `train` | Ca A-0025 dùng để học trong fold 1. |
| 69 | `fold_2` | `test` | Ca A-0025 dành đánh giá cuối trong fold 2, không được dùng train hoặc chọn threshold của fold 2. |
| 70 | `fold_3` | `train` | Ca A-0025 dùng để học trong fold 3. |
| 71 | `fold_4` | `train` | Ca A-0025 dùng để học trong fold 4. |

Ca này được làm test một lần, train bốn lần, không nằm trong validation của lần
chia nào. Điều đó hợp lệ với cách chia hiện tại: validation được chọn từ phần
còn lại trong từng fold, không bắt buộc mỗi ca phải làm validation đúng một lần.

Để hiểu 5-fold, hình dung 199 volume được chia thành năm nhóm test A–E:

| Fold | Test | Train | Validation |
| --- | --- | ---: | ---: |
| 0 | A: 40 ca | 127 | 32 |
| 1 | B: 40 ca | 127 | 32 |
| 2 | C: 40 ca, có A-0025 | 127 | 32 |
| 3 | D: 40 ca | 127 | 32 |
| 4 | E: 39 ca | 128 | 32 |

Train/validation lấy từ các nhóm ngoài test trong fold đang xét. Seed 42 cố định
kết quả; danh sách đã có trong [split JSON](../splits/covid19_20_5fold.json).
Mọi lát/slab của A-0025 đi cùng vai trò của volume này trong từng fold.

Nếu sau này chạy đủ 5-fold, mỗi fold phải huấn luyện mô hình riêng từ đầu. Không
mang mô hình đã học ở fold trước sang fold tiếp theo để đánh giá vì nó có thể đã
thấy các ca test của fold sau. Hiện mới có danh sách chia, chưa huấn luyện.

5-fold là protocol ban đầu của dự án, không phải lựa chọn bắt buộc hoặc đã chứng
minh tối ưu hơn 3/10-fold. Nó cho năm nhóm đánh giá thay vì một, đổi lại cần năm
lần train. Chưa xác minh patient-level independence khi group vẫn là volume ID.

## 10. Các thông tin cần đọc đúng khi diễn giải dòng này

- Có 61 lát gốc/canonical và 40 lát có lesion; số 11 là số lát trong một slab,
  không phải số lát của volume.
- 65,57% là tỷ lệ **lát có lesion**; 2,516% là tỷ lệ **voxel lesion trên toàn CT**.
  Cả hai đều không phải phần trăm phổi tổn thương hoặc độ chính xác model.
- Shape và độ phân giải giữ nguyên qua pipeline mặc định: 512×512×61;
  thống kê GT và mask đầu vào đều ở độ phân giải native.
- Đơn vị của ca này là mm từ header; chỉ tùy chọn assume-mm của audit cả cohort
  được bật, không có nghĩa ca này thiếu đơn vị.
- Group hiện bằng volume ID và patient ID trống. Không suy diễn mã 0025 thành
  patient ID đã được xác minh.

Tài liệu kỹ thuật nền: [orientation](https://nipy.org/nibabel/image_orientation.html),
[reorient về canonical](https://nipy.org/nibabel/reference/nibabel.funcs.html),
[tọa độ và affine](https://nipy.org/nibabel/coordinate_systems.html),
[shape/spacing NIfTI](https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/dim.html/document_view.html).
