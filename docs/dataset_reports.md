# Đọc CSV, hình ảnh và biểu đồ của hai dataset

Các báo cáo này giúp hiểu dữ liệu trước khi huấn luyện: volume dài/ngắn ra sao,
spacing là bao nhiêu, có bao nhiêu lát tổn thương và nhãn có khớp CT hay không.
Chúng chưa phải kết quả phân đoạn của mô hình AI.

## 1. Báo cáo Zenodo20 đang có những gì?

Đây là các báo cáo có sẵn từ repository gốc, nằm ở `analysis/zenodo20/`. Lần bổ sung
báo cáo COVID-19-20 không tải hoặc xử lý lại các ảnh CT/mask Zenodo20.

| File | Nội dung và cách hiểu |
| --- | --- |
| [case_statistics.csv](../analysis/zenodo20/case_statistics.csv) | 20 dòng, mỗi dòng một volume. Có mã ca, nguồn Coronacases/Radiopaedia, shape, số lát, spacing, orientation, intensity mẫu, số lát có/không có tổn thương, số voxel lesion/phổi và các kiểm tra nhãn. |
| [summary.json](../analysis/zenodo20/summary.json) | Tổng hợp toàn bộ 20 volume: số lát, tỷ lệ tổn thương, phân bố spacing, kích thước vùng tổn thương, độ liên tục qua các lát và ảnh hưởng của resize. |
| [dataset_overview.png](../analysis/zenodo20/dataset_overview.png) | Biểu đồ trên: mỗi cột là một volume, phần đỏ là lát có tổn thương, phần xám là lát không có. Biểu đồ dưới: phần trăm voxel lesion trên số voxel phổi, màu điểm biểu diễn spacing Z. |
| [lesion_trajectories.png](../analysis/zenodo20/lesion_trajectories.png) | Ba đường ví dụ: trục ngang là khoảng cách từ lát đầu, trục dọc là số pixel tổn thương trên mỗi lát. Đường cao hơn nghĩa là vùng được đánh dấu lớn hơn trên lát đó. |
| [README.md](../analysis/zenodo20/README.md) | Nguồn dữ liệu, cách tạo báo cáo, định nghĩa thống kê và giới hạn kiểm chứng. |

Báo cáo sẵn có ghi 20 volume từ hai nguồn, mỗi nguồn 10 ca; tổng 3.520 lát, gồm
1.844 lát có tổn thương và 1.676 lát có mask rỗng.

Một số thuật ngữ trong báo cáo Zenodo20:

| Thuật ngữ / cột | Ý nghĩa |
| --- | --- |
| Positive slice | Lát có ít nhất một voxel được GT đánh dấu tổn thương. |
| Empty slice | Lát có GT mask rỗng; không phải file CT rỗng hoặc hỏng. |
| Infection fraction of lung | Số voxel lesion chia cho số voxel phổi; cần có lung mask. |
| Component / `components_2d` | Một vùng foreground nối liền nhau trên một lát, theo kết nối 8 hướng. Một lát có thể có nhiều vùng rời nhau. |
| Adjacent Dice | Dice giữa **hai GT mask của hai lát liền kề** khi cả hai có tổn thương. Mô tả sự thay đổi của nhãn qua lát, không phải Dice của model. |
| Positive run | Một đoạn gồm các lát liên tiếp đều có tổn thương. |
| Resize loss | Vùng tổn thương mất khỏi mask sau khi thu nhỏ bằng nearest neighbor. Báo cáo kiểm tra kích thước 128, 256 và 352. |
| Intensity percentiles | Các mức phân vị cường độ CT, tính từ mẫu lấy theo stride (8,8,4), không phải percentile chính xác của mọi voxel. |
| Geometry match | CT và các mask có cùng shape, affine và spacing. |

Zenodo20 có mask phổi, lesion mask và combined mask nên tính được tỷ lệ lesion trên
phổi và voxel lesion nằm ngoài lung mask. Báo cáo ghi 114 voxel ngoài lung mask
ở ba volume Radiopaedia. Đây là mô tả sự khác nhau giữa các nhãn được cung cấp.
README cũng ghi CT nguồn mirror chưa được chứng minh giống từng byte với CT ZIP
gốc Zenodo. Cần giữ giới hạn này khi sử dụng các báo cáo cũ.

## 2. Báo cáo mới của COVID-19-20

Trước đây phần này mới có audit JSON và hai ảnh smoke test. Đã bổ sung báo cáo
để mở bằng Excel và nhìn tổng quan như Zenodo20:

| File | Nội dung |
| --- | --- |
| [case_statistics.csv](../analysis/covid19_20/case_statistics.csv) | 199 dòng tương ứng 199 volume; metadata gốc, số lát, spacing, intensity từ audit, thống kê GT, fold membership và cờ giả định đơn vị. |
| [summary.json](../analysis/covid19_20/summary.json) | Tổng hợp cohort và số voxel lesion của từng lát trong mỗi volume. |
| [dataset_overview.png](../analysis/covid19_20/dataset_overview.png) | Phân bố spacing, số lát mỗi volume, tỷ lệ lesion trên toàn CT và số lát có/không có lesion. |
| [lesion_trajectories.png](../analysis/covid19_20/lesion_trajectories.png) | Diện tích GT theo khoảng cách qua các lát, chọn ca ví dụ theo case ID trong từng nhóm spacing. |
| [ct_mask_examples.png](../analysis/covid19_20/ct_mask_examples.png) | Mỗi hàng là một ca ví dụ, gồm CT, GT và overlay. Dùng lát giữa, không chọn theo vị trí tổn thương. |
| [REPORT.md](../analysis/covid19_20/REPORT.md) | Bản tổng kết tiếng Việt, cách đọc các cột, số liệu thực tế và giới hạn. Nên mở file này trước. |

COVID-19-20 chỉ có lesion mask trong dữ liệu đang sử dụng, nên mẫu số của tỷ lệ
lesion là **toàn bộ khối CT**, bao gồm cả không khí ngoài cơ thể. Không so sánh trực
tiếp tỷ lệ này với tỷ lệ lesion/phổi của Zenodo20. Các chỉ số lung mask, components
và resize loss không được tính trong bản báo cáo mới; không điền số giả để giống
schema Zenodo20.

Ba ca thiếu đơn vị không gian được đánh dấu `spatial_units_assumed_mm=True`.
Patient ID chưa xác minh được để trống. Chỉ số `fold_0` đến `fold_4` mô tả volume
ở partition nào trong từng fold; chưa bảo đảm bệnh nhân có nhiều volume được gom
chung nếu chưa có ánh xạ patient đáng tin cậy.

## 3. Thứ tự đọc gợi ý

1. Mở `REPORT.md` của COVID-19-20 để xem số liệu và giới hạn.
2. Mở `case_statistics.csv` bằng Excel, mỗi hàng xem một volume. Các cột cơ bản ở
   đầu bảng; các cột `audit_*` cuối bảng giữ metadata chi tiết, gồm cả affine.
3. Xem `dataset_overview.png` để biết dữ liệu phân bố như thế nào.
4. Xem `ct_mask_examples.png` để đối chiếu CT với GT; overlay là mask có sẵn chồng
   lên CT, chưa phải dự đoán AI.
5. Xem `lesion_trajectories.png` để hiểu tổn thương GT thay đổi qua các lát.

Lệnh tạo lại báo cáo primary từ thư mục gốc repository:

```powershell
.\.venv\Scripts\python.exe tools/summarize_covid19_20.py --assume-mm
```

Lệnh đọc lại từng mask gốc, kiểm tra headers, đối chiếu audit rồi tạo báo cáo;
không train model hoặc thay đổi CT/mask gốc.
