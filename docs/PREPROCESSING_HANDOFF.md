# Bàn giao preprocessing CT + GT

**Ngày thực hiện: 21/11/2026. Trạng thái: READY_TO_USE cho preprocessing CPU.**
Đã fit thật và kiểm tra reader/loader trên dữ liệu thật. Không có kết luận về chất
lượng mô hình hoặc khả năng chạy training trên GPU. Source ở checkout hiện tại
`MADGNet`; chưa commit, push, chuyển nhánh hoặc merge.

## Đọc theo thứ tự nào?

1. Tài liệu này: dữ liệu đi qua những bước nào và lệnh sử dụng.
2. `configs/merit_preprocessing.yaml`: cấu hình chuẩn cho **lần train mới**, fold0.
3. `analysis/preprocessing/fit_stats.json`: bộ tham số đã fit từ127 CT TRAIN.
4. `src/merit_covid/data/pipeline/preprocessing.py`: reader/transform chung.
5. `dataset.py`, `slab.py`, `dataloader.py` cùng thư mục: cache một volume, chọn11 lát,
   tạo batch. Không có Dataset hoặc loader V3 thứ hai.
6. `analysis/preprocessing/CHANGE_SUMMARY.md` và `TEST_REPORT.md`: thay đổi và bằng chứng.

## Luồng dữ liệu duy nhất

```text
Manifest + saved 5-fold split
  ├─ CT + GT (train/validation): inspect_pair kiểm header/shape/affine/orientation
  └─ CT-only (inference): require_mask=False, không mở GT
       ↓
CT và GT → cùng phép hoán vị/lật về LPS, không nội suy
       ↓
CT scaled float32 hữu hạn; GT scaled float64 kiểm chính xác {0,1} trước cast float32
       ↓
CT: y = (clip(x,L,U)-mean)/std     GT: giữ nguyên nhãn, tọa độ và số voxel lesion
       ↓
XYZ → ZYX trong RAM, native512×512, không resize/crop/resample
       ↓
CTSlabDataset cache một volume → master +11 lát → DataLoader CPU
       ↓
Trainer / validation / CT-only inference hiện hữu dùng cùng PreprocessConfig
```

LPS ở đây là cách sắp xếp các trục mảng. Affine luôn được cập nhật đồng bộ và metadata
cho phép đảo mapping về lưới gốc; không đổi vị trí vật lý bằng nội suy. Ca đã LPS giữ
nguyên; LPI chỉ lật trục cần thiết. GT không bị clipping, Z-score, morphology hay resize.

## Một bộ stats chung, đã fit thật

| Tham số | Giá trị |
| --- | ---: |
| L — percentile0.1 | -1241.0 |
| U — percentile99.9 | 1032.0 |
| Mean sau clipping | -583.0622212292245 |
| Population std sau clipping | 488.0394718302036 |
| Tổng voxel127 TRAIN | 2.309.488.640 |
| Voxel dùng FIT | 1.835.675.740 |
| Voxel loại khỏi FIT | 473.812.900 |
| Voxel đúng−2048 còn giữ trong FIT | 9 |

Fit **chỉ từ127 TRAIN**, không dùng32 VAL/40 TEST hoặc GT. Runtime guard chặn mọi
`nib.load` ngoài whitelist TRAIN CT trong lần fit này; đúng127 CT được đọc hai lượt.
Không tải dữ liệu. Nguồn cache: `MedOtter/COVID-19-20`, revision
`e6ce5e1c01f59b29843e6a969655a42d33a52729`. Hash398 CT/GT sau công việc vẫn khớp
bản cache đã xác minh ở audit trước. Dataset, manifest và split không bị ghi lại.

Mỗi voxel mọi CT đều nhận cùng công thức trên, gồm padding và các ca UNKNOWN.
Ví dụ−2048 bị clip xuống L rồi trở thành khoảng **−1.348124**, không bị đổi thành0.
Không dò padding khi validation/test/inference, không fit lại theo ảnh hoặc theo nhãn.
Không có mean/std riêng từng bệnh nhân. Giá trị là NIfTI-scaled, **chưa xác minh HU**.

## Miền FIT: padding được xác định như thế nào?

Đây là quy tắc **CT-only, bảo thủ, có tiêu chí rõ**, không phải xác nhận giải phẫu:

- Trên từng lát LPS, tìm thành phần **4-connected** của giá trị chính xác−2048 chạm biên.
- Mọi vùng bị loại phải có neo ở góc; đủ cả bốn góc; không có component chạm biên riêng
  mà không neo góc. Có thể có1–4 component. Không dùng vòng tròn cố định.
- Mask boundary-connected phải **giống từng bit qua mọi lát**, ít nhất3 lát.
- Diện tích loại nằm trong5–30% mặt phẳng; không vào hình vuông trung tâm50%; độ sâu
  vào ảnh không vượt20% kích thước cạnh nhỏ. Các mức này khóa trước lần fit, không tune GT.
- Đạt đủ → `CONFIDENT` **theo rule**, chỉ loại mask đó khỏi FIT.
- Không có−2048 → `UNKNOWN`; đổi vị trí theo Z, ăn sâu, thiếu neo góc, component phụ
  hoặc không đạt giới hạn → `AMBIGUOUS`. **Cả ca dùng mọi voxel hữu hạn**, không loại
  một phần bằng cách đoán. CSV ghi `unverified_background_included=true` và lý do.
- Không suy mã−1024 hoặc mode của biên là padding. Không loại−2048 rời rạc bên trong.
- Nếu hơn20% TRAIN cần fallback, fit trả `BLOCKED`, không xuất bộ stats dùng được.

Kết quả thật: **120 CONFIDENT, 7 UNKNOWN, 0 AMBIGUOUS**. Năm pattern noncircular
đã audit trước đều qua cùng tiêu chí outer-band/stability, không ép chúng thành circle.
Bảy ca UNKNOWN đều vì không có−2048: A-0402, A-0494, A-0548, A-0570, A-0586,
A-0623, A-0698. Chúng được giữ toàn bộ; không coi là bảy CT hỏng. Danh sách đầy đủ,
số voxel và lý do ở `case_padding_audit.csv`.

CONFIDENT là mức tin cậy **của heuristic được khai báo**, không chứng minh mọi voxel
bị loại chắc chắn là padding theo DICOM. Sentinel không bị xóa khỏi input, CT hay GT;
mask FIT chỉ tồn tại trong RAM khi tính thống kê.

## Tính thống kê bằng CPU

Lượt1 tính histogram gộp voxel được chấp nhận. Lượng tử về bội gần nhất của1/16
đơn vị scaled (ties-to-even), đếm chính xác, rồi dùng rank tuyến tính như percentile
`linear`. Không sample ngẫu nhiên, không seed. Sai số tuyệt đối của phân vị so với
population float32 không quá **1/32 =0.03125 đơn vị scaled**; giá trị integer nằm đúng
trên lưới. Đây không phải độ chính xác HU được xác minh.

Lượt2 clip **giá trị accepted thật** về L/U, dùng Chan/Welford float64 tính mean/std
population. Không lấy moment từ histogram đã lượng tử. Accounting hai lượt phải khớp.
Chỉ giữ một CT float32 mỗi lượt (cap mặc định768MiB cho mảng đó), block4 lát và các
mask XY packbits. Không concatenate cả cohort; cap không phải số đo peak RAM toàn process.
Các bản gzip được đọc một lần mỗi volume/lượt, không tạo199 NIfTI mới hoặc cache ảnh đã sửa.

Fit mất **150,08 giây**; QC6 ca và kiểm hash cuối mất **35,38 giây** trên máy hiện tại.
Không coi đó là cam kết thời gian cho máy khác. Đã giữ riêng log/scope/hash và code SHA.

## Cách dùng: hai lệnh CPU và cấu hình cho người train

Chạy tại root repository với dependency trong `pyproject.toml`; dùng Python của
environment có Torch nếu muốn đọc batch. Không cần GPU. Trên máy này dùng
`.venv-gpu/Scripts/python.exe`; tên environment không có nghĩa là bước này dùng GPU.

**Lệnh fit một lần — đã chạy ở máy hiện tại:**

```powershell
python scripts/fit_train_zscore.py --config configs/merit_native512.yaml --fold 0 --policy train_fovaware_clipped_zscore_v1 --output-dir analysis/preprocessing
```

Không chạy lại vào thư mục đang có stats: CLI từ chối ghi đè. Người nhận ZIP dùng bộ
stats đã đóng băng nếu manifest/split/source bytes khớp; nếu cần fit lại chủ động chọn
thư mục output mới rồi chỉnh đường dẫn `zscore_fit` của config chuẩn, không ghi đè run cũ.

**Kiểm tra một batch CPU, không tạo model:**

```powershell
python scripts/read_training_batch.py --config configs/merit_preprocessing.yaml --fold 0 --batch-size 1 --workers 0
```

Đã chạy lệnh này thật: shape`[1,1,11,512,512]`, float32, device CPU. Master đầu tiên
của A-0003 có GT rỗng hợp lệ, không phải output dự đoán của mô hình. GT ở các ca
đại diện có lesion đã được kiểm binary/count/tọa độ và roundtrip nguyên vẹn.

Người train dùng **cùng config chuẩn** với `scripts/train_colab.py` hoặc
`scripts/merit_pipeline.py train`. Không có lệnh train nào được chạy trong công việc này.
Chọn output mới và không truyền Native checkpoint vào V3. Checkpoint fingerprint
phải khác khi đổi intensity policy; chương trình không migration checkpoint Native sang V3.

Đường dẫn CT/GT nằm trong manifest; `data.manifest`, `data.split_path`, và đường dẫn
stats trong YAML resolve theo vị trí YAML. Trên Drive/Kaggle phải cấu hình đường dẫn
thực mà người chạy có quyền truy cập. Không tự mount, tải, thay sharing hoặc giả định
tài khoản đã được cấp quyền. Manifest/split giữ nguyên bytes để binding stats khớp.

## Đã kiểm chứng và chưa kiểm chứng

- PASS: CPU-fit127 TRAIN; read guard; SHA TRAIN trước/sau; raw398 hash khớp cache cũ;
  stats/case/fold/manifest/split binding; source Native effective config/fingerprint giữ nguyên.
- PASS:6 TRAIN đại diện A-0074_1, A-0120, A-0295, A-0402, A-0599, A-0698; V3 đúng
  công thức, GT binary/count/tọa độ giữ nguyên, đảo về GT gốc chính xác. Có LPI,
  outlier, noncircular và ca không sentinel; CT-only inference không mở GT hoạt động.
- PASS: loader và CLI đọc batch hiện có. Không có reader/dataset/loader thứ hai.
- Chưa train, chưa thử model/GPU hoặc checkpoint V3 thật; không có Dice mới từ CT thật.
- Ngoại lệ kiểm thử: lượt regression đầu chọn lọc chưa đúng, đã chạy kèm hai test
  metric trên dữ liệu **giả** (một CLI và một evaluator CPU). Không train hoặc dùng
  CT thật đánh giá. Đã ghi rõ ở TEST_REPORT; các lượt sau loại test metric.

Clipping làm mất phân biệt ở tails ngoài L/U. UNKNOWN còn background không xác minh
trong miền fit. Cả hai là giới hạn có chủ ý, không chứng minh tăng Dice hoặc chữa
collapse epoch1. Tính độc lập hiện chỉ theo volume/group, không tự nhận theo bệnh nhân.
Ba header unknown-unit vẫn giữ giả định `assume_mm=true`, không biến thành đơn vị verified.

## Dọn mã và bảo vệ lịch sử

Không tìm được tệp preprocessing nguyên vẹn nào có thể xóa an toàn mà không đụng
API/tests hoặc source/run lịch sử. Đã xóa hai alias import private không còn người gọi
trong Dataset và một nhánh unreachable ở geometry. `CHANGE_SUMMARY.md` ghi chi tiết.
Giữ Native, global train_zscore và API bridge đang có references; chúng phục vụ tái lập,
không phải ba pipeline mới. Các source/notebook/report/checkpoint cũ trong outputs
không bị sửa hoặc xóa. Không thêm notebook train mới hay bản source backup song song.
