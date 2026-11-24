# Tổng kết quá trình tiền xử lý dữ liệu MERIT-UNETR++

Cập nhật ngày **21/11/2026**. Tài liệu tổng hợp các quyết định và kết quả quan trọng
từ đầu quá trình đến bộ bàn giao cuối, đối chiếu với code và báo cáo đã lưu.
Các phương án lịch sử được ghi riêng với cấu hình đang dùng.

## 1. Hiện tại đã hoàn thành đến đâu?

**Đã hoàn thiện và bàn giao phần preprocessing CT + GT, trạng thái `READY_FOR_TEAM_HANDOFF`.**

- Có dữ liệu gốc, manifest và split cố định.
- Có một đường reader/Dataset/DataLoader chung cho MERIT.
- Có cấu hình V3 và một bộ thống kê đã fit thật từ **127 CT TRAIN của fold 0**.
- Đã đọc đủ **199 cặp CT/GT qua API V3 production**, không chỉ kiểm tra header.
- Đã xác nhận CT sau transform hữu hạn, GT không đổi count/tọa độ và dữ liệu nguồn
  không bị ghi đè.
- Có source portable, dependencies, hướng dẫn và ZIP cuối để giao người train.

**Đây là hoàn thành phần chuẩn bị đầu vào; chưa chứng minh V3 tăng Dice hoặc huấn
luyện thành công trên GPU.**

## 2. Chuẩn bị và hiểu đúng dữ liệu gốc

### 2.1. Bộ dữ liệu chính

Dùng **COVID-19-20 gồm 199 volume CT**, mỗi volume đi cùng một GT mask phân đoạn
tổn thương. Tổng cộng **199 CT + 199 mask = 398 file NIfTI**.

- **CT:** ảnh ba chiều, gồm nhiều lát 2D, lưu giá trị cường độ.
- **GT mask:** nhãn do dataset cung cấp; `0` là nền, `1` là vùng tổn thương.
- **Volume ID:** mã ảnh/ca dữ liệu; không tự động đồng nghĩa một bệnh nhân duy nhất.

Nguồn local đã đối chiếu là Hugging Face `MedOtter/COVID-19-20`, revision
`e6ce5e1c01f59b29843e6a969655a42d33a52729`.

### 2.2. Vị trí dữ liệu trên máy

```text
C:\1_HDD\paper\merit-covid-ct\data\covid19_20
```

Các file CT và GT gốc nằm trong:

```text
C:\1_HDD\paper\merit-covid-ct\data\covid19_20\Train
```

**Tên thư mục `Train` là tên của bản phân phối dữ liệu, không có nghĩa cả199 ca đều
thuộc TRAIN của thí nghiệm.** TRAIN/VAL/TEST được xác định bằng split đã lưu.

### 2.3. Những file mô tả đã có

Manifest `data/covid19_20/cases.json` là danh mục mã ca và đường dẫn CT/GT, không phải
ảnh đã xử lý. Báo cáo/CSV mô tả shape, spacing, orientation, nhãn, số lát và phân
vùng của từng volume đã được xây dựng; schema CSV primary có71 cột được giải thích
trong [tài liệu data pipeline](data_pipeline.md#complete-saved-primary-csv-schema).

Zenodo20 đã được bàn và có các báo cáo dữ liệu riêng. Vai trò là **đánh giá ngoài**,
không tham gia fit thống kê V3 hay chọn tham số từ TEST. Không gọi nó là một phần
TRAIN của199 ca.

## 3. Chia dữ liệu bằng danh sách ID, không di chuyển ảnh

Đã lưu split cố định tại `splits/covid19_20_5fold.json`, seed42.

| Fold | TRAIN | VAL | TEST |
| --- | ---: | ---: | ---: |
| 0–3, mỗi fold | 127 | 32 | 40 |
| 4 | 128 | 32 | 39 |

- Split chỉ quy định loader được lấy ID nào; không copy ảnh vào ba thư mục mới.
- Năm fold giúp đánh giá trên nhiều phần held-out, giảm phụ thuộc một lần chia;
  đổi lại phải train nhiều lần nếu thực hiện đầy đủ cross-validation.
- Code dùng lại split đã lưu, không chia ngẫu nhiên lại khi đọc batch.
- **V3 hiện đã fit cho fold 0**, không phải đã fit/train đủ năm fold.

`group_id` gom các volume phải cùng partition. Hiện dùng volume ID làm fallback,
nên đã bảo đảm tách biệt theo **volume/group** nhưng **chưa chứng minh patient-level
independence**. Thiếu mapping để biết hai volume khác ID có cùng bệnh nhân hay không.

## 4. Chốt lại hình học: giữ native, không làm mất tổn thương

### 4.1. Bỏ resize khỏi đường MERIT chính

Ban đầu đã trao đổi về resize512→128. Rủi ro là tổn thương nhỏ có thể biến mất,
kéo theo thay đổi mask và nhãn phụ presence/salience.

Quyết định cho đường chính hiện tại:

- Giữ **native512×512**, `size_hw: null`.
- Không crop, không nội suy hoặc resample Z.
- Không resize GT, làm mịn hoặc morphology để “sửa” nhãn.
- Nếu lưới nguồn không khớp lưới model, báo lỗi thay vì tự thu nhỏ ảnh.

Các config resize cũ được giữ khi còn phục vụ nghiên cứu/tái lập; sự tồn tại của
chúng không có nghĩa pipeline V3 đang resize.

### 4.2. Thống nhất cách sắp xếp CT và GT về LPS

Pipeline đưa CT/GT về cùng LPS bằng **hoán vị/lật trục không nội suy**, đồng thời
cập nhật affine và giữ mapping để đảo về ảnh gốc. Đây là sắp xếp mảng trong RAM,
không ghi lại file NIfTI và không đổi vị trí vật lý bằng resampling.

QC199 ghi nhận nguồn có **193 LPS và6 LPI**, nên nhận định ban đầu “tất cả đã LPS”
không đúng cho toàn bộ dataset. Riêng TRAIN fold0 có125 LPS và2 LPI. Ca đã LPS
không cần đổi; ca LPI được lật trục tương ứng cho cả CT và GT.

NIfTI thường được mô tả bằng XYZ; tensor volume dùng ZYX. Đổi thứ tự chiều có thể
làm cách ghi shape thay đổi mà không thay số voxel. World coordinates của NiBabel
theo RAS+ là quy ước khác với hướng các trục mảng LPS.

### 4.3. Spacing và đơn vị

Spacing là khoảng cách giữa tâm voxel kề nhau theo từng trục, khác với shape là số
voxel. Ba ca A-0161, A-0585, A-0599 có đơn vị không gian `unknown` trong header.

Đã giữ nguyên header. Config chọn rõ `assume_mm: true` để cho phép giả định mm;
điều này **không xác minh đơn vị thật**. Sampling/position/weighting vẫn là `index`;
giải quyết chiến lược spacing được deferred/out-of-scope.

## 5. Cường độ CT: các quyết định đã thay đổi theo thời gian

Đây là phần cần phân biệt rõ nhất để không lấy nhầm cấu hình cũ.

| Giai đoạn/phương án | Đã làm gì? | Trạng thái và ý nghĩa |
| --- | --- | --- |
| Native | Giữ giá trị NIfTI-scaled, bỏ window/normalization tự chọn khỏi đường chính lúc đó | Được giữ để tái lập run cũ |
| Global TRAIN Z-score | Fit một mean/std trên tất cả voxel127 TRAIN, gồm nền và−2048 | Đã fit/kiểm CPU; là phương án lịch sử, chưa chứng minh tốt hơn Native |
| Audit outlier và sentinel | Đọc CT thật để hiểu tails và vùng−2048 trước khi chốt policy mới | Đã hoàn tất, là bằng chứng đầu vào cho quyết định V3 |
| V3 | Fit clipping + mean/std trên miền TRAIN được chọn bảo thủ; áp một transform chung cho mọi CT | Đã được người dùng cho phép, tích hợp, fit và QC199 hoàn tất |

Native vẫn áp dụng cách đọc slope/intercept theo NIfTI, không có nghĩa bỏ qua cách
giải mã file. **Chưa có bằng chứng độc lập để khẳng định toàn bộ các giá trị là HU**;
đơn vị không gian mm không chứng minh cường độ là HU.

V3 là thay đổi policy cường độ được chốt sau các bước audit, không phải âm thầm bật
lại một normalization đã bị yêu cầu bỏ. Native/global Z-score cũ vẫn được giữ.

## 6. Những audit dữ liệu thật đã thực hiện

### 6.1. Audit127 TRAIN

Đã đọc127 CT và127 GT, kiểm finite values, binary mask, shape, affine và orientation;
so SHA trước/sau để chứng minh không sửa dữ liệu.

Các phát hiện đáng chú ý:

- Có **2.309.488.640 voxel CT TRAIN**.
- Foreground GT chiếm khoảng **0,4453%** số voxel, cho thấy dữ liệu mất cân bằng rõ.
- Min đo được−9653, max32737; có tails cực trị nhưng chúng chiếm tỷ lệ nhỏ.
- Có **473.812.909 voxel đúng−2048**, khoảng20,52% TRAIN.
- Dtype không hoàn toàn giống nhau: phần lớn CT int16, A-0599 float64; khác dtype
  tự nó không phải lỗi nếu giá trị/nhãn và geometry hợp lệ.

Những phát hiện này không chứng minh HU sai hoặc xác định duy nhất nguyên nhân
model suy giảm. Không chọn HU-window chỉ bằng cách nhìn min/max.

### 6.2. Audit sentinel/FOV

Đã xác nhận120/127 TRAIN có−2048, phần lớn có cấu trúc ổn định ở vùng ngoài ảnh.
Khoảng99,8096% số voxel−2048 nằm ngoài vòng tròn tham chiếu của audit.

Nhưng có năm pattern không tròn và bảy ca không có−2048; không thể dùng một vòng
tròn cố định hoặc tự coi−1024 là padding cho mọi CT. Vòng tròn chỉ dùng phân tích,
**không trở thành mask crop của production**.

### 6.3. Thử Native và global Z-score trước V3

Đã fit global Z-score trên127 TRAIN, mean≈−883,4049 và std≈735,3071; kiểm transform
và GT trên16 ca thật. Đây là bộ số **cũ**, khác stats V3 hiện tại.

Pilot Native gặp OOM trên GPU4GB trước optimizer update đầu tiên; nhánh B dừng theo
quy trình. Vì vậy **chưa hoàn thành phép so sánh chất lượng Native/global Z-score**.
Các notebook full A/B/QUICK100 được chuẩn bị trước đây không tự là kết quả huấn luyện.

## 7. V3 xử lý padding như thế nào?

Cần phân biệt hai loại mask:

| Loại mask | Vai trò |
| --- | --- |
| GT mask | Nhãn tổn thương0/1 của dataset; phải bảo toàn |
| Mask miền FIT | Mask tạm suy từ CT để chọn voxel tính stats; không phải nhãn tổn thương, không đưa vào model |

Rule V3 chỉ dùng CT, không dùng GT: tìm−2048 **4-connected ra biên**, có neo ở góc,
ổn định từng bit qua ít nhất3 lát, diện tích và độ sâu nằm trong giới hạn đã khóa.
Không ép hình tròn, không loại−2048 rời rạc bên trong.

- `CONFIDENT`: chỉ loại vùng được rule nhận diện khỏi **phép FIT thống kê**.
- `UNKNOWN/AMBIGUOUS`: dùng toàn bộ voxel hữu hạn của ca, ghi rõ fallback.
- Nếu fallback quá20% số ca, trả `BLOCKED` thay vì xuất stats dùng được.

Kết quả thật: **120 CONFIDENT,7 UNKNOWN,0 AMBIGUOUS**. Bảy UNKNOWN vì không có
−2048, không đồng nghĩa bảy CT hỏng. Đã loại473.812.900 voxel khỏi miền FIT và giữ
lại9 voxel−2048 rời rạc bên trong.

**Không voxel nào bị xóa/crop/zero-fill khỏi ảnh CT hoặc GT.** CONFIDENT là mức
tin cậy theo heuristic, chưa phải xác nhận padding bằng DICOM.

## 8. Đã fit và khóa một bộ stats V3 chung

Chỉ dùng đúng127 TRAIN của fold0, không dùng VAL/TEST/GT để chọn cường độ hoặc tính stats.

1. Tính percentile0.1 và99.9 trên population voxel được chấp nhận để có L/U.
2. Clip chính population đó vào L/U, rồi tính lại mean và population std.
3. Lưu stats cùng IDs, split/manifest hashes, source hashes và policy để kiểm provenance.

| Tham số hiện tại | Giá trị |
| --- | ---: |
| L | -1241.0 |
| U | 1032.0 |
| Mean | -583.0622212292245 |
| Std | 488.0394718302036 |
| Voxel dùng FIT | 1.835.675.740 |

Fit chạy CPU hai lượt, theo từng CT/block, không concatenate toàn bộ cohort vào RAM.
Phân vị dùng histogram deterministic, lưới1/16 đơn vị scaled, sai số tối đa1/32;
mean/std tính trên voxel thật sau clipping, không lấy moment từ histogram lượng tử.
Thời gian fit đã đo khoảng150 giây trên máy hiện tại.

Stats thật lưu ở [`analysis/preprocessing/fit_stats.json`](../analysis/preprocessing/fit_stats.json).
Không copy bộ mean/std global Z-score cũ vào V3 và không fit lại chỉ để chuyển máy.

## 9. Pipeline hiện tại đưa gì vào model?

```text
Manifest + split đã lưu
  → chọn case đúng partition
  → kiểm paired headers/shape/affine/orientation/spacing
  → CT và GT cùng LPS, không nội suy
  → kiểm CT hữu hạn và GT exact0/1 trước cast
  → chỉ CT nhận V3: (clip(x,L,U)-mean)/std
  → giữ GT nguyên nhãn và mapping
  → XYZ thành ZYX trong RAM
  → lấy master +11 lát, giữ512×512
  → DataLoader tạo batch CPU
  → người train chuyển batch cho mô hình
```

Mọi voxel CT, **kể cả padding**, đều dùng cùng bốn tham số. −2048 trở thành khoảng
−1,348124 sau transform, không thành0. Không tìm FOV hoặc mean/std riêng khi đọc
VAL/TEST/inference. Inference CT-only không cần mở GT.

Batch thực đã đọc bằng CLI có shape **`[1,1,11,512,512]`**, tương ứng
`[batch,channel,số lát,height,width]`, dtype float32, device CPU.

## 10. Dữ liệu sau xử lý được lưu ở đâu?

**Không có thư mục mới chứa199 CT NIfTI đã chuẩn hóa.**

- Trên ổ đĩa: giữ CT/GT gốc, manifest, split, stats và báo cáo.
- Khi lấy batch: reader đọc volume, áp V3 trong RAM; Dataset cache một volume rồi
  lấy các master/slab. Khi cần volume khác, cache được thay.
- Khi train trên GPU: tensor CPU mới được chuyển vào VRAM theo trainer.

Vì vậy, người train vẫn gọi preprocessing khi loader đọc dữ liệu, nhưng **không
fit lại stats**. “Đã chạy QC199” nghĩa là đã kiểm đủ đường xử lý từng ca, không phải
đã lưu thêm199 bản ảnh biến đổi.

## 11. Đã bảo toàn GT và kiểm đủ199 ca

Quá trình kiểm tăng dần: kiểm sâu6 ca đại diện trước, rồi chốt bằng **toàn199 cặp**
qua chính `read_arrays` production V3, CPU từng ca.

| Phần dữ liệu fold0 | Đã QC | Kết quả |
| --- | ---: | --- |
| TRAIN | 127 | PASS |
| VAL | 32 | PASS |
| TEST | 40 | PASS |
| Tổng | 199 | 199 PASS,0 FAIL |

Mỗi ca đã kiểm:

- ID/filename CT–GT, header/shape/affine/orientation/spacing theo logic hiện hữu.
- GT scaled có đúng nhãn0/1 **trước cast float32**, tránh làm tròn nhãn sai thành hợp lệ.
- CT sau V3 hữu hạn; CT/GT đồng bộ LPS, native512 và không resample.
- Lesion count trước, sau LPS và sau inverse bằng nhau.
- **0 voxel lệch tọa độ** khi đảo GT đã xử lý về lưới GT gốc.
- Hash CT/GT trước và sau bằng nhau;398 file khớp cache đã xác minh.

QC toàn199 mất khoảng **516,87 giây**. Đọc VAL/TEST ở bước này chỉ để kiểm tính đúng
của dữ liệu/pipeline với stats đã cố định; không fit, chọn ngưỡng hoặc đánh giá model.

Các tests CPU chọn lọc cho lần tích hợp V3 đạt76; bản source riêng ở lần chốt đạt31
tests và build wheel offline. Các nhóm test có trùng nhau, không cộng thành107 test
độc lập. Lượt regression đầu từng chạy nhầm2 test metric trên mẫu giả; đã ghi rõ
trong [TEST_REPORT](../analysis/preprocessing/TEST_REPORT.md), không phải Dice V3 trên CT thật.

## 12. Các file chính và bộ bàn giao cuối

| File/thư mục | Dùng để làm gì? |
| --- | --- |
| `data/covid19_20/Train/` | 398 CT/GT NIfTI gốc |
| `data/covid19_20/cases.json` | Manifest để tìm đúng cặp ảnh |
| `splits/covid19_20_5fold.json` | Chia ID cố định, không di chuyển dữ liệu |
| `configs/merit_preprocessing.yaml` | Config V3 cho training mới, fold0 |
| `analysis/preprocessing/fit_stats.json` | Bộ stats đã fit và provenance |
| `analysis/preprocessing/case_padding_audit.csv` | Phân loại padding/fallback của127 TRAIN |
| `src/merit_covid/data/pipeline/preprocessing.py` | Reader và transform chung |
| `src/merit_covid/data/pipeline/intensity_fit.py` | Rule miền FIT, thống kê và kiểm provenance |
| `dataset.py`, `slab.py`, `dataloader.py` cùng thư mục | Cache volume, tạo slab và batch |
| `scripts/fit_train_zscore.py` | CLI fit hiện có, được mở rộng cho V3 |
| `scripts/read_training_batch.py` | Đọc thử một batch CPU từ config chuẩn |
| `scripts/qc_preprocessing.py` | QC read-only từng cặp qua API production |

Đã tích hợp vào đường code chung, không tạo Dataset/loader riêng cho V3. Đã bỏ
hai private alias không có caller và một nhánh unreachable; không xóa nguyên file
còn phục vụ API, test hoặc run lịch sử.

Bộ đầy đủ mới nhất là
[`PREPROCESSING_V3_TEAM_HANDOFF_FINAL.zip`](../outputs/preprocessing_team_handoff_final_20261121/PREPROCESSING_V3_TEAM_HANDOFF_FINAL.zip).
Nó có source MERIT cần thiết trong `project/`, dependencies, config/stats, manifest/
split metadata, QC199, docs và hash checker; **không còn chỉ là overlay cũ**.
Không kèm NIfTI, mask hay checkpoint.

Repo gốc `torusama/merit-covid-ct`; commit nền
`13d7695d494b9da40629afd6a45ac294285e2b08`. Snapshot được lấy từ working tree
`MADGNet` đang dirty, chưa commit/push. Hash và khác biệt từng file được ghi trong
bundle; commit nền một mình chưa bao gồm các thay đổi V3 chưa commit.

Đọc [HANDOFF_FOR_TEAMMATE](../analysis/preprocessing_team_handoff_20261121/HANDOFF_FOR_TEAMMATE.md)
để dựng source, trỏ cache/manifest/split, verify và đọc batch. Các folder tạm dùng
kiểm bundle đã dọn, không xóa dữ liệu hoặc kết quả cũ.

## 13. Người nhận cần làm gì tiếp?

1. Nhận **hai phần riêng**: dataset gốc `data/covid19_20` và ZIP source FINAL.
   Upload dataset không đồng nghĩa đã upload chương trình; ZIP code không chứa ảnh.
2. Giải nén source vào thư mục mới, kiểm hash và cài dependencies theo handoff.
3. Trỏ manifest đến cache CT/GT hiện có; giữ bytes/IDs của manifest/split/stats,
   không tạo lại split hoặc fit thông số mới cho thuận đường dẫn.
4. Chạy kiểm một batch CPU bằng config chuẩn:

```bash
python scripts/read_training_batch.py --config configs/merit_preprocessing.yaml --fold 0 --batch-size 1 --workers 0
```

Sau các bước trên, người train chuẩn bị GPU/môi trường rồi tạo **run V3 mới**.
**Không resume checkpoint Native như thể nó đã train bằng V3.** Các run cũ phải giữ
nguyên preprocessing đã ghi trong checkpoint của chúng.

## 14. Những giới hạn vẫn còn

| Vấn đề | Trạng thái |
| --- | --- |
| Preprocessing/code/stats/QC199 | Đã hoàn thành và sẵn sàng bàn giao |
| Hiệu quả segmentation của V3 | Chưa đo Dice/IoU trên CT thật, chưa chứng minh cải thiện |
| GPU/model training V3 | Chưa kiểm chứng bằng một run huấn luyện thật |
| Patient independence | Chưa xác minh; hiện tách theo volume/group |
| Đơn vị không gian | 3 ca header unknown, chỉ có giả định mm được khai báo |
| Padding | 120 CONFIDENT theo heuristic;7 UNKNOWN fallback, không phải DICOM-verified FOV |
| HU | Chưa xác minh độc lập cho toàn bộ dataset |
| Clipping | Làm mất phân biệt các giá trị ngoài L/U; chưa chứng minh tác động tốt/xấu lên Dice |

“Unknown unit”, “UNKNOWN padding” và “patient identity unverified” là **ba vấn đề
khác nhau**, không phải cùng một danh sách ca lỗi.

## 15. Các báo cáo dùng để đối chiếu

- [Audit voxel127 TRAIN](../outputs/data_qc_step1_20261120/DATA_QC_STEP1.md).
- [Audit sentinel/FOV](../outputs/sentinel_fov_fold0_20261120/SENTINEL_FOV_FINDINGS.md).
- [Native/global Z-score trước V3](../outputs/step2b_native_zscore_20261120/STEP_2B_REPORT.md).
- [Thay đổi code V3](../analysis/preprocessing/CHANGE_SUMMARY.md).
- [QC199 từng ca](../analysis/preprocessing_team_handoff_20261121/qc/all199_ct_gt_qc.csv)
  và [summary](../analysis/preprocessing_team_handoff_20261121/qc/all199_ct_gt_qc_summary.json).
- [Kiểm chứng bàn giao](../analysis/preprocessing_team_handoff_20261121/TEAM_HANDOFF_VERIFICATION.json)
  và [integrity ZIP](../outputs/preprocessing_team_handoff_final_20261121/BUNDLE_RECEIPT.json).

Tài liệu tổng hợp này chỉ bổ sung cách đọc lịch sử và trạng thái hiện tại; không
thay stats, pipeline, dữ liệu hoặc bundle FINAL đã đóng băng.
