# Đánh giá MERIT-UNETR++ cho phân đoạn COVID CT

Ngày 11/11/2026. Phạm vi: mã nguồn và cấu hình hiện tại ở commit `2638a79f3f63b9ad85391ae3b6e107ed3edb2aa8`, các báo cáo dữ liệu/preflight đã có, và kiểm tra split. Hai notebook đang có thay đổi cục bộ; báo cáo này không đánh giá chi tiết những thay đổi đó. Đây là đánh giá nghiên cứu và độ sẵn sàng của pipeline, không phải kết quả hiệu năng mô hình hoặc xác nhận lâm sàng.

## Kết luận

**Chưa nên chạy full 5-fold để lấy kết quả chính thức.** Mã có một baseline MERIT hợp lý về mặt cấu trúc: giữ trục lát cắt, có TEP/MER/PCG, train trên nhãn slab thực và suy luận toàn volume không dùng GT để chọn ROI. Tuy nhiên, chưa có bằng chứng chứng minh ba module này có ích cho tổn thương COVID, cũng chưa thể kết luận code tái hiện chính xác tác giả gốc. Các quyết định về intensity, nhóm bệnh nhân, mất tổn thương khi resize và khoảng cách lát cắt vẫn cần giải quyết hoặc công bố giới hạn trước khi khóa protocol.

Paper tham chiếu trong [tài liệu mô hình](merit_unetrpp.md) nghiên cứu nodule trên LIDC-IDRI với ROI dựa vào annotation. Bài toán ở đây là mask nhiễm COVID trên toàn CT: tổn thương có thể lan tỏa, nhiều ổ, nhỏ và không đều; tỷ lệ foreground trên toàn ảnh rất thấp. Vì vậy dùng kiến trúc paper làm baseline là có lý, nhưng các module và hyperparameter cần được kiểm nghiệm lại trên bài toán mới. Không có kết quả ablation ở đây để khẳng định module nào tốt hay hại.

## Những vấn đề cần xử lý trước full training

### P0. Intensity chưa được xác minh và baseline đang dùng giá trị thô

[`merit_index.yaml`](../configs/merit_index.yaml) đặt `intensity_encoding: native`. [`normalize_ct`](../src/merit_covid/data/pipeline/preprocessing.py) giữ nguyên giá trị sau NIfTI scaling; không clipping hay chuẩn hóa. Điều này không tự xác nhận HU, và giá trị ngoại lai có thể ảnh hưởng học. Không thể chỉ từ min/max kết luận mỗi voxel có sai HU hay không.

[Preflight cũ](../outputs/merit_preflight_fold0/REPORT.md) đo đầy đủ **127 ca train fold 0**, không phải 199 ca: khoảng -9653 đến 32737, 39 ca có min dưới -2048 hoặc max trên 10000, slope/intercept 1/0 cho nhóm đó. [Báo cáo 199 volume](../analysis/covid19_20/REPORT.md) ghi khoảng -10380 đến 32737, nhưng tái sử dụng audit CT cũ và không xác nhận byte nguồn hoặc calibration từng ca. Hai khoảng không mâu thuẫn vì khác phạm vi. Nguồn challenge mô tả HU, nhưng lịch sử DICOM-to-NIfTI từng ca chưa được chứng minh.

**Việc cần làm:** kiểm tra nguồn và slope/intercept, datatype, histogram và ca ngoại lai cho toàn bộ cohort; phân biệt giá trị padding/ngoại lai với mô bệnh học. Chọn và đóng băng xử lý intensity bằng train/validation, fit riêng mỗi fold nếu dùng thống kê dữ liệu, áp dụng giống nhau khi inference. Không dùng Zenodo20 để chọn chính sách. Preflight đề xuất z-score train-fold nhưng code hiện chưa triển khai policy fitted đó; không nhầm đề xuất với tính năng đã chạy.

### P0. Spatial unit chưa chắc chắn và split chưa xác nhận bệnh nhân

[`geometry.py`](../src/merit_covid/data/pipeline/geometry.py) mặc định từ chối unit không rõ. Tuy nhiên **baseline index hiện bật `assume_mm: true`**, khác checklist đã gửi; [`merit_physical.yaml`](../configs/merit_physical.yaml) đặt `false`. Ba ca `volume-covid19-A-0161`, `A-0585`, `A-0599` thiếu unit theo báo cáo cohort. Khi bật giả định, khoảng cách và HD95 tính theo mm của các ca đó phụ thuộc giả định, không phải bằng chứng nguồn. Cần xác minh từ metadata nguồn hoặc ghi rõ giả định/xử lý có provenance trước khi báo cáo số mm.

Chạy `python scripts/merit_pipeline.py check-split --config configs/merit_index.yaml` thành công: fold 0–3 có 127/32/40 train/val/test, fold 4 có 128/32/39. Preflight cũ xác nhận không trùng case/group/path trong 15 cặp partition. Nhưng [`cases.py`](../src/merit_covid/data/pipeline/cases.py) dùng `volume_id_fallback`, và split ghi `patient_ids_available: false`: **không chứng minh được độc lập theo bệnh nhân**, cũng không loại trừ dữ liệu giống nhau dưới tên khác. Cần nguồn patient ID đáng tin cậy; nếu không có, chỉ gọi volume-level split và nêu giới hạn. Không tự ý thay split sau khi có kết quả.

### P0. 128×128 làm mất thông tin nhãn

[`resize_in_plane`](../src/merit_covid/data/pipeline/preprocessing.py) dùng bilinear cho CT, nearest-exact cho mask và không resize Z. Trên **127 ca train fold 0**, [preflight](../outputs/merit_preflight_fold0/REPORT.md) đo:

| Chỉ số | 128×128 | 256×256 |
| --- | ---: | ---: |
| Thành phần 2D biến mất hoàn toàn | 325 | 186 |
| Foreground native không được khôi phục sau down/up | 4,4185% | 2,3822% |
| Positive slice trở thành empty | 34 | 10 |

Đây là QC hình học, không phải Dice của model hay số lesion 3D. Có 6.295 thành phần 2D trong mẫu đo. 256 giữ nhãn tốt hơn theo tổng nhưng thử full optimizer trên GTX 1650 4 GB đã OOM, kể cả batch 1. Không tự nâng lên 256. Đánh giá thêm theo kích thước/vị trí tổn thương bằng train, cân nhắc crop/tiling dựa trên ảnh CT mà không dùng GT khi inference, và profile lại bộ nhớ. Hiện patch engine baseline chưa tích hợp MERIT.

### P0. Xác nhận hình học và phục hồi trên dữ liệu thật

[`inspect_pair`](../src/merit_covid/data/pipeline/geometry.py) kiểm tra shape, affine và orientation trước đọc voxel; reorientation chỉ đổi trục/lật. Mask được kiểm tra 0/1 và resize nearest-exact. [`predict_slab_case`](../src/merit_covid/engine/slab_inference.py) tạo dataset CT-only, dự đoán từng master, nội suy probability về XY native rồi ghép lát; evaluator đọc GT sau dự đoán để tính metric native. Đây là thiết kế đúng hướng, không phải bằng chứng tất cả 199 ca đã qua end-to-end.

Pilot cũ dùng một ca train A-0025, bốn optimizer steps, kiểm tra inference không mở mask, khôi phục shape/affine gốc 512×512×61; chưa chứng minh convergence hoặc độ chính xác. Cần kiểm thêm alignment, nhãn và restoration ở các orientation/spacing/biên khác nhau, đối chiếu native GT, xác nhận không thiếu/trùng slice và nêu unit giả định. Không dùng GT để chọn crop hoặc master inference.

## Khoảng cách lát cắt: mã làm gì và có nghĩa gì?

Theo [thống kê cohort](../analysis/covid19_20/REPORT.md), 188/199 volume ghi spacing 5 mm, còn lại 0,799988/1/2 mm; ba unit chưa rõ phải coi là giả định. Số liệu là spacing header, không chứng minh slice thickness hoặc khoảng cách lát acquisition bất quy tắc. NIfTI affine chỉ mô tả lưới đều và không thể tự khôi phục vị trí từng lát gốc nếu conversion đã làm mất thông tin.

Baseline lấy 11 chỉ số liên tiếp quanh master: nếu spacing 5 mm thì tâm context danh nghĩa trải từ -25 đến +25 mm; nếu 1 mm thì chỉ -5 đến +5 mm. Hai volume vì vậy có receptive context vật lý khác, ngoài khác biệt về cấu trúc và intensity. Gần mép volume, `edge` lặp lát biên.

[`slab.py`](../src/merit_covid/data/pipeline/slab.py) có lựa chọn `physical_mm`: lấy lát native gần các offset mục tiêu, **không nội suy theo Z**, lưu cả requested offset và offset thực. Offset thực lấy từ displacement tâm lát theo cột thứ ba affine; với shear đây là khoảng cách giữa tâm tương ứng dọc cột affine, không nhất thiết là khoảng cách vuông góc giữa mặt phẳng. Không thể dựa vào nó để kết luận đã chuẩn hóa slice thickness. Target nhỏ hơn native spacing thường chọn trùng lát; target lớn hơn có thể bỏ qua lát, còn ranh giới vẫn lặp.

Ba lựa chọn độc lập cần ablation trên cùng fold, seed, preprocessing, kiến trúc và các đơn vị đã xử lý thống nhất:

| Thí nghiệm | Sampling | Position encoding | Trọng số loss |
| --- | --- | --- | --- |
| E1 | index | index | index |
| E2 | physical mm | index | index |
| E3 | physical mm | actual mm | index |
| E4 | physical mm | actual mm | actual mm |

E2 template cố ý để `target_spacing_mm: null`, phải chọn tham số từ train/validation. Khi so E1–E4 phải thống nhất policy unit: hai YAML hiện khác `assume_mm`, nếu giữ nguyên thì không còn là so sánh một yếu tố. Báo cáo native slices khác nhau, actual offsets, độ lặp theo volume và biên, cũng như Dice/recall theo spacing. `position_unit_mm` và `sigma_mm` cần được định nghĩa và cố định. Không mặc định 5 mm hay physical PE tốt nhất.

## Đánh giá các module

[`model.py`](../src/merit_covid/models/merit_unetrpp/model.py) đặt TEP ở bottleneck, MER sau decoder3 và PCG trước output; [tài liệu](merit_unetrpp.md) ghi công thức, kích thước, số tham số và các lựa chọn paper chưa đặc tả. Đây là reimplementation dựa trên paper chứ không phải source tác giả; khớp số tham số hoặc test công thức không chứng minh tái hiện kết quả.

- **TEP:** gộp thông tin các lát thành prototype theo salience và vị trí rồi attention trở lại từng lát. Với COVID đa ổ/lan tỏa, giả thuyết dùng context hợp lý nhưng một prototype chung có thể không phù hợp mọi tổn thương; đó là giả thuyết cần đo, không phải lỗi đã chứng minh. So UNETR++ baseline với TEP, theo lesion size và vị trí, kiểm tra salience có tương quan lesion hay anatomy/intensity.
- **MER:** master query attention tới context, cập nhật master trong module; các lát khác không được cập nhật bởi chính MER. Context xa có thể hữu ích hoặc nhiễu. So TEP-only với TEP+MER, kiểm tra 3/5/11 lát và phạm vi mm; đảm bảo cấu hình model và input khớp khi thay độ sâu.
- **PCG:** dự đoán presence và cộng log-odds vào foreground logits, warmup theo epoch. Có thể giảm false positive nhưng cũng có thể chặn tổn thương nhỏ; đo sensitivity, false positives trên slice rỗng, calibration ECE/Brier và so có/không PCG. `alignment_pool=max` nhạy với một pixel; `top_k_mean` đã là lựa chọn ablation trong loss, chưa có bằng chứng chọn phương án nào.
- **Duplicate context:** `duplicate_slice_policy: unique` chỉ loại bản trùng khỏi **loss** qua `valid_region`; [`modules.py`](../src/merit_covid/models/merit_unetrpp/modules.py) không nhận mask validity, vì vậy TEP/MER vẫn attention vào lát lặp. Ví dụ master đầu volume với edge padding có nhiều bản sao master trong context; kết quả attention có thể bị trọng số lệch. Đây là hành vi xác nhận từ code, chưa chứng minh gây giảm Dice. Thử tùy chọn mask và so baseline, bảo đảm softmax an toàn nếu không còn context hợp lệ; lưu ý mask attention không xóa ảnh hưởng duplicate qua các convolution 3D trước attention.

Master hiện luôn ở giữa slab trong data pipeline, dù model API nhận vị trí khác. Central master là protocol hợp lệ; shifted window là thí nghiệm riêng, không phải lỗi bắt buộc sửa ngay.

## Training và đánh giá

- [`CTSlabDataset`](../src/merit_covid/data/pipeline/dataset.py) tạo một master cho mỗi lát. Volume dài đóng góp nhiều mẫu hơn. [`CaseGroupedSampler`](../scripts/train_colab.py) chỉ xáo thứ tự case rồi lát và vẫn thăm mỗi master một lần; **không phải volume-balanced sampling**. Thử sampler cân bằng volume và positive/negative như ablation, tránh dùng nhãn để lọc inference.
- Đường train MERIT chính chưa thấy augmentation trong `CTSlabDataset`; augmentation ở `data/dataset.py` là dataset khác, không đồng nghĩa được dùng ở đây. Nếu bổ sung, transform hình học phải đồng bộ toàn slab/mask, chỉ trên train và có seed/provenance.
- [`metrics.py`](../src/merit_covid/data/pipeline/metrics.py) đã có Dice/IoU, precision/recall, HD95 và policy mask rỗng; aggregate theo group. Cần bổ sung hoặc báo cáo rõ lesion-size recall, false positives trên lát rỗng và ECE/Brier cho presence. HD95 undefined khi một trong hai mask rỗng và mm phụ thuộc unit đúng.
- Chốt protocol: U-Net 2D và UNETR++ cùng split/preprocessing/evaluation; baseline MERIT; ablation từng module và spacing; 5-fold báo cáo per-case/per-fold, mean ± SD; chỉ dùng Zenodo20 sau khi khóa preprocessing, model và threshold. Không so Dice trực tiếp với paper khác protocol.

## Kiểm thử và giới hạn bằng chứng

Kiểm tra split hiện tại thành công với hash `7b31f9e250e64826f1fff0d9fec4a4aba034252cec7d5123d5f0393ec665e54d`. Preflight lịch sử ngày 08/11 báo 341 tests pass, pilot CT và profile GPU như trên; đó là kết quả lịch sử, không đại diện tự động cho mọi thay đổi notebook hiện tại. Không có full training, 5-fold hay external evaluation được thực hiện trong lần review này.

Lần review này chạy `python -m pytest tests/test_merit_unetrpp.py tests/test_merit_pipeline.py tests/test_preprocessing.py tests/test_reconstruction.py tests/test_splitting.py -q`: **77 passed trong 77,84 giây** trên môi trường hiện tại. Đây là các test chọn lọc, không phải kiểm thử toàn bộ repo hay xác nhận hiệu năng COVID.

Index GitNexus hiện báo chậm hơn HEAD hai commit; lần refresh bị gián đoạn, nên các kết luận ở đây dựa trên đọc source trực tiếp và artifact đã lưu, không khẳng định graph review đầy đủ. Audit cũ không chứng minh dữ liệu hiện tại chưa đổi; cần pin phiên bản/hash dữ liệu trước nghiên cứu chính thức.

## Thứ tự thực hiện đề xuất

1. Xác minh nguồn intensity/units, patient identity; lưu quyết định và provenance, không thay split tùy tiện.
2. Đo retention và lựa chọn input khả thi trên train/val; kiểm hình học và CT-only restoration ở nhiều ca/biên.
3. Thêm và test tùy chọn duplicate attention mask, augmentation đồng bộ, sampler và metric thiếu khi cần; giữ baseline cũ cho đối chứng.
4. Chạy pilot dài hơn và xác nhận finite loss, resume, inference, VRAM trên cấu hình thực; khóa config/seed/fold và chọn checkpoint bằng validation native-volume.
5. Chạy baseline và ablation trước khi kết luận nên thay TEP/MER/PCG hoặc physical spacing; giữ external test độc lập.
