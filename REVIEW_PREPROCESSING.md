# Review tiền xử lý V3, B2 và B3A — 24/11/2026

**Đây là cập nhật phần dữ liệu trên nhánh `train/merit-unetrpp-colab`, chưa phải
bản tích hợp train Tiling128.** Điểm đo gốc dùng workspace nền
`13d7695d494b9da40629afd6a45ac294285e2b08`. Bản publish áp phần dữ liệu lên
train nền `eacbfdc`, giữ nguyên hỗ trợ frozen BatchNorm và các tính năng đã có
trên train. So diff với `eacbfdc` để xác định phần mới; không gộp lịch sử model
trước commit nền thành thay đổi preprocessing của lượt này. Không merge vào main.

## 1. Phạm vi thay đổi

- **V3:** fit thống kê trên127 TRAIN fold0 và áp clipped Z-score qua reader chung,
  giữ grid native512, GT và split. Có guard ràng buộc stats/source/fold trong
  `engine/training.py`; không đổi công thức loss, optimizer hoặc training loop.
- **B2:** CLI QC độc lập chia/ghép16 tile128 mỗi lát, kiểm đủ199 ca V3 và slab11.
- **B3A:** hai Dataset **prototype trong tools/**, exporter subset13 TRAIN,
  benchmark CPU và notebook. Chưa thay production Dataset thành tile Dataset.
- Kiến trúc MERIT/UNETR++, loss và Native512 baseline giữ nguyên so với train nền eacbfdc.
- Không đưa các module/sửa lỗi MADGNet, nghiên cứu PHNet/MedNeXt hoặc công việc
  notebook cũ chưa liên quan từ workspace vào nhánh review này.

## 2. GPT web nên đọc theo thứ tự

1. [Tổng kết preprocessing](docs/TONG_KET_TIEN_XU_LY_DU_LIEU.md) và
   [hướng dẫn V3](docs/PREPROCESSING_HANDOFF.md).
2. [Báo cáo B2](analysis/tiling128_b2_20261124/B2_REPORT.md),
   [summary JSON](analysis/tiling128_b2_20261124/SUMMARY.json),
   [QC CSV199 ca](analysis/tiling128_b2_20261124/tiling128_qc_199.csv).
3. [B3A README](analysis/b3a_delivery_20261124/B3A_README.md),
   [protocol](analysis/b3a_delivery_20261124/B3A_PROTOCOL.md),
   [decision](analysis/b3a_delivery_20261124/b3a_decision.md).
4. [Benchmark JSON](analysis/b3a_delivery_20261124/b3a_benchmark.json),
   [24 raw repetitions CSV](analysis/b3a_delivery_20261124/b3a_benchmark.csv),
   [correctness thật](analysis/b3a_delivery_20261124/b3a_correctness.json),
   [audit cuối](analysis/b3a_delivery_20261124/FINAL_VERIFICATION.json).
5. Xem code: [V3 fitter](src/merit_covid/data/pipeline/intensity_fit.py),
   [reader](src/merit_covid/data/pipeline/preprocessing.py),
   [B2](tools/qc_tiling128.py), [B3A Dataset/exporter](tools/b3a_delivery.py),
   [runner](tools/b3a_compare.py), [tests](tests/test_b3a_delivery.py).

## 3. Kết quả thực và giới hạn

- V3 QC:199/199 cặp đạt; fit chỉ dùng TRAIN, không dùng VAL/TEST.
- B2:199/199 đạt,219280 tile-master; CT/GT roundtrip mismatch0.
- B3A:19 test synthetic,4656 sample A-0003 thật khớp và24 lượt CPU benchmark
  trên13TRAIN,23520sample/pass. Source core trong nhánh khớp hash lần đo.
- Benchmark **local Windows**, `KAGGLE_BENCHMARK_NOT_RUN`. Không có GPU/VRAM,
  train hoặc kết quả Dice mới từ B3A. Mọi lựa chọn data delivery là provisional.
- Precomputed tile-major/w0 nhanh nhất local:38,998s so với dynamic71,092s ở
  cùng setup. Case-major/w0 thì dynamic nhanh hơn; không tự đổi sampler train.
- Giữ giới hạn provenance của dataset V3; không khẳng định199 ca raw-exact hoặc
  patient-level independence đã xác minh. Split vẫn theo original volume/group.
- **Dừng B3A**; B3B/model128/loss/inference stitch/B4/B5 chưa thực hiện.

## 4. Hai loại input không được nhầm

- [Config V3](configs/merit_preprocessing.yaml) dùng dữ liệu **chưa V3** và
  [fit_stats](analysis/preprocessing/fit_stats.json) đã khóa từ TRAIN fold0.
- Dataset **precomputed V3** phải dùng `native` pass-through với `size_hw:null`;
  xem [YAML thực](analysis/precomputed_v3_reference/merit_precomputed_v3.yaml)
  và [metadata export](analysis/precomputed_v3_reference/EXPORT_METADATA.json).
  Không clip/Z-score lần hai, không dùng size_hw128 để giả làm tiling.
- NPY tile lưu CTfloat32 và GTuint8 lossless; sample GT vẫn float32 giống dynamic.
  Tile affine/shape và full-volume geometry giữ riêng, source qform/sform được
  lưu trong manifest. Không coi tile là bệnh nhân mới.

## 5. Chạy lại và đọc các artifact lịch sử

[Notebook Kaggle CPU](notebooks/kaggle_b3a_delivery_cpu.ipynb) được dựng lại từ
**chính snapshot train sẽ publish** để không nhúng module nghiên cứu MADGNet
hoặc các sửa demo đang để dở chưa publish.
Nó giữ nguyên13 ID, source hashes và V3 stats của benchmark local. Chọn CPU,
Add Input dataset V3 đầy đủ, Run All; chưa có kết quả Kaggle từ lượt publish.

```text
python -m pytest tests/test_intensity_fit.py tests/test_qc_preprocessing.py tests/test_tiling128_qc.py tests/test_b3a_delivery.py -q
python tools/b3a_compare.py --data-root <PRECOMPUTED_V3_ROOT> --output-dir <NEW_DIR> --synthetic-xml <PASS_XML> --workers 0 2 --batch-size 2 --repetitions 3
```

Data/checkpoint/416 array tile không có trên GitHub; các manifest chỉ là metadata.
JSON và một số báo cáo gốc giữ nguyên branch/path của máy đã đo (`MADGNet`,
`outputs/...`); đây là provenance lịch sử, không phải lệnh cần tạo đường dẫn máy
cũ. Bản benchmark và QC cần đọc online đã được copy vào `analysis/` như các link
trên. Source file SHA là căn cứ đối chiếu nội dung, không coi snapshot dirty là
một commit mới. Không suy từ status của nhánh này rằng toàn bộ model đã train
hoặc toàn bộ suite model/GPU vừa chạy lại.

## 6. Các câu cần review trước B3B

1. A/B có thực sự giữ cùng pixels, mask11, validity, tile/world/full geometry?
2. Itinerary, grouping, cache1volume/worker và hai order có được so công bằng?
3. CSV/raw stats/RSS workers/I/O và break-even có đủ để khuyến nghị có điều kiện?
4. Nên giữ dynamic hay precomputed trên Kaggle và chọn order/worker nào?
5. B3B còn cần đổi gì ở adapter/model128/loss targets/stitch, trong khi giữ
   Native512 baseline và không âm thầm thay normalization hoặc sampling?

## 7. Ki?m tra khi xu?t b?n

Xem [b?o c?o ki?m th?](analysis/b3a_delivery_20261124/PUBLISH_TEST_REPORT.md):
83 PASS, 1 l?i resume Windows t?i hi?n c? tr?n train n?n; ch?a s?a trong l??t n?y.
Kh?ng coi nh?nh n?y l? to?n b? suite ?? PASS.
