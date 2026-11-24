# B3A — Đề xuất từ số đo thực tế

**Chỉ so sánh data delivery CPU; chưa B3B/model/train hoặc benchmark GPU.**

Kaggle: `KAGGLE_BENCHMARK_NOT_RUN`. Kết quả local không được gán thành kết quả Kaggle.

Pilot: 13 original TRAIN volumes, 23,520 tile-master mỗi lượt, đủ16 tile/master.
Batch=2, seed42, repetitions=3, pin_memory=False.

| Order | Workers | Dynamic median s (min–max) | Precomputed median s (min–max) | Throughput gain | Break-even pilot epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| case_master_tile | 0 | 70.561 (70.463–70.607) | 84.029 (83.566–84.354) | -16.03% | không hòa vốn |
| case_master_tile | 2 | 96.930 (96.546–97.162) | 66.996 (66.881–67.038) | +44.68% | 2 |
| case_tile_master | 0 | 71.092 (70.860–71.500) | 38.998 (38.870–39.047) | +82.30% | 2 |
| case_tile_master | 2 | 94.174 (93.684–94.433) | 61.858 (61.830–61.971) | +52.24% | 2 |

## Nhận định có điều kiện

Kết quả phụ thuộc setup/order. Chỉ cân nhắc precomputed-volume tại các setup vượt15% ổn định; giữ dynamic làm lựa chọn đơn giản nếu chưa chốt được order/worker phù hợp. Không âm thầm đổi sampler production chỉ để đạt số đo thuận lợi. Đây chỉ là **provisional recommendation**, phải đo Kaggle trước khi chốt.

Ứng viên nhanh nhất đã đo local là **precomputed-volume + case→tile→master,
0 worker**: 38,998s so với dynamic71,092s trong chính cùng setup, throughput
tăng82,30%; hòa vốn chi phí export sau khoảng2 lượt pilot đầy đủ. Ngược lại,
với case→master→tile/0worker, dynamic nhanh hơn. Vì vậy chưa chọn một chế độ
chung cho mọi order/worker và chưa đổi sampler training hoặc tích hợp B3B.

## Chi phí, RAM và độ phân tán

- Export một lần: 57.470s; 416 file; 1,926,811,648 bytes.
- Stage cả hai mode về cùng storage: 6.738s, 1,283,053,761 bytes; chi phí chung này không trộn vào steady-state/epoch.
- Break-even = ceil(export_seconds / (median_dynamic_epoch - median_precomputed_epoch)) khi mẫu số dương.
- Epoch ở đây là toàn bộ13-case pilot, không phải epoch127/199 CT. Không ngoại suy tuyến tính ra full199 vì cache/I/O/startup có thể thay đổi.
- Chưa đo upload/copy qua mạng, cleanup hoặc GPU. Cleanup không được tự xóa dữ liệu gốc; prototype giữ output để audit.

Một lượt hash-validation bổ sung được đo **sau benchmark** mất4,802s; đây không
phải thời gian startup ban đầu đã được đo. Nếu dùng số này để ước lượng thêm một
lần validation ngoài chi phí export, break-even của case→master→tile/2worker
tăng từ2 lên3 lượt pilot; hai setup case→tile→master vẫn khoảng2 lượt. Không
được coi các chi phí chưa đo là0 hoặc gán con số này thành epoch full127/199.

| Order | Workers | A stdev s | B stdev s | A peak RSS MiB | B peak RSS MiB | A/B first-batch median s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| case_master_tile | 0 | 0.074 | 0.396 | 2213.9 | 230.3 | 5.007 / 0.012 |
| case_master_tile | 2 | 0.311 | 0.082 | 4596.3 | 660.8 | 9.230 / 4.688 |
| case_tile_master | 0 | 0.324 | 0.091 | 2214.7 | 257.4 | 5.960 / 0.011 |
| case_tile_master | 2 | 0.380 | 0.075 | 4591.2 | 707.8 | 9.051 / 4.588 |

- RSS gồm main+descendants/workers, lấy mẫu20ms; shared pages có thể đếm lặp, không phải USS hoặc RAM vật lý duy nhất.
- OS cache không kiểm soát cold-cache. Đã warmup; mỗi lượt vẫn tạo iterator/worker và cache Dataset được reset.
- Process I/O là counter logical có cả IPC; declared file bytes là kích thước file tại cache miss, không phải số page mmap thực đọc từ ổ.
- Consumer ép materialize/sum tensor CPU của full DataLoader+collate. Đây chưa phải throughput train có MER/GPU.
- A cache một full volume và dùng ROI view + native Z gather, tương đương slab11-then-crop về pixel/geometry; tránh tạo slab512 lặp cho từng tile. Đây là prototype data-only, production Dataset chưa đổi.
- Benchmark đo hai itinerary tuần tự đã yêu cầu. VolumeShuffleSampler random chỉ được test correctness, chưa đo tốc độ; không tự đổi sampler train sang tile-major để lấy lợi thế cache.
- NPY lớn hơn file nén nhưng tránh decode lại/memory-map được; kết quả còn phụ thuộc format/storage, không chỉ phép cắt XY.
- Ba repetition chỉ mô tả độ phân tán đã thấy, không chứng minh ý nghĩa thống kê hay ổn định trên mọi máy.

## Giới hạn và cổng dừng

- Giữ giới hạn provenance của V3; không tuyên bố199 ca raw-exact. V3 stats fold0 không dùng cho nghiên cứu5fold.
- Đã kiểm sample/geometry, không thay model/loss/checkpoint/GT/preprocessing hoặc production sampler.
- RAM/disk khả dụng được kiểm trên máy chạy. Pilot vừa không chứng minh full199 sẽ vừa: riêng NPY float32 CT+uint8 GT của13705 masters ước tính khoảng16,73GiB trước overhead; không export full199 ở B3A.
- Chưa có kết quả segmentation, không chọn checkpoint hoặc threshold. Phải được review trước B3B; B4/B5 chưa thực hiện.
