# B2 — Kết quả QC Tiling128 trên 199 ca V3

Ngày: **24/11/2026, Asia/Bangkok**. **B2 PASS — 199/199 cặp CT/GT thật**, không
có ca FAIL và không còn ca chưa kiểm. Đây là QC dữ liệu, không phải kết quả model.

## 1. Đã chạy ở đâu và trên dữ liệu nào?

- Đọc bản precomputed V3 local tại
  `C:\1_HDD\paper\material\merit_v3_precomputed_nii`.
- Có đủ `EXPORT_COMPLETE.json`, manifest, split, YAML và 398 file CT/GT.
  Tất cả 398 hash file trước/sau khớp hash bộ xuất.
- Source làm việc: nhánh **MADGNet**, nền commit `13d7695d494b9da40629afd6a45ac294285e2b08`,
  có thay đổi chưa commit từ các lượt trước. SHA từng file QC/reader/slab/config
  được lưu trong JSON; không coi toàn bộ snapshot là một commit mới.
- **Chưa chạy job trên Kaggle.** Đã chuẩn bị CLI portable và lệnh Kaggle, kiểm
  import và cùng bộ unit test trên bản source tách biệt.
- Không tự tải dữ liệu, không dùng lại pipeline chuẩn hóa để xuất V3 lần nữa.

## 2. PASS/FAIL theo yêu cầu

| Mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Manifest/export/fingerprint | PASS | Đủ 199 case ID, file/hash/shape và phân vùng khớp |
| Tách biệt split volume/group | PASS | Validator đủ 5 fold; fold0 TRAIN127/VAL32/TEST40; không chia lại |
| Shape/affine/orientation/spacing CT–GT | PASS | 199 cặp qua kiểm tra geometry của reader production |
| CT finite và GT binary | PASS | CT float32 hữu hạn; GT đúng 0/1 trước cast của reader |
| Tiling 4×4, 128×128 | PASS | 13.705 lát → 219.280 cặp tile CT/GT, không resize/overlap/interpolation |
| So equality toàn bộ voxel | PASS | 3.592.683.520 vị trí CT/GT được so sau ghép; CT/GT mismatch=0, CT bit-pattern mismatch=0 |
| Bảo toàn lesion | PASS | 14.818.707 pixel GT dương trước và sau bằng nhau |
| Slab11 edge/unique | PASS | 1.791 master đại diện trên 199 ca, kiểm cả context GT đúng Z/XY |
| Tọa độ tile/partition | PASS | Đối soát độc lập đủ 219.280 dòng metadata, đúng row→Y, col→X và master_z |
| Giữ nguyên input | PASS | 398 source hashes trước/sau khớp; manifest/split/config/export không đổi |
| Không sửa source production hiện có | PASS | 203 file source/config/split/test/tool đã tồn tại giữ nguyên hash |
| Unit tests | PASS | 27 test trên repo; cùng 27 test trên bản source portable cũng PASS |
| Model/training/B3 | NOT_RUN | Không import Torch trong CLI QC, không tạo model hoặc optimizer update |

Danh sách lỗi ca: **rỗng**. `status=PASS`, `b2_gate_pass=true` chỉ áp dụng cho
thao tác QC B2 trên bản precomputed V3 đã đọc.

## 3. Số tile theo fold 0

Mỗi tile được đếm cho **một master slice**, có lesion nếu GT tile chứa ít nhất
một pixel 1. Không nhân thêm 11 cho context và chưa sampling/balancing.

| Partition | CT/GT | Tổng tile | Tile có lesion | Tile trống | Pixel lesion |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 127 | 140.960 | 8.507 | 132.453 | 10.283.591 |
| VAL | 32 | 37.152 | 2.097 | 35.055 | 1.642.366 |
| TEST | 40 | 41.168 | 2.458 | 38.710 | 2.892.750 |
| **Tổng** | **199** | **219.280** | **13.062** | **206.218** | **14.818.707** |

Tile có lesion chỉ khoảng 5,96% tổng tile. Đây là số liệu để thảo luận sampling
ở bước sau; **không tự thay đổi tỷ lệ lấy mẫu trong B2**.

## 4. Kiểm ca A-0003 và artifact B1

- Lượt smoke A-0003: 291 lát, 4.656 cặp tile, 261 tile có lesion và 4.395 tile
  trống. CT/GT mismatch=0; 738.319 pixel lesion giữ nguyên; 9 master slab đạt.
- Lượt này báo **PARTIAL_PASS, 1/199 tested**, không giả thành 199 PASS.
- Sau đó A-0003 nằm trong lượt đủ199, không cộng lặp thành 200 ca.
- Đã tìm thấy NPZ B1 tại Downloads và đọc thực tế; không giả định nó có trong repo.
  Crop B2 tại A-0003/master145 khớp NPZ về CT, GT, slice_indices và slice_valid;
  CT/GT mismatches=0. Checksum và kết quả trong `B1_ARTIFACT_CROSSCHECK.json`.
- Không đưa tensor NPZ vào gói B2; chỉ đưa metadata và evidence nhỏ.

## 5. Thời gian

| Lượt | Thời gian |
| --- | ---: |
| 27 unit tests trên repo | 1,79 giây |
| 27 test trên source portable | 2,11 giây |
| Smoke riêng A-0003 | 7,38 giây |
| **QC đầy đủ199 ca** | **361,45 giây ≈ 6 phút 1,45 giây** |

Lượt đủ199 gồm kiểm input 0,19 giây, tổng thời gian từng ca khoảng 359,97 giây
và phần điều phối/metadata/GC còn lại. Thời gian từng ca gồm hash trước/sau,
đọc CT/GT, so tất cả voxel và kiểm slab. Không gồm chuẩn bị code, cài môi trường,
đóng gói ZIP hoặc chạy Kaggle; không dùng các số này để dự đoán thời gian train.

## 6. File mới và cách chạy

- [CLI QC](../../tools/qc_tiling128.py).
- [Unit tests](../../tests/test_tiling128_qc.py).
- [Hướng dẫn local/Kaggle, schema output và phạm vi B3](../../docs/B2_TILING128_QC.md).
- File report này; cập nhật ghi chú hội thoại tại `CONVERSATION_HANDOFF.md`.

Không sửa file Dataset/DataLoader/model/loss/inference/config production.
`B2_DIFF.patch` trong bộ output chỉ gồm các file B2 mới, không gom thay đổi cũ
của workspace vào diff của lượt này.

Output canonical:

```text
outputs/tiling128_b2_20261124/
  qc199/tiling128_qc_199.json
  qc199/tiling128_qc_199.csv
  qc199/tiling128_tile_metadata.jsonl.gz
  a0003/...
  tests_initial.xml
  tests_portable.xml
  B1_ARTIFACT_CROSSCHECK.json
  ARTIFACT_AND_SOURCE_VERIFICATION.json
  B2_DIFF.patch
  TILING128_B2_QC_HANDOFF.zip
```

JSON chứa hash từng CT/GT, hash source/metadata, geometry của từng volume và chi
tiết slab. CSV có một hàng cho mỗi ca. Gzip JSONL chỉ lưu tọa độ/count từng tile,
không ghi tensor tile ra ổ. ZIP có source subset độc lập, tests, hướng dẫn và
báo cáo; không có CT, GT, checkpoint hoặc code model.

## 7. Những giới hạn không được bỏ qua

1. Bộ xuất gốc ghi **COMPLETE_WITH_PROVENANCE_LIMITATION**: chưa chứng minh byte/
   voxel chính xác với dữ liệu raw gốc. B2 giữ nguyên giới hạn này; chỉ chứng minh
   thao tác tiling bảo toàn đúng bản V3 hiện có.
2. Split không leak **volume/group**; danh tính bệnh nhân vẫn chưa được xác minh.
3. Giữ `assume_mm` và metadata đơn vị theo YAML/reader hiện tại; B2 không bổ sung
   bằng chứng đơn vị của dữ liệu raw hay đổi spacing.
4. Không fit/clip/Z-score lại. `native` ở đây là pass-through của CT đã V3.
5. QC dữ liệu không chứng minh tương thích model/checkpoint grid128 hoặc tăng
   Dice/IoU. Chưa có một kết quả train tiling nào trong lượt này.

## 8. Stop Gate

**Dừng sau B2.** B3 mới cần index `(volume,master_z,tile)` trong Dataset/DataLoader,
kiểm model input/grid128 và checkpoint, nhãn presence/salience theo tile/context,
và ghép đủ16 tile về512 trước metric. Danh sách file dự kiến nằm ở cuối hướng dẫn.
Chưa thực hiện các thay đổi đó; chưa commit/push/merge.
