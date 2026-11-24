# B3A — Kết quả và cách chạy lại

Branch/HEAD tại benchmark: `MADGNet` / `13d7695d494b9da40629afd6a45ac294285e2b08`.
Source hashes và trạng thái dirty được ghi trong b3a_benchmark.json; không có commit/push/merge mới.

## Evidence

- Synthetic: 19 test PASS; A-0003 thật: 4656 sample khớp, physical max error=0.0mm.
- Real correctness: 19.383s, lesion count trước/sau=738319/738319.
- b3a_benchmark.csv/json có toàn bộ repetition thực; b3a_precompute_manifest.json có416 file thuộc13 case. Không có array trong handoff.
- Đọc b3a_decision.md để xem median/dispersion/RSS/break-even và khuyến nghị có điều kiện.

## Chạy lại trên repo đầy đủ

```text
python -m pytest tests/test_b3a_delivery.py -q --basetemp <NEW_TEMP> --junitxml <NEW_XML>
python tools/b3a_compare.py --data-root <EXPORTED_V3_ROOT> --output-dir <NEW_EXPERIMENT> --synthetic-xml <PASS_XML> --workers 0 2 --batch-size 2 --repetitions 3
python tools/b3a_reports.py --experiment <EXPERIMENT> --output-dir <NEW_REPORT_DIR>
```

Phase prepare/correctness/benchmark có thể chạy riêng để tái sử dụng export đã hoàn chỉnh, không overwrite/re-export pilot. Không dùng b3a_precompute_manifest.json độc lập nếu thiếu data/marker/hash.

## Phạm vi source

- tools/b3a_delivery.py: hai Dataset prototype, canonical key, metadata và exporter/validation.
- tools/b3a_compare.py: chọn13 TRAIN, staging cùng storage, correctness và đo CPU full loader.
- tools/b3a_reports.py: tạo report từ số đo thật; tests/test_b3a_delivery.py kiểm semantics.
- Không sửa source production. Patch B3A chỉ thêm file mới; dependencies từ B2/V3 đã tồn tại trong workspace dirty.
- Không cherry-pick một commit B3A vì chưa có commit đó. Backup working tree, so patch từng file, kiểm source hashes và chạy tests trước khi áp sang branch khác.

## Cách đọc kết quả

Raw CSV/JSON tách warmup khỏi measured. Samples/s là tile-masters/s; original masters-equivalent/s = samples/s/16 vì mỗi lượt có đủ16 tile/master. Original volumes/patients không được đếm thành16 lần.
Kaggle status: KAGGLE_BENCHMARK_NOT_RUN. File notebook chỉ là công cụ chạy lại nếu chưa có job thật.


## Protocol chi tiet va cach dung notebook


### Dữ liệu và source

Làm trên repo MERIT đầy đủ, nhánh `MADGNet`, nền HEAD
`13d7695d494b9da40629afd6a45ac294285e2b08`, có thay đổi cũ chưa commit.
Không chuyển branch hoặc reset/clean. Hash218 file hiện có được lưu trước lượt
B3A để xác nhận source production không thay đổi.

Dataset: `C:\1_HDD\paper\material\merit_v3_precomputed_nii`. Đây là CT đã V3,
không phải raw chưa chuẩn hóa. YAML `native` là pass-through; `size_hw:null`.
Không fit/clip/Z-score lần hai, không resize hoặc thay mask/split. Giữ giới hạn
provenance của bộ xuất; không khẳng định đủ199 ca raw-exact. V3 stats chỉ fit
TRAIN fold0; B3A không dùng chúng để chạy nghiên cứu5fold.

13 ca được chọn deterministic seed42 từ TRAIN fold0, gồm A-0003 và12 ca đa dạng
Z/spacing/lesion sparsity. Không dùng VAL/TEST để chọn chất lượng; không sampling,
balancing hoặc bỏ tile trống. Pilot có1470 master và23520 tile-master/lượt.
Danh sách ID, shape, spacing và partition có trong `b3a_pilot.json`.

### Hai mode tương đương về dữ liệu

- **A dynamic:** cache một full volume/worker, chọn đúng11Z và ROI cố định theo
  tile key. Thực thi bằng ROI view rồi native Z gather để tránh copy full512slab
  lặp16 lần; đây chỉ là tối ưu của prototype, source Dataset production chưa đổi.
- **B precomputed-volume:** export một lần16 cặp volume tile `[Z,128,128]` mỗi ca,
  NPY float32 CT/uint8 GT lossless, sau đó mmap/cache một volume tile/worker.
  GT trong sample vẫn float32 giống A. Không lưu slab11 lặp thành file.
- 416 data files cho13 ca; không export full199. Lý do NPY: giữ bit pattern,
  hỗ trợ mmap và không đưa thêm NIfTI scaling/affine quantization.
- Cả hai trả cùng key `(original_case_id,master_z,tile_id)`, row→Y/col→X,
  image/mask11/GT master/validity/attention/valid_region và tile-local world pose.
- Tile affine float64 = full canonical affine × translate(x0,y0,0). Manifest giữ
  qform/sform nguồn và full geometry riêng. NPY không tự có header NIfTI;
  không dùng shape128 hoặc tile affine thay cho original full512 geometry.
- `cases`/`ends` nhóm theo original volume.13 CT không biến thành208 bệnh nhân.
- Metadata prototype có `full_geometry` và `tile_geometry` riêng; chưa phải
  drop-in replacement cho Native512 trainer/inference. B3B phải review adapter,
  model128, loss targets và stitch; chưa làm ở B3A.

### Correctness trước benchmark

19 test synthetic đã chạy: Z1/2/>11, mask rỗng/tiny/boundary, flip/oblique/shear,
qform/sform, CT-only không mở GT, restore, index/group/sampler, collate0/2worker,
tiếp tục từ consumed cursor và interrupted/corrupted export. Đây không phải
checkpoint/RNG/optimizer resume test của trainer.

A-0003 thật được kiểm toàn bộ291×16=4656 sample, kể cả context tại master
0,1,5,145,285,290. CT/GT bitwise-equal; physical max error0mm; reconstruct exact,
GT lesion738319 trước/sau. `exact_tensor_values` đếm cả context/master lặp trong
sample, không phải số voxel độc lập. Correctness FAIL thì không benchmark.

### Benchmark đo gì?

- Đủ23520 sample/11760 batch mỗi pass, batch2, cả hai mode cùng key order.
- Hai order case→master→tile và case→tile→master; workers0/2 khi đủ RAM.
- Warmup tách riêng;3 measured repetition theo AB/BA/AB, seed42. Không thay
  Population/số lượng mẫu giữa hai order; không đổi production sampler.
- Full DataLoader + collate + CPU materialization/sum, không chỉ reshape/crop.
  Không model, gradient, optimizer hoặc GPU/VRAM measurement.
- Worker startup/teardown có trong full-pass/first-batch latency;
  persistent_workers=False, pin_memory=False, prefetch_factor=2 khi có worker.
- Đo RSS main+descendants/workers20ms: shared pages có thể đếm lặp, không phải
  USS hoặc toàn bộ RAM vật lý/file cache của hệ thống.
- Process I/O counter là logical/IPC, có thể không phản ánh đầy đủ mmap page
  faults. `declared_open_file_bytes` là tổng kích thước file tại cache miss,
  **không phải số byte thực đã đọc từ SSD**.
- A dùng production reader với finite/binary validation khi load full volume.
  B dựa vào export/reload/hash validation trước timer, runtime kiểm shape/dtype.
  Đây là so sánh hai delivery pipelines trọn vẹn, không cô lập riêng phép cắt.
- OS cache không bị flush/kiểm soát cold-cache; không gọi là cold benchmark.
  CPU power/frequency và mọi ứng dụng nền cũng không bị khóa.
- Source V3 pilot và precomputed đều stage vào cùng NTFS C: ở local; notebook
  luôn stage cả hai về `/kaggle/working`, không so Input-vs-Working lẫn storage.
- Export/staging/correctness và các hash validation ngoài timer được báo riêng;
  break-even chuẩn dùng export_seconds như công thức prompt, không tự biến
  chi phí chưa đo (upload/cleanup/GPU) thành số0.

### Vấn đề phát hiện và đã sửa

Warmup đầu của prototype A tạo full512slab cho từng tile, gây copy thừa.
Đã dừng khi chưa có measured run nào, giữ log/source trong archive, thay bằng
ROI view + Z gather và chạy lại toàn bộ tests/correctness. Không trộn số cũ vào
benchmark cuối. Export B không đổi thuật toán và không chạy lại.

### Chạy notebook Kaggle CPU

1. Upload `kaggle_b3a_delivery_cpu.ipynb`, chọn **Accelerator: None**.
2. Add Input đúng dataset V3 có đủ199 CT/GT và4 file metadata. Nếu nhiều bản V3,
   điền `DATA_ROOT` trong cell đầu; không sửa dữ liệu để vượt kiểm tra.
3. Run All. Notebook giải nén snapshot230 source files, kiểm SHA; không cần
   Git clone/push và không kèm CT/GT/checkpoint trong notebook.
4. Notebook kiểm đúng13 ID, V3 stats và source hashes như local; chạy test, stage,
   export, correctness rồi mới benchmark. Memory gate có thể báo NOT_RUN cho
  2worker; phải giữ trạng thái thật đó.
5. Tải `B3A_KAGGLE_REPORT.zip` từ Output Files để review; dừng sau B3A.

Hiện chỉ kiểm compile/source CRC/SHA và pinned metadata của notebook trên local,
**chưa thực thi notebook trên Kaggle**. Mọi khuyến nghị local là provisional;
loader speedup không bảo đảm GPU train speedup. B4 mới đo end-to-end khi được yêu cầu.

### Source/patch và hướng dẫn chuyển sang branch khác

Handoff mang source snapshot đầy đủ và các file B3A mới. Không có commit B3A để
cherry-pick vì không commit/push. Backup working tree, xem patch từng file và
`git apply --check` trước khi áp. Không force/overwrite các thay đổi đang dở.
Các dependencies B2/V3 là source đã có sẵn trong workspace dirty, không được
giả định chỉ áp patch B3A lên HEAD gốc là đủ. Source SHA trong gói là mốc đối chiếu.

Không đưa `storage/`, array tile, CT/GT, checkpoint hoặc pytest synthetic arrays
vào handoff. Ghi chú tạm tắt máy sẽ được thay bằng README/decision cuối;
giữ XML/log/manifest/hashes làm bằng chứng.


## Audit cuoi

{
  "status": "PASS",
  "correctness_real_samples": 4656,
  "synthetic_tests": 19,
  "measured_runs": 24,
  "warmups": 8,
  "samples_per_pass": 23520,
  "batches_per_pass": 11760,
  "itinerary_and_payload_checksums_equal_within_each_order": true,
  "exact_payload_bytes_paired": true,
  "rss_includes_workers_verified": true,
  "protected_existing_files_unchanged": 218,
  "changed_protected_files": [],
  "benchmark_source_hashes_match": true,
  "precomputed_data_files": 416,
  "original_and_staged_source_hashes_verified": 52,
  "precomputed_gt_lesion_voxels": 2386389,
  "post_benchmark_export_hash_validation_seconds": 4.801902000000155,
  "validation_time_note": "New integrity validation measured after benchmark, not the original startup duration",
  "exporter_ast_unchanged_since_single_export": true,
  "kaggle_status": "KAGGLE_BENCHMARK_NOT_RUN",
  "b3b_started": false,
  "training_updates": 0
}

Timing: measured24 passes=1753.451s; warmup8 passes=587.085s. Day la DataLoader pass, khong phai train epochs.

Nguon CLI/source/metadata trong goi phai duoc giu nguyen SHA. Chi copy file moi sau khi review; khong co commit B3A de cherry-pick.
