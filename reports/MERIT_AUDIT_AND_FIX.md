# MERIT-UNETR++ COVID: audit, sửa và xác minh

Ngày 11/11/2026. Phạm vi thay đổi: source hiện tại tại HEAD `2638a79f3f63b9ad85391ae3b6e107ed3edb2aa8` cùng các thay đổi nêu bên dưới. Hai notebook `notebooks/colab_github.ipynb` và `notebooks/train_merit_colab.ipynb` đã sửa trước task, không bị chỉnh sửa trong task này. Dữ liệu nguồn và split không bị thay đổi. Không train full, không sử dụng Zenodo20.

## A. Môi trường và phạm vi

- Windows, Python 3.11.9, PyTorch 2.5.1+cu118, MONAI 1.5.2; CUDA khả dụng, GeForce GTX 1650 4 GiB (thông số GPU chi tiết theo preflight trước). Seed baseline 42, [`configs/merit_index.yaml`](../configs/merit_index.yaml), split hiện hữu.
- Không có phiên Codex document/Colab được kết nối qua công cụ hiện có; chạy kiểm thử ngắn trên máy cục bộ. Không download dataset, không đọc toàn bộ 199 volume trong task này theo yêu cầu người dùng.
- Lần đầu chạy nhiều tiến trình đồng thời khiến Windows báo WinError 1450/1455 (thiếu system resources/pagefile), có test bị access violation; chạy lại tuần tự. Full suite trước khi sửa bị gián đoạn và không được tính là PASS; trước task, lượt review trước đã chạy 77 selected test, nhưng đó không thay thế full suite trước sửa.

## B. Findings

| ID / mức / loại | Hành vi, bằng chứng và xử lý |
| --- | --- |
| F1 / cao / UNVERIFIED | CT `native` dùng giá trị đã scale mà chưa xác minh HU từng ca; [`preprocessing.py`](../src/merit_covid/data/pipeline/preprocessing.py). Preflight 127 train fold 0 có -9653 đến 32737 và 39 ca ngoại lai; [báo cáo cũ](../outputs/merit_preflight_fold0/REPORT.md). Outlier không chứng minh calibration sai hay xác định được padding. Đã thêm z-score tùy chọn fit trên toàn train fold; chưa fit dữ liệu thật hoặc chứng minh tốt hơn native. Cần nguồn DICOM/challenge và thống kê từng ca. **BLOCKED về HU; RESEARCH PENDING về chính sách**. |
| F2 / cao / UNVERIFIED | Ba ca A-0161, A-0585, A-0599 thiếu unit; baseline thực sự `assume_mm: true`, physical template `false`. Affine có thể tính displacement nhưng không xác minh đơn vị, slice thickness hoặc vị trí acquisition bị mất. HD95/volume theo mm phụ thuộc giả định. Đọc nguồn hoặc báo cáo rõ giả định; chưa chứng minh unit. **BLOCKED**. |
| F3 / cao / UNVERIFIED | Split được xác minh theo case/group/path, group nguồn là `volume_id_fallback`; không có patient identity tin cậy. Không tuyên bố patient-level CV; trùng nội dung dưới tên khác chưa kiểm. Không sửa split. **BLOCKED**. |
| F4 / cao / RISK | Resize 128 có thể xóa lesion nhỏ. Preflight train fold 0 đo 325 component 2D mất, 4,4185% foreground mất sau down/up, 34 positive slice thành rỗng; 256 giảm mất mát nhưng GTX 1650 OOM trong preflight cũ. Đây là QC nhãn, không phải Dice, và chưa tái lập trên toàn cohort. Không đổi mặc định hoặc crop theo GT. **RESEARCH PENDING**. |
| F5 / trung bình / MISSING | Duplicate slice chỉ bị loại khỏi loss; TEP/MER attention vẫn dùng bản sao, đặc biệt ở biên. Có thể làm lệch attention; chưa chứng minh làm giảm chất lượng. Đã thêm mask tùy chọn, mặc định tắt, test cả trường hợp không còn context hợp lệ. Conv3D trước attention vẫn nhìn thấy lát trùng. **PASS về hành vi tùy chọn; RESEARCH PENDING về hiệu quả**. |
| F6 / trung bình / MISSING | Train MERIT không có augmentation đồng bộ; sampler hiện tại thăm mỗi lát một lần nên volume dài đóng góp nhiều hơn. Đã thêm in-plane flip và lấy mẫu cố định số master mỗi volume với replacement, đều tắt mặc định và chỉ cho train. **PASS về phần mềm; hiệu quả cần ablation**. |
| F7 / trung bình / MISSING | Dice/IoU, precision/recall, HD95, per-case và group macro đã có; empty HD95 là null. Đã thêm slice FP, recall theo kích thước component GT 3D và presence AUROC/ECE/Brier theo từng case; confidence intervals và calibration tổng hợp theo fold chưa tích hợp. **PASS test tổng hợp; MISSING CI và pooled calibration**. |
| F8 / trung bình / RISK | Index context 11 lát tương ứng ±25 mm nếu bước 5 mm và ±5 mm nếu 1 mm. Physical sampler chọn native gần nhất, không nội suy Z, có thể trùng/bỏ qua lát; actual offsets lấy theo affine. [Protocol E1–E4](../configs/merit_spacing_ablation.md) yêu cầu thống nhất unit policy; chưa chứng minh mm tốt hơn index. **RESEARCH PENDING**. |
| F9 / trung bình / UNVERIFIED | Geometry checks, nearest-exact mask, CT-only master inference, native evaluation và checkpoint resume đã có test. Pilot cũ chỉ một ca A-0025, không chứng minh 199 ca. Không giả định kết quả unit/affine chưa xác minh. **PARTIAL**. |

## C. Thay đổi source

| File | Thay đổi và lý do | Bằng chứng |
| --- | --- | --- |
| `data/pipeline/slab.py` | Xuất `attention_valid` duy nhất theo native slice, độc lập loss policy. | `test_duplicate_mask_respects_master_and_is_independent_of_loss` |
| `models/merit_unetrpp/modules.py`, `model.py`, `adapter.py` | `mask_duplicate_context` opt-in: mask trước softmax TEP/MER; MER không context hợp lệ cho update bằng 0. Không thay baseline. | `test_masked_tep_and_mer_ignore_context_and_have_finite_gradients`, `test_mer_masked_chunking_matches_full` |
| `engine/training.py` | Flip XY đồng bộ và opt-in, truyền mask; kiểm tra fingerprint checkpoint cũ trước khi chuẩn hóa defaults mới để resume an toàn. | `test_flips_are_seeded_and_shared_across_slices_and_masks`; regression/resume suite |
| `data/pipeline/dataloader.py`, `scripts/merit_pipeline.py`, `scripts/train_colab.py` | Sampler train-only cố định số master mỗi volume với replacement, seed/generator checkpoint; mặc định zero giữ sampler cũ. | `test_balanced_sampler_counts_seed_and_train_only`; regression/resume suite |
| `configs/merit_spacing_ablation.md` | Ghi cách dựng E1–E4 trên cùng unit/intensity/split, để tham số chưa chọn là placeholder. | Kiểm tra cấu hình thủ công; không chạy full ablation |
| `scripts/fit_train_zscore.py`, `data/pipeline/config.py`, `preprocessing.py`, `engine/training.py` | Fit mean/std theo native scaled voxel của toàn train fold; lưu fold/split/case IDs và kiểm tra trước train. Không sử dụng validation/test voxels; baseline native không đổi. | `tests/test_research_metrics_normalization.py` |
| `data/pipeline/metrics.py`, `engine/slab_inference.py`, `models/merit_unetrpp/adapter.py` | Xuất presence master không GT, tính FP trên lát rỗng, 3D component recall theo native voxels và Brier/ECE/AUROC theo case sau inference. | `tests/test_research_metrics_normalization.py`, metric và inference regression |
| `configs/merit_spacing_e2.yaml`, `e3`, `e4` | Template thống nhất unit policy với E1; spacing và sigma cần được đặt rõ trước khi chạy. | Các template chưa dùng để train |
| `tests/test_repository_hygiene.py`, `README.md` | Khai báo ngoại lệ duy nhất cho báo cáo tiếng Việt theo yêu cầu, không quét virtualenv `site-packages` như mã dự án; liên kết báo cáo. | `test_repository_hygiene.py` |

Tùy chọn `model.mask_duplicate_context: true`, `training.augmentation_flips: true`, `training.samples_per_volume: <số dương>` chỉ dành cho ablation. Không có bằng chứng chúng tốt hơn baseline. Flips dùng RNG PyTorch và cùng phép biến đổi cho mọi lát, mask và valid region; validation/inference không dùng augmentation. Sampler có replacement nên không đảm bảo mọi lát được thăm trong mỗi epoch; cần ghi số mẫu và so sánh công bằng.

## D. Kiểm thử và cách chạy lại

- QC thực tế ba ca fold-0 train A-0003/A-0011/A-0013: stored int16, slope/intercept 1/0, đơn vị header mm, range lần lượt [-2048,1604], [-2048,1825], [-2048,2103]. Histogram và percentile từng ca ở [intensity_cases.csv](../outputs/merit_audit_20261111_sample2/intensity_cases.csv). Không kết luận -2048 là padding hay HU đã xác minh.
- Trên cùng ba ca, 512 component 2D; 128 mất 118 component và 3,2822% foreground native sau down/up, 4 positive slice thành rỗng; 256 mất 79 component và 1,7886%, không slice thành rỗng. [Số liệu](../outputs/merit_audit_20261111_sample2/resize_summary.csv) chỉ là mẫu train, không đại diện 199 ca.
- Pilot CUDA full model baseline trên A-0025: 4 optimizer steps, 0 AMP skip, gradient hữu hạn, checkpoint khớp khi resume; CT-only inference đủ 61 lát, restore 512×512×61, affine khớp, probability hữu hạn [0,0711; 0,9523], peak allocated 1021,62 MiB. [Pilot JSON](../outputs/merit_audit_20261111_pilot/pilot.json). Kết quả không đo chất lượng segmentation.
- Synthetic CPU và CUDA smoke: train, validate, checkpoint reload và resume hoàn tất; [CPU](../outputs/merit_audit_20261111_cpu_smoke.json), [CUDA](../outputs/merit_audit_20261111_cuda_smoke2.json). Selected option tests: 6 pass. Full suite trước sửa bị gián đoạn; lượt full sau sửa đầu tiên 347 pass, 2 fail do hygiene (báo cáo tiếng Việt và `site-packages`), sau đó đã sửa đúng phạm vi và chạy lại nhóm hygiene/sampler: 15 pass.
- Suite chọn lọc sau các thay đổi cuối: `python -m pytest tests/test_merit_unetrpp.py tests/test_merit_pipeline.py tests/test_preprocessing.py tests/test_reconstruction.py tests/test_splitting.py tests/test_merit_audit_options.py -q` — **83 passed, 0 failed, 0 skipped, 70,94 giây**, CPU local.
- `python scripts/merit_pipeline.py check-split --config configs/merit_index.yaml`: **PASS** theo volume/group; fold 0–3 là 127/32/40, fold 4 là 128/32/39. SHA-256 LF `7b31f9e250e64826f1fff0d9fec4a4aba034252cec7d5123d5f0393ec665e54d`; không chứng minh patient-level.
- GitNexus index còn chậm hơn HEAD hai commit; refresh bị dừng sau khi máy cạn tài nguyên. Kết luận dựa trên source và test trực tiếp, không khẳng định graph cập nhật.
- Lần full suite thứ hai sau sửa hygiene đã xuất hiện hai dấu `F` ở khoảng 41%, sau đó bị dừng khi Windows không khởi tạo được shell; không có traceback cuối nên **chưa xác định nguyên nhân từng failure và full suite không PASS**. Không suy ra hai failure đó chắc chắn do tài nguyên. Cần chạy lại toàn suite trên môi trường ổn định để lấy traceback.
- PDF paper không có trong `references/` của checkout tại thời điểm kiểm tra; đối chiếu dựa trên source, tests và tài liệu mapping đã có, không phải tái kiểm độc lập từng phương trình với bản PDF.
- Lượt bổ sung: `tests/test_research_metrics_normalization.py` **3 passed, 19,32 s**; `tests/duong/test_duong_metrics.py` cùng hai test inference/validation của `tests/test_merit_pipeline.py` **18 passed, 12,81 s**. Sau khi thêm test evaluation, chạy `tests/test_research_metrics_normalization.py` cùng test checkpoint resume của `tests/test_merit_pipeline.py`: **5 passed, 37,56 s**. Chỉ dùng dữ liệu tổng hợp. Không chạy lại full suite hay audit 199 ca vì yêu cầu giới hạn thời gian.

Z-score dùng mọi voxel native scaled trước resize, gồm cả khả năng có padding; không tự gọi là HU và không clipping. Script chỉ fit toàn bộ train partition và lưu case ID, fold, split hash trong YAML/checkpoint. Kiểm tra này chống nhầm fold/partition, không chứng minh nội dung CT không bị thay đổi sau fit; cần pin/hash nguồn riêng. `scripts/fit_train_zscore.py --config configs/merit_index.yaml --fold 0 --output outputs/zscore_fold0.yaml` sẽ đọc **toàn train fold**, nên không chạy trong lượt kiểm thử ngắn này. Giá trị fit phải được lựa chọn bằng train/validation và fit lại cho từng fold; không dùng external/test.

Lesion recall dùng GT native 3D 6-connected và coi phát hiện khi có bất kỳ voxel giao nhau. Bin `<100`, `100–999`, `>=1000` là số voxel, phụ thuộc spacing và không phải kích thước vật lý hay phân loại lâm sàng; voxel overlap rất dễ đạt nên không thay thế phân tích lesion có ngưỡng. ECE là 10 bin bằng chiều rộng, Brier/AUROC tính theo slice trong từng case; AUROC null nếu GT chỉ có một lớp, bin rỗng không báo, không cộng dồn các case. Những con số này không phải calibration được xác nhận.

Các test kiểm chứng phần mềm, không đo hiệu năng lâm sàng. Chạy từng lệnh tuần tự trên máy ít RAM/VRAM:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python -m pytest -q
python -m pytest tests/test_merit_unetrpp.py tests/test_merit_pipeline.py tests/test_preprocessing.py tests/test_reconstruction.py tests/test_splitting.py tests/test_merit_audit_options.py -q
python scripts/merit_pipeline.py check-split --config configs/merit_index.yaml
python scripts/merit_preflight.py inspect --config configs/merit_index.yaml --fold 0 --sizes 128 256 --assume-mm --block-depth 8 --limit 3 --output outputs/new_sample_qc
python scripts/merit_pipeline.py smoke --device cuda --output outputs/new_cuda_smoke.json
```

Không thêm `--assume-mm` nếu chưa có chính sách đơn vị được phê duyệt trong protocol; tham số ở lệnh QC chỉ minh họa giả định cũ và không chứng minh nguồn là mm. Output tool dùng tên mới để tránh ghi đè. Các ca `--limit` là sample, không đại diện toàn cohort hay dùng để chọn test set.

## E. Câu hỏi nghiên cứu còn lại

TEP có giúp COVID đa ổ không; MER tận dụng được context xa không; PCG có làm mất lesion nhỏ không; physical sampling/PE/loss có cải thiện không; chọn độ phân giải/crop thế nào theo retention và VRAM: tất cả đều **RESEARCH PENDING**, không phải bug đã chứng minh. Paper mô tả nodule LIDC ROI, project dự đoán COVID whole slice. Module là reimplementation theo mô tả paper, không phải implementation đã được tác giả xác nhận.

## F. Readiness

**Baseline kỹ thuật:** có thể chạy pilot giới hạn để kiểm tra phần mềm, với các giả định và dữ liệu được ghi rõ. **Full 5-fold để công bố:** chưa sẵn sàng do F1–F4 và validation dữ liệu thật còn thiếu. Chưa chứng minh patient independence, HU, đơn vị vật lý, lesion retention chấp nhận được hay hiệu năng các ablation. Cần bổ sung metric, khóa preprocessing/split/checkpoint protocol và chạy pilot trước khi full training.
