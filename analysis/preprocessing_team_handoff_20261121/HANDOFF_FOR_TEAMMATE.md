# Bàn giao preprocessing MERIT-UNETR++

**21/11/2026 — READY_FOR_TEAM_HANDOFF cho phần preprocessing.**
Đã QC **199/199 cặp qua reader V3 production**:127 TRAIN,32 VAL,40 TEST đều PASS;
GT trước/sau có cùng số voxel lesion và **0 tọa độ lệch** sau đảo về lưới gốc.
QC CPU từng ca mất516,87s, không fit/train hoặc tính Dice/IoU trên dữ liệu này.

## Phiên bản source

- Repo gốc: https://github.com/torusama/merit-covid-ct
- Commit nền: `13d7695d494b9da40629afd6a45ac294285e2b08`.
- Working branch lúc đóng gói: `MADGNet`, **dirty, chưa commit/push/merge**.
- Đây là **gói source MERIT cần thiết đã chụp đầy đủ**, không phải overlay cũ hay
  một commit mới. Source nằm ở `project/`; không cần chép thêm file từ workspace Codex.
- `SOURCE_PROVENANCE.json` và `SOURCE_FILE_MANIFEST.json` ghi hash từng file, những
  file khác commit nền, và phân biệt thay đổi nội dung với khác biệt CRLF/LF.
  Các nghiên cứu MADGNet, notebook/demo và kết quả cũ không liên quan không được gói.

Giải nén vào **thư mục mới**, không apply lên run/checkpoint cũ. Nếu quản lý bằng Git,
dùng commit nền trên rồi nhập đúng các file snapshot theo manifest; chỉ checkout
commit nền mà không lấy các file snapshot sẽ **chưa có V3**.

## Bất biến của V3

`Case → kiểm CT/GT → cùng LPS → CT clipped-Z-score / GT nguyên nhãn → native512 → 11 lát → loader`.
Giữ nguyên model/loss/BN/optimizer/sampler/evaluator. Padding−2048 chỉ được loại khỏi
miền FIT theo heuristic đã khóa; input vẫn nhận cùng công thức cho **mọi voxel**:

```text
y = (clip(x, -1241, 1032) - (-583.0622212292245)) / 488.0394718302036
```

Không zero-fill/crop/resize/resample, không normalization GT và không dò padding
lại khi inference. CT-only inference vẫn dùng `require_mask=False`.
Spacing **deferred/out-of-scope**: sampling/position/weighting đều giữ `index`.

Stats cố định: `project/analysis/preprocessing/fit_stats.json`, fit từ127 TRAIN;
120 CONFIDENT/7 UNKNOWN fallback. **Không fit lại để cài bundle.**

- SHA256 toàn file stats: `583b32eb4bdeeef36db9dbc10282b221b13cd8add18c7ff18230e7f5aa9a38c4`.
- Checksum provenance bên trong: `d80d5f347b004375ef667a767fa8fa10d1ddcbbe9dd8acbfb6f14945d80ab03b`.

## Dựng môi trường và trỏ dataset

Từ thư mục đã giải nén, kiểm source trước; sau đó cài từ `project/`:

```bash
python verify_bundle.py
cd project
python -m pip install -e ".[unetrpp,test]"
```

Dependencies khai báo ở `pyproject.toml`; môi trường đã kiểm ghi trong báo cáo CPU.
Package đã build wheel offline và qua31 tests CPU trong thư mục source tách biệt.
Chưa thử cài/download dependency trong tài khoản Colab/Kaggle của người nhận.

Giữ `data/covid19_20/cases.json` và `splits/covid19_20_5fold.json` **đúng bytes** đã
cung cấp. Manifest dùng đường dẫn tương đối `Train/..._ct.nii.gz`/`..._seg.nii.gz`.
Đưa cache199 cặp hiện có vào đúng vị trí bằng liên kết thư mục, không copy/đổi CT:

```bash
# Khi đang ở project/, Linux/Colab/Kaggle: thay target bằng thư mục Train thật.
ln -s /duong_dan_cache/Train data/covid19_20/Train
```

Windows có thể dùng `New-Item -ItemType Junction -Path data/covid19_20/Train -Target 'C:\cache\Train'`.
Hoặc đặt manifest nguyên bytes cạnh thư mục Train hiện có, rồi sửa **chỉ đường dẫn**
`data.manifest`, `data.split_path`, `preprocess.zscore_fit` trong một YAML runtime
riêng. Không sửa danh sách case/split hoặc số stats. Bản source tách biệt đã đọc
batch thật thành công với cách đổi đường dẫn này.

Ở thư mục ngoài `project/`, có thể kiểm nhanh398 hash của cache (không decode/fit):
`python verify_bundle.py --data-root /duong_dan_cache` với target chứa `Train/`.
Hash phải khớp CSV QC; thiếu/sai file thì dừng, không refit để bỏ qua sai khác.

## Kiểm CPU và dùng cho training mới

```bash
# Tại project/; lệnh này không tạo model hoặc train.
python scripts/read_training_batch.py --config configs/merit_preprocessing.yaml --fold 0 --batch-size 1 --workers 0
```

Kỳ vọng: CPU,float32,`[1,1,11,512,512]`, phân vùng127/32/40. Master đầu có thể có GT
rỗng hợp lệ, không phải dự đoán all-background của model. Dùng **config V3 này cho
training mới** qua trainer hiện hữu, output mới; các giá trị model/loss/sampling
không cần chỉnh. **Không resume Native checkpoint như thể đã train bằng V3.**
Native/global train_zscore còn nguyên để tái lập run cũ. Người train cần xác nhận
môi trường/GPU và bắt đầu một run mới; lượt bàn giao này không chạy bước đó.

## Bằng chứng và giới hạn

- `reports/all199_ct_gt_qc.csv` + summary: từng case, count/coordinate/hash/partition.
- `reports/CPU_TEST_REPORT.md`: tests source riêng, build offline, batch và integrity.
- Header/affine/orientation/spacing theo logic hiện có; exact binary GT được kiểm
  trước cast float32; source hashes trước/sau bằng nhau. Trong summary gốc,
  `source_files_hash_unchanged=199` đếm **cặp**, tương ứng398 file.
- Chưa có Dice V3, model/GPU test hoặc kết luận chữa collapse. HU và patient identity
  chưa xác minh; giữ giả định đơn vị mm cũ cho header unknown.7 UNKNOWN vẫn giữ toàn
  voxel trong FIT, không có nghĩa là7 CT hỏng. Clipping mất phân biệt ở tails.
- Không xóa file production/nghiên cứu nào ở lượt chốt; chỉ dọn các thư mục tạm do
  lượt kiểm bundle này tạo sau khi đã đóng gói và kiểm chứng.
