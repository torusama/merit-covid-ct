# Ki?m tra tr??c khi push nh?nh train ? 24/11/2026

- Nh?nh ??ch: `train/merit-unetrpp-colab`; n?n `eacbfdc146d89f031f6cd377aacb7b2428d2e7b8`.
- B? ki?m tra li?n quan: **83 PASS / 1 FAIL**, kh?ng c? test b? b? qua.
- To?n b? 77 test intensity/QC/B2/B3A PASS; 6 test frozen BatchNorm PASS.
- FAIL: `test_full_trainer_restores_policy_and_matches_mid_epoch_resume`.
  Resume ch?y qua c?c batch nh?ng Windows b?o `PermissionError [WinError 5]`
  khi x?a `in_progress.pt` cu?i epoch; ch?y l?i ngo?i sandbox v?n g?p l?i.
- ?? t?i hi?n c?ng l?i v?i **source train n?n ch?a ?p V3** trong worktree train,
  t?i `training.py:505`; b?n m?i t??ng ?ng d?ng 534. Hai XML ??nh k?m:
  `publish_tests.xml` v? `publish_base_resume_test.xml`.
  ??y l? l?i c? s?n tr?n m?i tr??ng Windows ?ang ki?m tra; ch?a s?a trong l??t
  ch? xu?t b?n preprocessing. Kh?ng kh?ng ??nh resume ho?c to?n b? suite ??t.
- Notebook: compile 5 code cell, CRC v? SHA c?a 213 file source PASS; 13 hash
  core source kh?p benchmark th?c. Ch?a th?c thi notebook tr?n Kaggle/GPU.
- Ki?n tr?c model, loss, frozen BatchNorm v? code resume gi? nguy?n so v?i n?n.
- Fixture `tests/fixtures/covid19_20_cases_metadata.json` ch? ch?a metadata c?a
  manifest ?? test QC ch?y t? checkout s?ch; kh?ng ch?a ?nh ho?c checkpoint.
- Ph?n d? li?u ch? th?m V3/ki?m tra th?ng k?, B2 QC v? B3A prototype/benchmark.
  Ch?a t?ch h?p Tiling128 v?o production train; ch?a ch?y B3B ho?c train th?t.
