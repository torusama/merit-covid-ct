"""Create B3A data-only decision/readme from measured JSON, never invent runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def make_reports(experiment, output_dir):
    experiment, output = Path(experiment), Path(output_dir)
    benchmark = json.loads((experiment / 'b3a_benchmark.json').read_text(encoding='utf-8'))
    correctness = json.loads((experiment / 'b3a_correctness.json').read_text(encoding='utf-8'))
    pilot = json.loads((experiment / 'b3a_pilot.json').read_text(encoding='utf-8'))
    if correctness['status'] != 'PASS' or benchmark['status'] not in ('PASS', 'PASS_WITH_RESOURCE_LIMIT'):
        raise ValueError('Cannot recommend: correctness/benchmark gate not complete')
    output.mkdir(parents=True, exist_ok=False)
    lines = ['# B3A — Đề xuất từ số đo thực tế', '',
        '**Chỉ so sánh data delivery CPU; chưa B3B/model/train hoặc benchmark GPU.**', '',
        f"Kaggle: `{benchmark['kaggle_status']}`. Kết quả local không được gán thành kết quả Kaggle.", '',
        f"Pilot: {pilot['original_case_count']} original TRAIN volumes, {pilot['full_tile_master_count']:,} tile-master mỗi lượt, đủ16 tile/master.",
        f"Batch={benchmark['batch_size']}, seed42, repetitions={benchmark['repetitions']}, pin_memory=False.", '',
        '| Order | Workers | Dynamic median s (min–max) | Precomputed median s (min–max) | Throughput gain | Break-even pilot epochs |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    stable_candidates = []
    for item in benchmark['aggregates']:
        a, b = item['dynamic'], item['precomputed']
        stable = b['max_seconds'] < a['min_seconds']
        if item['throughput_gain_percent'] >= 15 and stable and item['pilot_epoch_break_even'] is not None and item['pilot_epoch_break_even'] <= 100:
            stable_candidates.append(item)
        lines.append(f"| {item['order']} | {item['workers']} | {a['median_seconds']:.3f} ({a['min_seconds']:.3f}–{a['max_seconds']:.3f}) | {b['median_seconds']:.3f} ({b['min_seconds']:.3f}–{b['max_seconds']:.3f}) | {item['throughput_gain_percent']:+.2f}% | {item['pilot_epoch_break_even'] if item['pilot_epoch_break_even'] is not None else 'không hòa vốn'} |")
    lines += ['', '## Nhận định có điều kiện', '']
    if len(stable_candidates) == len(benchmark['aggregates']) and stable_candidates:
        text = 'Precomputed-volume là ứng viên ưu tiên cho bước review tiếp: tăng throughput ≥15% trong mọi setup đã đo, các khoảng min–max không giao nhau và có hòa vốn trong100 lượt pilot.'
    elif stable_candidates:
        text = 'Kết quả phụ thuộc setup/order. Chỉ cân nhắc precomputed-volume tại các setup vượt15% ổn định; giữ dynamic làm lựa chọn đơn giản nếu chưa chốt được order/worker phù hợp. Không âm thầm đổi sampler production chỉ để đạt số đo thuận lợi.'
    else:
        text = 'Ưu tiên dynamic ở phạm vi hiện tại: chưa có setup chứng minh mức tăng≥15% ổn định đủ để bù thêm vận hành/storage.'
    if benchmark['kaggle_status'] == 'KAGGLE_BENCHMARK_NOT_RUN':
        text += ' Đây chỉ là **provisional recommendation**, phải đo Kaggle trước khi chốt.'
    lines += [text, '', '## Chi phí, RAM và độ phân tán', '',
        f"- Export một lần: {benchmark['export_seconds']:.3f}s; {benchmark['export_data_files']} file; {benchmark['export_data_bytes']:,} bytes.",
        f"- Stage cả hai mode về cùng storage: {pilot['staging_seconds']:.3f}s, {pilot['staging_bytes']:,} bytes; chi phí chung này không trộn vào steady-state/epoch.",
        '- Break-even = ceil(export_seconds / (median_dynamic_epoch - median_precomputed_epoch)) khi mẫu số dương.',
        '- Epoch ở đây là toàn bộ13-case pilot, không phải epoch127/199 CT. Không ngoại suy tuyến tính ra full199 vì cache/I/O/startup có thể thay đổi.',
        '- Chưa đo upload/copy qua mạng, cleanup hoặc GPU. Cleanup không được tự xóa dữ liệu gốc; prototype giữ output để audit.',
        '', '| Order | Workers | A stdev s | B stdev s | A peak RSS MiB | B peak RSS MiB | A/B first-batch median s |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for item in benchmark['aggregates']:
        a, b = item['dynamic'], item['precomputed']
        lines.append(f"| {item['order']} | {item['workers']} | {a['stdev_seconds']:.3f} | {b['stdev_seconds']:.3f} | {a['peak_rss_bytes']/2**20:.1f} | {b['peak_rss_bytes']/2**20:.1f} | {a['median_first_batch_seconds']:.3f} / {b['median_first_batch_seconds']:.3f} |")
    lines += ['', '- RSS gồm main+descendants/workers, lấy mẫu20ms; shared pages có thể đếm lặp, không phải USS hoặc RAM vật lý duy nhất.',
        '- OS cache không kiểm soát cold-cache. Đã warmup; mỗi lượt vẫn tạo iterator/worker và cache Dataset được reset.',
        '- Process I/O là counter logical có cả IPC; declared file bytes là kích thước file tại cache miss, không phải số page mmap thực đọc từ ổ.',
        '- Consumer ép materialize/sum tensor CPU của full DataLoader+collate. Đây chưa phải throughput train có MER/GPU.',
        '- A cache một full volume và dùng ROI view + native Z gather, tương đương slab11-then-crop về pixel/geometry; tránh tạo slab512 lặp cho từng tile. Đây là prototype data-only, production Dataset chưa đổi.',
        '- Benchmark đo hai itinerary tuần tự đã yêu cầu. VolumeShuffleSampler random chỉ được test correctness, chưa đo tốc độ; không tự đổi sampler train sang tile-major để lấy lợi thế cache.',
        '- NPY lớn hơn file nén nhưng tránh decode lại/memory-map được; kết quả còn phụ thuộc format/storage, không chỉ phép cắt XY.',
        '- Ba repetition chỉ mô tả độ phân tán đã thấy, không chứng minh ý nghĩa thống kê hay ổn định trên mọi máy.',
        '', '## Giới hạn và cổng dừng', '',
        '- Giữ giới hạn provenance của V3; không tuyên bố199 ca raw-exact. V3 stats fold0 không dùng cho nghiên cứu5fold.',
        '- Đã kiểm sample/geometry, không thay model/loss/checkpoint/GT/preprocessing hoặc production sampler.',
        '- RAM/disk khả dụng được kiểm trên máy chạy. Pilot vừa không chứng minh full199 sẽ vừa: riêng NPY float32 CT+uint8 GT của13705 masters ước tính khoảng16,73GiB trước overhead; không export full199 ở B3A.',
        '- Chưa có kết quả segmentation, không chọn checkpoint hoặc threshold. Phải được review trước B3B; B4/B5 chưa thực hiện.']
    if benchmark['skipped']:
        lines += ['', 'Setup chưa chạy:'] + ['- ' + json.dumps(item, ensure_ascii=False) for item in benchmark['skipped']]
    (output / 'b3a_decision.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    readme = ['# B3A — Kết quả và cách chạy lại', '',
        f"Branch/HEAD tại benchmark: `{benchmark['source']['branch']}` / `{benchmark['source']['head']}`.",
        'Source hashes và trạng thái dirty được ghi trong b3a_benchmark.json; không có commit/push/merge mới.', '',
        '## Evidence', '',
        f"- Synthetic: {correctness['synthetic_tests']} test PASS; A-0003 thật: {correctness['matched_samples']} sample khớp, physical max error={correctness['coordinate_max_abs_error_mm']}mm.",
        f"- Real correctness: {correctness['seconds']:.3f}s, lesion count trước/sau={correctness['lesion_voxels_before']}/{correctness['lesion_voxels_after']}.",
        '- b3a_benchmark.csv/json có toàn bộ repetition thực; b3a_precompute_manifest.json có416 file thuộc13 case. Không có array trong handoff.',
        '- Đọc b3a_decision.md để xem median/dispersion/RSS/break-even và khuyến nghị có điều kiện.', '',
        '## Chạy lại trên repo đầy đủ', '',
        '```text',
        'python -m pytest tests/test_b3a_delivery.py -q --basetemp <NEW_TEMP> --junitxml <NEW_XML>',
        'python tools/b3a_compare.py --data-root <EXPORTED_V3_ROOT> --output-dir <NEW_EXPERIMENT> --synthetic-xml <PASS_XML> --workers 0 2 --batch-size 2 --repetitions 3',
        'python tools/b3a_reports.py --experiment <EXPERIMENT> --output-dir <NEW_REPORT_DIR>',
        '```', '',
        'Phase prepare/correctness/benchmark có thể chạy riêng để tái sử dụng export đã hoàn chỉnh, không overwrite/re-export pilot. Không dùng b3a_precompute_manifest.json độc lập nếu thiếu data/marker/hash.', '',
        '## Phạm vi source', '',
        '- tools/b3a_delivery.py: hai Dataset prototype, canonical key, metadata và exporter/validation.',
        '- tools/b3a_compare.py: chọn13 TRAIN, staging cùng storage, correctness và đo CPU full loader.',
        '- tools/b3a_reports.py: tạo report từ số đo thật; tests/test_b3a_delivery.py kiểm semantics.',
        '- Không sửa source production. Patch B3A chỉ thêm file mới; dependencies từ B2/V3 đã tồn tại trong workspace dirty.',
        '- Không cherry-pick một commit B3A vì chưa có commit đó. Backup working tree, so patch từng file, kiểm source hashes và chạy tests trước khi áp sang branch khác.', '',
        '## Cách đọc kết quả', '',
        'Raw CSV/JSON tách warmup khỏi measured. Samples/s là tile-masters/s; original masters-equivalent/s = samples/s/16 vì mỗi lượt có đủ16 tile/master. Original volumes/patients không được đếm thành16 lần.',
        f"Kaggle status: {benchmark['kaggle_status']}. File notebook chỉ là công cụ chạy lại nếu chưa có job thật."]
    (output / 'B3A_README.md').write_text('\n'.join(readme) + '\n', encoding='utf-8')
    return dict(status='PASS', reports=str(output), provisional=benchmark['kaggle_status'] == 'KAGGLE_BENCHMARK_NOT_RUN')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_reports(args.experiment, args.output_dir)))
