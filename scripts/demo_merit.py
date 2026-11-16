"""Interactive CT segmentation demo for the MERIT research model."""
from pathlib import Path
import sys, json

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import pandas as pd
import streamlit as st
from tools import merit_demo as demo

st.set_page_config(page_title='MERIT CT Demo', layout='wide')

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(max_entries=4, show_spinner='Loading CT volume...')
def load_case(ct, mask):
    return demo.view_case(ct, mask)


@st.cache_resource(show_spinner='Loading model checkpoint to GPU...')
def model_resource(model_id):
    sel = next(m for m in demo.models() if m['id'] == model_id)
    return demo.load_model(sel)


@st.cache_data(show_spinner=False)
def cached_predict(model_id, case_id, ct_path, master_idx):
    sel = next(m for m in demo.models() if m['id'] == model_id)
    cache = demo.cache_file(case_id, master_idx, sel)
    if cache.exists():
        return np.load(cache)
    try:
        model, preprocess = model_resource(model_id)
        return demo.predict(model, preprocess, case_id, ct_path, master_idx, sel)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_history(history_path):
    if not history_path:
        return []
    path = ROOT / history_path
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r['epoch'])


# ---------------------------------------------------------------------------
# Sidebar (always visible, independent of tabs)
# ---------------------------------------------------------------------------
registered = demo.models()
model_map = {m['id']: m for m in registered}

with st.sidebar:
    st.title('MERIT CT Demo')

    st.subheader('Model')
    selected_model_id = st.selectbox(
        'Checkpoint',
        [m['id'] for m in registered],
        format_func=lambda x: f"{model_map[x]['label']}  ({x[:8]}...)",
        label_visibility='collapsed',
    )
    selection = model_map[selected_model_id]

    report = demo.model_report(selection)
    if report:
        mean_dice = np.mean([c['dice'] for c in report['per_case']])
        st.caption(f'Mean validation Dice: **{mean_dice:.4f}**')

    st.markdown('---')

    st.subheader('Validation Case')
    available = demo.cases(selection)
    ids = [c['case_id'] for c in available]
    by_id = {c['case_id']: c for c in available}

    default_case = 'volume-covid19-A-0319'
    default_idx = ids.index(default_case) if default_case in ids else 0
    selected = st.selectbox(
        'Case',
        ids,
        index=default_idx,
        format_func=lambda x: (
            f"{x}  (Dice {by_id[x]['dice']:.3f})"
            if by_id[x].get('dice') is not None
            else x
        ),
        label_visibility='collapsed',
    )
    info = by_id[selected]
    st.caption(f'{len(ids)} cases available.')

    st.markdown('---')

    st.subheader('Display')
    threshold = st.slider('Prediction threshold', .05, .95, .5, .05)
    opacity = st.slider('Overlay opacity', .1, .9, .45, .05)
    show_gt = st.checkbox('Show ground truth overlay', value=True)

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------
history = load_history(selection.get('history'))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_viewer, tab_stats = st.tabs(['Segmentation Viewer', 'Training Statistics'])

# ============================= TAB 1: VIEWER ==============================
with tab_viewer:
    try:
        ct, gt, low, high = load_case(str(info['ct']), str(info['mask']))
    except Exception as error:
        st.error(str(error))
        st.stop()

    num_slices = len(ct)

    slice_key = f'slice_{selected}'
    if slice_key not in st.session_state:
        st.session_state[slice_key] = num_slices // 2

    def go_prev():
        if st.session_state[slice_key] > 0:
            st.session_state[slice_key] -= 1

    def go_next():
        if st.session_state[slice_key] < num_slices - 1:
            st.session_state[slice_key] += 1

    nav_left, nav_slider, nav_right = st.columns([1, 10, 1])
    with nav_left:
        st.button('<', on_click=go_prev, use_container_width=True, key='prev_btn')
    with nav_slider:
        master = st.slider(
            'Slice', 0, num_slices - 1,
            st.session_state[slice_key],
            key=slice_key,
            label_visibility='collapsed',
        )
    with nav_right:
        st.button('\\>', on_click=go_next, use_container_width=True, key='next_btn')

    st.caption(f'Slice {master + 1} / {num_slices}')

    # Auto-predict
    probability = cached_predict(selection['id'], selected, str(info['ct']), master)

    # Three columns
    left, middle, right = st.columns(3)

    with left:
        st.markdown('**CT Scan**')
        st.image(demo.render(ct[master], low=low, high=high), use_container_width=True)

    with middle:
        st.markdown('**Ground Truth**')
        if show_gt:
            st.image(
                demo.render(ct[master], gt[master], low=low, high=high,
                            alpha=opacity, color=(255, 75, 75)),
                use_container_width=True,
            )
        else:
            st.image(demo.render(ct[master], low=low, high=high), use_container_width=True)

    with right:
        st.markdown('**Model Prediction**')
        if probability is not None:
            pred_mask = probability >= threshold
            st.image(
                demo.render(ct[master], pred_mask, low=low, high=high,
                            alpha=opacity, color=(30, 200, 255)),
                use_container_width=True,
            )
        else:
            st.image(demo.render(ct[master], low=low, high=high), use_container_width=True)
            st.caption('Prediction unavailable (no GPU or model error).')

    if probability is not None:
        prediction = probability >= threshold
        truth = gt[master]
        intersection = int((prediction & truth).sum())
        den = int(prediction.sum() + truth.sum())
        dice_val = 2 * intersection / den if den else 1.0

        m1, m2, m3 = st.columns(3)
        m1.metric('Slice Dice', f'{dice_val:.3f}')
        m2.metric('True lesion pixels', f'{int(truth.sum()):,}')
        m3.metric('Predicted pixels', f'{int(prediction.sum()):,}')

        vol_dice = info.get('dice')
        if vol_dice is not None:
            st.caption(
                f'Metrics above are for the current slice only. '
                f'Full-volume Dice for this case: {vol_dice:.4f}. '
                f'Red = ground truth, blue = prediction.'
            )

# ========================== TAB 2: STATISTICS =============================
with tab_stats:
    st.header('Training Statistics')
    st.caption(f'Model: {selection["label"]} | Epochs completed: {len(history)}')

    if not history:
        st.warning('No training history available yet. History will appear here after each completed epoch.')
    else:
        epochs = [h['epoch'] for h in history]
        train_data = [h.get('train', {}) for h in history]

        # ==================================================================
        # SECTION 1: Training Loss
        # ==================================================================
        st.subheader('Training Loss')

        loss_total = [t.get('total') for t in train_data]
        loss_seg = [t.get('segmentation') for t in train_data]
        loss_presence = [t.get('presence') for t in train_data]
        loss_align = [t.get('alignment') for t in train_data]
        loss_salience = [t.get('salience') for t in train_data]

        loss_df = pd.DataFrame({
            'Epoch': epochs,
            'Total': loss_total,
            'Segmentation': loss_seg,
            'Presence': loss_presence,
            'Alignment': loss_align,
            'Salience': loss_salience,
        }).set_index('Epoch')

        # Show as metrics for single epoch, charts for multiple
        if len(history) == 1:
            row = history[0]
            t = row.get('train', {})
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('Total Loss', f'{t.get("total", 0):.4f}')
            c2.metric('Segmentation', f'{t.get("segmentation", 0):.4f}')
            c3.metric('Presence', f'{t.get("presence", 0):.4f}')
            c4.metric('Alignment', f'{t.get("alignment", 0):.4f}')
            c5.metric('Salience', f'{t.get("salience", 0):.4f}')
        else:
            st.line_chart(loss_df)

        # Training progress table
        st.markdown('**Training Progress**')
        progress_df = pd.DataFrame({
            'Epoch': epochs,
            'Optimizer Steps': [t.get('optimizer_steps') for t in train_data],
            'AMP Skipped': [t.get('amp_skipped_steps') for t in train_data],
            'Examples': [t.get('examples') for t in train_data],
            'Learning Rate': [h.get('learning_rate') for h in history],
        }).set_index('Epoch')
        st.dataframe(progress_df, use_container_width=True)

        st.markdown('---')

        # ==================================================================
        # SECTION 2: Validation Metrics
        # ==================================================================
        st.subheader('Validation Metrics')

        macro_rows = []
        for h in history:
            val = h.get('validation', {})
            per_case = val.get('per_case', [])
            if not per_case:
                continue
            dices = [c['dice'] for c in per_case if 'dice' in c]
            ious = [c['iou'] for c in per_case if 'iou' in c]
            precs = [c['precision'] for c in per_case if 'precision' in c]
            recs = [c['recall'] for c in per_case if 'recall' in c]
            hd95s = [c['hd95_mm'] for c in per_case
                     if c.get('hd95_mm') is not None and c.get('hd95_status') == 'defined']
            macro_rows.append({
                'Epoch': h['epoch'],
                'Mean Dice': np.mean(dices) if dices else None,
                'Mean IoU': np.mean(ious) if ious else None,
                'Mean Precision': np.mean(precs) if precs else None,
                'Mean Recall': np.mean(recs) if recs else None,
                'Mean HD95 (mm)': np.mean(hd95s) if hd95s else None,
                'Cases': len(per_case),
                'Best Dice': h.get('best_dice'),
            })

        if macro_rows:
            macro_df = pd.DataFrame(macro_rows).set_index('Epoch')

            if len(macro_rows) == 1:
                # Single epoch: show as metric cards
                r = macro_rows[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Mean Dice', f'{r["Mean Dice"]:.4f}' if r["Mean Dice"] else '—')
                c2.metric('Mean IoU', f'{r["Mean IoU"]:.4f}' if r["Mean IoU"] else '—')
                c3.metric('Mean Precision', f'{r["Mean Precision"]:.4f}' if r["Mean Precision"] else '—')
                c4.metric('Mean Recall', f'{r["Mean Recall"]:.4f}' if r["Mean Recall"] else '—')

                c5, c6 = st.columns(2)
                c5.metric('Mean HD95 (mm)', f'{r["Mean HD95 (mm)"]:.2f}' if r["Mean HD95 (mm)"] else '—')
                c6.metric('Validation Cases', r['Cases'])
            else:
                # Multiple epochs: line charts
                overlap_cols = ['Mean Dice', 'Mean IoU', 'Mean Precision', 'Mean Recall']
                st.markdown('**Segmentation Metrics Across Epochs**')
                st.line_chart(macro_df[overlap_cols])

                if macro_df['Mean HD95 (mm)'].notna().any():
                    st.markdown('**HD95 (mm) Across Epochs**')
                    st.line_chart(macro_df[['Mean HD95 (mm)']])

                if macro_df['Best Dice'].notna().any():
                    st.markdown('**Best Dice Over Training**')
                    st.line_chart(macro_df[['Best Dice']])

            # Epoch summary table
            st.markdown('**Epoch Summary**')
            st.dataframe(macro_df.style.format({
                'Mean Dice': '{:.4f}',
                'Mean IoU': '{:.4f}',
                'Mean Precision': '{:.4f}',
                'Mean Recall': '{:.4f}',
                'Mean HD95 (mm)': '{:.2f}',
                'Best Dice': '{:.4f}',
                'Cases': '{:.0f}',
            }, na_rep='—'), use_container_width=True)

            st.markdown('---')

            # ==============================================================
            # SECTION 3: Per-Case Detail
            # ==============================================================
            st.subheader('Per-Case Validation Detail')

            current_val = demo.model_report(selection)
            if current_val:
                per_case = current_val['per_case']
                case_rows = []
                for c in per_case:
                    case_rows.append({
                        'Case ID': c['case_id'],
                        'Dice': c.get('dice'),
                        'IoU': c.get('iou'),
                        'Precision': c.get('precision'),
                        'Recall': c.get('recall'),
                        'HD95 (mm)': c.get('hd95_mm'),
                        'GT Voxels': c.get('ground_truth_foreground_voxels'),
                        'Pred Voxels': c.get('prediction_foreground_voxels'),
                    })
                case_df = pd.DataFrame(case_rows).set_index('Case ID').sort_values('Dice', ascending=True)

                st.markdown('**Dice per Validation Case**')
                st.bar_chart(case_df[['Dice']])

                st.markdown('**All Case Metrics**')
                st.dataframe(case_df.style.format({
                    'Dice': '{:.4f}',
                    'IoU': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'HD95 (mm)': '{:.2f}',
                    'GT Voxels': '{:,.0f}',
                    'Pred Voxels': '{:,.0f}',
                }, na_rep='—'), use_container_width=True)
            else:
                st.info('No per-case validation data for this model.')
        else:
            st.info('No validation data in training history.')
