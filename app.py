"""Demo chính: phát hiện clickbait đa phương thức (PhoBERT + ViT, Late Fusion).

Chạy: streamlit run app.py

- Mô hình: `concat` (Late Fusion) seed 42 (checkpoints/concat_seed42.pth). Bài không có ảnh
  được chấm bằng mô hình `text` seed 42 thay vì ghép ảnh đen giả.
- Số liệu ở sidebar và tab báo cáo đọc từ results/summary.csv (tập test độc lập,
  3 seed) và results/summary_v4.csv (20 seed độc lập), không nhập tay.
- Bản phân tích chi tiết hơn (so sánh 3 mô hình, ngưỡng, lịch sử): app_v2.py.
"""
import io
import os
import sys
import time

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from clickbait.explain import build_points, render_contributions, summary_sentence
from clickbait.inference import (DEMO_SEED, attention_heatmap, image_contribution, load_pipeline,
                                 predict_proba, word_importance)
from clickbait.scraper import fetch_article
from clickbait.signals import extract_signals, render_highlighted
from clickbait.ui_paste import paste_image

# Cấu hình UTF-8 cho hệ thống
if sys.platform == 'win32' and sys.stdout is not None:
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phát Hiện Clickbait Đa Phương Thức",
    page_icon="📰",
    layout="wide"
)

SUMMARY_PATH = os.path.join('results', 'summary.csv')
SUMMARY_V4_PATH = os.path.join('results', 'summary_v4.csv')
DEMO_MODE = 'concat'
ABLATION_8010_PATH = os.path.join('experiments', 'coattention_80_10_10', 'ablation_study_results.csv')
MODEL_NAMES = {
    'text': 'Text-only (PhoBERT)',
    'image': 'Image-only (ViT)',
    'concat': 'Late Fusion (Concat)',
    'multimodal': 'Multimodal Cross-Attention',
}

SAMPLES = {
    "Mẫu 1 (Clickbait giật tít): SỐC! Bật mí sự thật kinh hoàng đằng sau vẻ đẹp của hot girl vạn người mê khiến ai cũng ngã ngửa":
        "SỐC! Bật mí sự thật kinh hoàng đằng sau vẻ đẹp của hot girl vạn người mê khiến ai cũng ngã ngửa",
    "Mẫu 2 (Clickbait giật tít): Không thể tin nổi: Cách kiếm 50 triệu/tháng tại nhà chỉ với 15 phút mỗi ngày ai cũng giấu!":
        "Không thể tin nổi: Cách kiếm 50 triệu/tháng tại nhà chỉ với 15 phút mỗi ngày ai cũng giấu!",
    "Mẫu 3 (Tin chính thống): Sân bay Vinh đóng cửa 6 tháng để nâng cấp đường băng":
        "Sân bay Vinh đóng cửa 6 tháng để nâng cấp đường băng",
    "Mẫu 4 (Tin chính thống): Bộ Y tế khuyến cáo người dân các biện pháp phòng chống dịch sốt xuất huyết":
        "Bộ Y tế khuyến cáo người dân các biện pháp phòng chống dịch sốt xuất huyết",
}


# ==============================================================================
# HÀM BÓC TÁCH TIÊU ĐỀ VÀ ẢNH TỪ LINK BÁO THẬT (URL)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def extract_article_from_url(url):
    """Trả về (tiêu đề, ảnh bytes hoặc None, lỗi hoặc None). Cache theo URL để không tải lại mỗi lần rerun."""
    art = fetch_article(url)
    if 'error' in art:
        return "", None, art['error']
    return art['title'], art['image_bytes'], None


# ==============================================================================
# HÀM LOAD MODEL (CACHE ĐỂ TĂNG TỐC ĐỘ DEMO)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_components(mode):
    return load_pipeline(mode)


# ==============================================================================
# SỐ LIỆU THỰC NGHIỆM (đọc từ file kết quả, không nhập tay)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_summary(path=SUMMARY_PATH):
    """File summary CSV → DataFrame với cột mean/std dạng số. None nếu chưa có file."""
    if not os.path.exists(path):
        return None
    raw = pd.read_csv(path, encoding='utf-8-sig')
    out = pd.DataFrame({'mode': raw['mode'], 'n_seeds': raw['n_seeds']})
    for col in ['accuracy', 'macro_f1', 'f1_clickbait', 'precision_clickbait', 'recall_clickbait']:
        parts = raw[col].astype(str).str.split('±', expand=True)
        out[col] = parts[0].astype(float)
        out[col + '_std'] = parts[1].astype(float)
    return out.set_index('mode')


def fmt_pct(mean, std):
    return f"{mean*100:.2f}% ± {std*100:.2f}"


# ==============================================================================
# GIẢI THÍCH KẾT QUẢ
# ==============================================================================
def _point_expander(i, point):
    icon = '⚠️' if point.direction == 'clickbait' else '✅'
    with st.expander(f"{icon} Điểm {i}: «{point.text}» · {point.label}"):
        st.markdown(f"**Vì sao:** {point.reason}")
        if point.words:
            side = 'clickbait' if point.effect > 0 else 'tin chính thống'
            st.markdown(f"**Mức ảnh hưởng trong mô hình:** `{point.effect:+.2f}` (kéo về phía {side}; "
                        f"từ được tính: {', '.join(f'«{w}»' for w in point.words)})")
        elif point.from_rules:
            st.markdown("**Mức ảnh hưởng trong mô hình:** không tách được thành từ riêng "
                        "(ví dụ dấu câu dính liền với từ khác).")
        if point.from_rules and point.words and point.effect <= 0:
            st.caption("Bộ luật ngôn ngữ nhận ra dấu hiệu này, nhưng trong tiêu đề này mô hình "
                       "không coi nó là lý do kéo về clickbait.")
        if point.tip:
            st.markdown(f"**So với tin chính thống:** {point.tip}")


def show_explanation(pipeline, title, image, prob):
    """Phần "Vì sao": tóm tắt, tiêu đề tô màu theo mô hình, từng điểm bấm mở được, ảnh, bảng chi tiết."""
    st.divider()
    st.subheader("🔎 VÌ SAO MÔ HÌNH KẾT LUẬN NHƯ VẬY?")
    with st.spinner("Đang phân tích lý do (đo đóng góp của từng từ)..."):
        _, contributions = word_importance(pipeline, title, image)
        image_effect = image_contribution(pipeline, title, image)
    signals = extract_signals(title)
    cb_points, normal_points = build_points(contributions, signals)

    st.info(summary_sentence(prob, contributions, signals, image_effect))

    st.markdown("**Mỗi từ ảnh hưởng thế nào tới quyết định** (đỏ = kéo về clickbait, xanh = kéo về "
                "chính thống, màu càng đậm càng mạnh; rê chuột lên từ để xem số):")
    st.markdown(
        f'<div style="font-size:1.15rem; line-height:2.1; padding:0.6rem 0.9rem; '
        f'border:1px solid rgba(128,128,128,0.35); border-radius:8px;">'
        f'{render_contributions(contributions)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### ⚠️ Các điểm khiến tiêu đề giống clickbait")
    if cb_points:
        st.caption("Bấm vào từng điểm để xem giải thích chi tiết.")
        for i, point in enumerate(cb_points, 1):
            _point_expander(i, point)
    else:
        st.caption("Không có điểm nào đáng kể.")

    st.markdown("#### ✅ Các điểm khiến tiêu đề giống tin chính thống")
    if normal_points:
        for i, point in enumerate(normal_points, 1):
            _point_expander(i, point)
    else:
        st.caption("Không có điểm nào đáng kể.")

    if image is not None:
        with st.expander("🖼️ Ảnh thumbnail ảnh hưởng thế nào?"):
            if image_effect is not None:
                with_img, without = image_effect
                st.markdown(f"- Xác suất clickbait **với ảnh thật**: `{with_img*100:.1f}%`\n"
                            f"- Khi thay ảnh bằng **ảnh xám trung tính**: `{without*100:.1f}%`\n"
                            f"- Ảnh làm thay đổi: `{(with_img-without)*100:+.1f}` điểm phần trăm")
            heat = attention_heatmap(pipeline, image)
            if heat is not None:
                st.markdown("**Vùng ảnh mô hình tập trung** (Attention Rollout, vùng đỏ = chú ý nhiều):")
                h1, h2 = st.columns(2)
                h1.image(image.resize((224, 224)), caption="Ảnh gốc", width='stretch')
                h2.image(heat, caption="Attention heatmap", width='stretch')

    with st.expander("📊 Bảng đóng góp của tất cả các từ"):
        df = pd.DataFrame({'Từ': [w for w, _ in contributions], 'Đóng góp': [d for _, d in contributions]})
        df['Kéo về'] = ['clickbait' if d > 0 else 'chính thống' for d in df['Đóng góp']]
        st.dataframe(df, hide_index=True, width='stretch',
                     column_config={'Đóng góp': st.column_config.NumberColumn(format='%+.2f')})
        if signals:
            st.markdown("**Dấu hiệu theo bộ luật ngôn ngữ** (rê chuột lên cụm tô màu để xem lý do):")
            st.markdown(
                f'<div style="font-size:1.05rem; line-height:1.9; padding:0.5rem 0.8rem; '
                f'border:1px solid rgba(128,128,128,0.35); border-radius:8px;">'
                f'{render_highlighted(title, signals)}</div>',
                unsafe_allow_html=True,
            )

    with st.expander("ℹ️ Cách đọc phần giải thích"):
        st.markdown(
            "- **Đóng góp của từng từ** được đo trực tiếp trên mô hình: thêm dần các từ vào tiêu đề "
            "theo nhiều thứ tự ngẫu nhiên, ghi lại mỗi từ làm điểm clickbait (log-odds) tăng hay "
            "giảm bao nhiêu, rồi lấy trung bình (giá trị Shapley). Số dương kéo về clickbait, số âm "
            "kéo về chính thống.\n"
            "- **Dấu hiệu theo bộ luật** (khoảng trống tò mò, cảm thán, câu hỏi, ...) do một bộ luật "
            "ngôn ngữ riêng tìm ra để người đọc dễ hiểu; đó **không** phải lý do bên trong mô hình. "
            "Khi luật và mô hình không khớp, ô giải thích có ghi rõ.\n"
            "- **Ảnh hưởng của ảnh** được đo bằng cách thay ảnh thật bằng ảnh xám trung tính.\n"
            "- **Heatmap** cho biết vùng ảnh mà ViT chú ý, không cho biết vùng đó làm tăng hay giảm "
            "điểm clickbait.")


# ==============================================================================
# GIAO DIỆN CHÍNH STREAMLIT
# ==============================================================================
def main():
    st.title("📰 HỆ THỐNG PHÁT HIỆN CLICKBAIT ĐA PHƯƠNG THỨC TIẾNG VIỆT")
    st.markdown("""
    **Đề tài Khóa luận Cử nhân CNTT**: *Phát hiện clickbait đa phương thức bằng học sâu* (PhoBERT + ViT, Late Fusion)
    **Sinh viên thực hiện**: Nguyễn Hữu Thiện, Thi Lý Hữu Toàn, Trần Quốc Tuấn | **GVHD**: ThS. Trần Khải Thiện
    """)
    st.divider()

    with st.spinner("Đang khởi động hệ thống và nạp mô hình..."):
        mm = load_components(DEMO_MODE)
    device = mm['device']
    summary = load_summary()

    # Sidebar: Thông tin hệ thống & kết quả thực nghiệm
    with st.sidebar:
        st.header("⚙️ Thông Tin Mô Hình")
        st.write(f"**Thiết bị chạy:** `{device.type.upper()}`")
        if device.type == 'cuda':
            st.write(f"**Card đồ họa:** `{torch.cuda.get_device_name(0)}`")
        st.write("**Văn bản:** `PhoBERT-base` (tách từ pyvi)")
        st.write("**Hình ảnh:** `Vision Transformer (ViT)`")
        st.write("**Cơ chế kết hợp:** `Late Fusion (ghép đặc trưng)`")
        if mm['loaded']:
            st.caption(f"Checkpoint: `{mm['checkpoint']}`")
        else:
            st.error(f"Không tìm thấy `{mm['checkpoint']}`: mô hình đang mang trọng số ngẫu nhiên, kết quả không có ý nghĩa.")
        st.divider()

        st.header("📊 Kết Quả Trên Tập Test")
        if summary is not None and DEMO_MODE in summary.index:
            r = summary.loc[DEMO_MODE]
            st.metric(label="🎯 Accuracy", value=f"{r['accuracy']*100:.2f}%", delta=f"± {r['accuracy_std']*100:.2f}", delta_color="off")
            st.metric(label="⭐ Macro F1-Score", value=f"{r['macro_f1']*100:.2f}%", delta=f"± {r['macro_f1_std']*100:.2f}", delta_color="off")
            st.caption(f"Trung bình ± độ lệch chuẩn qua {int(r['n_seeds'])} seed, tập test độc lập (15% dữ liệu, tách trước khi huấn luyện).")
        else:
            st.info("Chưa có results/summary.csv. Chạy `python -m scripts.summarize`.")
        if 'latency' in st.session_state:
            st.metric(label="⚡ Tốc độ phản hồi (lần gần nhất)", value=f"{st.session_state['latency']:.3f}s")

    # Tabs chức năng
    tab1, tab2 = st.tabs(["🔍 Dự Đoán Trực Tiếp", "📈 Báo Cáo Nghiên Cứu & Ablation Study"])

    with tab1:
        # Chọn phương thức nhập dữ liệu
        input_mode = st.radio(
            "Chọn phương thức kiểm định:",
            [
                "🌐 Cách 1: Dán link bài báo thật (URL báo mạng tự động cào ảnh + tiêu đề)",
                "✍️ Cách 2: Tự nhập tiêu đề & Tải ảnh lên từ máy tính",
                "🎯 Cách 3: Chọn bài báo mẫu có sẵn để thử nhanh"
            ],
            index=0
        )

        title_input = ""
        pil_image = None

        col1, col2 = st.columns([1.2, 1])

        # --- CÁCH 1: DÁN LINK URL BÁO THẬT ---
        if "Cách 1" in input_mode:
            with col1:
                st.markdown("##### Dán link bài báo từ các trang tin (VnExpress, Dân Trí, Tuổi Trẻ, Kênh 14, Báo Mới...):")
                article_url = st.text_input(
                    "Đường dẫn bài báo (URL):",
                    placeholder="https://vnexpress.net/san-bay-dien-bien-san-sang-don-chuyen-bay-thuong-mai-4683515.html"
                )
                if article_url.strip():
                    with st.spinner("Đang tự động bóc tách tiêu đề và ảnh thumbnail từ trang báo..."):
                        t, img_bytes, err = extract_article_from_url(article_url.strip())
                    if err:
                        st.warning(f"Không thể cào tự động trang này: {err} Bạn có thể nhập tay tiêu đề bên dưới.")
                    else:
                        title_input = t
                        if img_bytes:
                            try:
                                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            except Exception:
                                st.warning("Lấy được tiêu đề nhưng ảnh bìa bị lỗi định dạng.")

                title_input = st.text_area("Tiêu đề trích xuất được:", value=title_input, height=100)
                if article_url.strip() and pil_image is None:
                    st.caption("Không lấy được ảnh bìa? Copy ảnh thumbnail trên trang báo rồi dán vào đây:")
                    pil_image = paste_image(key='paste_url')

        # --- CÁCH 2: TỰ NHẬP TIÊU ĐỀ & ẢNH ---
        elif "Cách 2" in input_mode:
            with col1:
                title_input = st.text_area("Nhập tiêu đề bài báo:", height=100, placeholder="Nhập tiêu đề bài báo vào đây...")
                uploaded_file = st.file_uploader("Tải ảnh thumbnail lên (tùy chọn):", type=['jpg', 'jpeg', 'png', 'webp'])
                st.markdown("hoặc")
                pasted = paste_image(key='paste_manual')
                if pasted is not None:
                    # Ảnh dán được ưu tiên hơn ảnh tải lên (xóa ảnh dán để dùng lại ảnh tải lên)
                    pil_image = pasted
                elif uploaded_file is not None:
                    try:
                        pil_image = Image.open(uploaded_file).convert("RGB")
                    except Exception as e:
                        st.error(f"Lỗi file ảnh: {e}")

        # --- CÁCH 3: MẪU CÓ SẴN ---
        else:
            with col1:
                sample_choice = st.selectbox("Chọn bài báo mẫu thử nghiệm:", list(SAMPLES))
                title_input = SAMPLES[sample_choice]
                st.info("Các mẫu này chỉ có tiêu đề nên được chấm bằng mô hình Text-only.")

        # Hiển thị ảnh bên cột phải
        with col2:
            st.markdown("**Ảnh Thumbnail của bài báo:**")
            if pil_image is not None:
                st.image(pil_image, caption="Thumbnail thực tế", width='stretch')
            else:
                st.info("Chưa có ảnh. Bài không có ảnh sẽ được chấm bằng mô hình Text-only (PhoBERT), "
                        "không ghép ảnh giả để tránh sai lệch.")

        # Nút bấm dự đoán
        predict_btn = st.button("🚀 PHÂN TÍCH & DỰ ĐOÁN CLICKBAIT", type="primary", width='stretch')

        if predict_btn:
            if not title_input.strip():
                st.warning("Vui lòng nhập hoặc cung cấp tiêu đề bài báo!")
                return

            if pil_image is not None:
                pipeline, used = mm, "Late Fusion (PhoBERT + ViT)"
            else:
                with st.spinner("Đang nạp mô hình Text-only..."):
                    pipeline, used = load_components('text'), "Text-only (PhoBERT), vì bài không có ảnh"

            start_t = time.time()
            prob = predict_proba(pipeline, title_input.strip(), pil_image)
            latency = time.time() - start_t
            st.session_state['latency'] = latency

            is_clickbait = prob >= 0.5
            confidence = prob if is_clickbait else (1.0 - prob)

            st.divider()
            st.subheader("🎯 KẾT QUẢ ĐÁNH GIÁ TỪ MÔ HÌNH HỌC SÂU:")
            st.caption(f"Mô hình sử dụng: **{used}** · checkpoint seed {DEMO_SEED}")

            res_col1, res_col2 = st.columns([1.5, 1])
            with res_col1:
                if is_clickbait:
                    st.error("⚠️ **KẾT LUẬN: BÀI BÁO NÀY LÀ CLICKBAIT (GIẬT TÍT)**")
                else:
                    st.success("✅ **KẾT LUẬN: BÀI BÁO CHÍNH THỐNG (NON-CLICKBAIT)**")
                st.markdown(f"- **Độ tin cậy dự đoán:** `{confidence*100:.2f}%`")
                st.progress(float(prob), text=f"Xác suất clickbait = {prob*100:.2f}% (ngưỡng 50%)")
                if 0.4 <= prob <= 0.6:
                    st.warning("Xác suất nằm sát ngưỡng 50%: mô hình không chắc chắn với bài này.")

            with res_col2:
                st.metric("Tốc độ xử lý", f"{latency:.4f} giây")
                st.metric("Xác suất Clickbait", f"{prob*100:.2f}%")

            show_explanation(pipeline, title_input.strip(), pil_image, prob)

    with tab2:
        st.subheader("Bảng Đối Sánh Toàn Diện (Ablation Study)")
        if summary is None:
            st.info("Chưa tìm thấy results/summary.csv. Chạy `python -m scripts.summarize`.")
        else:
            n = int(summary['n_seeds'].iloc[0])
            st.markdown(f"""
            Kết quả trên **tập test độc lập** (15% dữ liệu, 480 mẫu, tách phân tầng *trước* khi huấn luyện),
            trung bình ± độ lệch chuẩn qua **{n} seed**. Checkpoint được chọn theo Macro-F1 trên tập validation.
            """)
            order = [m for m in ['text', 'image', 'concat', 'multimodal'] if m in summary.index]
            table = pd.DataFrame({
                'Mô hình': [MODEL_NAMES[m] for m in order],
                'Accuracy': [fmt_pct(summary.loc[m, 'accuracy'], summary.loc[m, 'accuracy_std']) for m in order],
                'Macro F1': [fmt_pct(summary.loc[m, 'macro_f1'], summary.loc[m, 'macro_f1_std']) for m in order],
                'F1 (clickbait)': [fmt_pct(summary.loc[m, 'f1_clickbait'], summary.loc[m, 'f1_clickbait_std']) for m in order],
            })
            st.dataframe(table, hide_index=True, width='stretch')

            f1 = summary['macro_f1']
            st.markdown(f"""
            ### 📌 Nhận xét
            1. **Văn bản là nguồn thông tin chính:** Text-only đạt Macro F1 = {f1['text']*100:.2f}%, còn Image-only chỉ {f1['image']*100:.2f}%.
               Ảnh đứng riêng không đủ để nhận biết clickbait.
            2. **Cross-Attention và Late Fusion:** {f1['multimodal']*100:.2f}% so với {f1['concat']*100:.2f}%. Chênh lệch nhỏ hơn
               độ lệch chuẩn giữa các seed (khoảng {summary.loc[['concat', 'multimodal'], 'macro_f1_std'].max()*100:.1f} điểm),
               nên với cỡ dữ liệu hiện tại chưa khẳng định được cách kết hợp nào tốt hơn.
            """)

        v4 = load_summary(SUMMARY_V4_PATH)
        if v4 is not None:
            st.subheader("Lặp lại độc lập trên 20 seed mới (50–69)")
            st.markdown(f"""
            Quy tắc kiểm định chốt trước khi chạy (docs/protocol-v4.md): Wilcoxon hai phía, hiệu chỉnh Holm–Bonferroni.

            - **Late Fusion so với Image-only:** {v4.loc['concat', 'macro_f1']*100:.2f}% so với {v4.loc['image', 'macro_f1']*100:.2f}%,
              cao hơn ở cả 20 seed, **có ý nghĩa thống kê**.
            - **Late Fusion so với Text-only:** {v4.loc['concat', 'macro_f1']*100:.2f}% so với {v4.loc['text', 'macro_f1']*100:.2f}%,
              **không khác nhau có ý nghĩa** (p = 0,20). Trên bộ dữ liệu này, thêm ảnh không cải thiện được so với chỉ dùng tiêu đề.
            """)

        if os.path.exists(ABLATION_8010_PATH):
            with st.expander("Thử nghiệm phụ: Co-Attention hai chiều, chia 80/10/10 (1 seed)"):
                st.dataframe(pd.read_csv(ABLATION_8010_PATH), hide_index=True, width='stretch')
                st.caption("Khác cách chia tập (80/10/10), khác tập mẫu (ảnh lỗi thay bằng ảnh đen) và không tách từ, "
                           "nên không so sánh trực tiếp với bảng chính. Chi tiết: experiments/coattention_80_10_10/README.md.")


if __name__ == '__main__':
    main()
