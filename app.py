
import json
import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ==========================================
# 1. CẤU HÌNH ỨNG DỤNG
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "checkpoints" / "resnet18_v2_best.pth"
CLASSES_PATH = BASE_DIR / "artifacts" / "resnet18_v2_classes.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
)


# ==========================================
# 2. GIAO DIỆN CSS
# ==========================================

st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #16a34a;
        text-align: center;
    }

    .sub-title {
        text-align: center;
        color: #64748b;
        margin-bottom: 30px;
    }

    .result-card {
        background-color: rgba(22, 163, 74, 0.08);
        border: 1px solid #16a34a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .result-label {
        font-size: 28px;
        font-weight: bold;
        color: #16a34a;
    }

    .result-confidence {
        font-size: 36px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {MODEL_PATH}"
        )

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy classes: {CLASSES_PATH}"
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model, classes


# ==========================================
# 4. TIỀN XỬ LÝ ẢNH
# ==========================================

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==========================================
# 5. HÀM DỰ ĐOÁN
# ==========================================

def predict_image(image, model, classes):

    image = image.convert("RGB")

    input_tensor = preprocess(image)

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(3, len(classes)),
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start_time

    results = []

    for probability, index in zip(
        top_probs.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        results.append({
            "class": classes[index],
            "confidence": probability,
        })

    return results, inference_time


# ==========================================
# 6. HEADER
# ==========================================

st.markdown(
    '<div class="main-title">♻️ AI WASTE CLASSIFIER</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Hệ thống phân loại rác sử dụng Deep Learning - ResNet18'
    '</div>',
    unsafe_allow_html=True,
)


# ==========================================
# 7. THÔNG TIN MODEL
# ==========================================

try:

    model, classes = load_model()

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "ResNet18")
    col2.metric("Số lớp", len(classes))
    col3.metric(
        "Thiết bị",
        "GPU CUDA" if DEVICE.type == "cuda" else "CPU",
    )

    st.success("Model đã được tải thành công!")

except Exception as error:

    st.error(f"Lỗi khi tải model: {error}")
    st.stop()


# ==========================================
# 8. UPLOAD ẢNH
# ==========================================

st.divider()

st.subheader("📤 Upload ảnh rác cần phân loại")

uploaded_file = st.file_uploader(
    "Chọn một ảnh từ máy tính",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        # Hiển thị ảnh đầu vào
        with left:

            st.subheader("Ảnh đầu vào")

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            st.caption(
                f"Kích thước ảnh: {image.width} x {image.height}"
            )

        # ==================================
        # 9. PREDICTION
        # ==================================

        with right:

            st.subheader("Kết quả phân loại")

            with st.spinner("AI đang phân tích ảnh..."):

                results, inference_time = predict_image(
                    image,
                    model,
                    classes,
                )

            best_result = results[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <div>Loại rác dự đoán</div>
                    <div class="result-label">
                        {best_result['class'].upper()}
                    </div>
                    <div class="result-confidence">
                        {best_result['confidence']:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            st.metric(
                "Thời gian inference",
                f"{inference_time * 1000:.2f} ms",
            )

        # ==================================
        # 10. TOP 3 PREDICTIONS
        # ==================================

        st.divider()

        st.subheader("📊 Top 3 kết quả dự đoán")

        for index, result in enumerate(results):

            col_name, col_confidence = st.columns([3, 1])

            with col_name:

                st.write(
                    f"**{index + 1}. {result['class'].upper()}**"
                )

                st.progress(result["confidence"])

            with col_confidence:

                st.write(
                    f"**{result['confidence']:.2%}**"
                )

        st.info(
            "Confidence là xác suất đầu ra sau Softmax, "
            "không đảm bảo mô hình luôn dự đoán đúng."
        )

    except Exception as error:

        st.error(f"Không thể xử lý ảnh: {error}")

else:

    st.info(
        "Hãy upload một ảnh rác để bắt đầu kiểm thử mô hình."
    )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Waste Classification Project | ResNet18 | PyTorch"
)
