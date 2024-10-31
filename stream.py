import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# TensorFlow 경고 메시지 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# DepthwiseConv2D 레이어 사용자 정의
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, kernel_size, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(kernel_size, **kwargs)

# 사용자 정의 객체 등록
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# 페이지 기본 설정
st.set_page_config(
    page_title="이미지 분류 앱",
    page_icon="🔍",
    layout="wide"
)

# CSS 스타일 적용
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4CAF50, #8BC34A);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .upload-box {
        border: 2px dashed #ccc;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
    }
    .css-1v0mbdj.e115fcil1 {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 타이틀과 설명
st.title("🔍 이미지 분류기")
st.markdown("---")

# 사이드바에 정보 표시
with st.sidebar:
    st.header("ℹ️ 앱 정보")
    st.info("""
    이 앱은 Teachable Machine으로 학습된 모델을 사용하여 
    이미지를 분류합니다.
    
    1. 이미지를 업로드하세요
    2. 자동으로 분류가 시작됩니다
    3. 각 클래스별 확률을 확인하세요
    """)

# 모델 로드 함수
@st.cache_resource
def load_classification_model():
    # 모델 로드 시 compile=False 옵션 사용
    model = load_model("keras_Model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    class_names = [line.strip() for line in open("labels.txt", "r", encoding="utf-8").readlines()]
    return model, class_names

try:
    model, class_names = load_classification_model()
    
    # 이미지 전처리 및 예측 함수
    def process_image(image):
        # 이미지 리사이징
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        
        # 이미지를 numpy 배열로 변환
        image_array = np.asarray(image)
        
        # 정규화
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # 모델 입력 shape에 맞게 데이터 준비
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        return data

    # 메인 컬럼 설정
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="업로드된 이미지", use_column_width=True)

    with col2:
        if uploaded_file is not None:
            # 예측 수행
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.subheader("📊 분류 결과")
            
            # 이미지 처리 및 예측
            with st.spinner('분석 중...'):
                processed_data = process_image(image)
                prediction = model.predict(processed_data, verbose=0)
            
            # 결과 정렬
            results = []
            for i, score in enumerate(prediction[0]):
                class_name = class_names[i][2:] if class_names[i].startswith("0 ") else class_names[i]
                results.append((class_name, float(score)))
            
            # 확률 높은 순으로 정렬
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 결과 표시
            for class_name, confidence_score in results:
                percentage = confidence_score * 100
                st.write(f"**{class_name}**")
                st.progress(confidence_score)
                st.write(f"{percentage:.1f}%")
                st.write("---")
            
            st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"""
    ⚠️ 에러가 발생했습니다!
    
    다음 사항을 확인해주세요:
    1. keras_Model.h5 파일이 존재하는지 확인
    2. labels.txt 파일이 존재하는지 확인
    3. 파일들이 올바른 형식인지 확인
    
    에러 메시지: {str(e)}
    """)