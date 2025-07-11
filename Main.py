import os
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 📦 환경변수 불러오기 (.env 또는 Streamlit secrets)
load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY") or st.secrets.get("WEATHER_API_KEY")

CITY = "Seoul"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

def get_weather():
    """OpenWeatherMap에서 실시간 날씨 데이터 가져오기"""
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind = data['wind']['speed']
            rain = data.get('rain', {}).get('1h', 0)
            return temp, rain, wind, humidity
        else:
            st.warning(f"API 연결 실패 (status: {response.status_code}) - 기본값 사용")
            return 20, 0, 1, 50
    except Exception as e:
        st.warning("날씨 정보를 가져오는 데 실패했습니다. 기본값 사용")
        return 20, 0, 1, 50

# 🧠 학습 데이터 및 모델 정의
X = np.array([
    [30, 0, 1, 40],   # 반팔
    [22, 0, 2, 50],   # 긴팔
    [18, 0, 1, 55],   # 가디건
    [5, 0, 3, 30],    # 패딩
    [20, 10, 2, 70],  # 우비
    [15, 2, 1.5, 60], # 가디건
    [10, 0, 2, 50],   # 패딩
    [25, 5, 1, 65],   # 우비
    [28, 0, 0.5, 30], # 반팔
])
y = np.array([0, 1, 2, 3, 4, 2, 3, 4, 0])
y_encoded = to_categorical(y, num_classes=5)

model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y_encoded, epochs=200, verbose=0)

# 👕 옷 이름 매핑
clothes = ['반팔', '긴팔', '가디건', '패딩', '우비']

# 🌐 Streamlit UI
st.set_page_config(page_title="오늘 뭐 입지? AI 추천", page_icon="👕")
st.title("👕 오늘의 옷 추천 AI")
st.write("실시간 날씨 데이터를 기반으로 인공지능이 적절한 옷차림을 추천합니다.")

if st.button("날씨 기반 옷 추천 받기"):
    temp, rain, wind, humidity = get_weather()
    input_data = np.array([[temp, rain, wind, humidity]])
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    # 결과 출력
    st.subheader("📍 현재 날씨 정보")
    st.metric("기온 (℃)", f"{temp:.1f}")
    st.metric("강수량 (mm)", f"{rain}")
    st.metric("바람 속도 (m/s)", f"{wind}")
    st.metric("습도 (%)", f"{humidity}")

    st.subheader("🤖 AI 추천 결과")
    st.success(f"오늘의 추천 옷차림은: **{clothes[predicted_class]}** 입니다!")

    st.caption("※ 이 추천은 간단한 신경망 모델 기반이며 참고용입니다.")
