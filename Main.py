# app.py
import os
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.neural_network import MLPClassifier

# 🌐 환경 변수 불러오기
load_dotenv()
API_KEY = st.secrets["WEATHER_API_KEY"]
CITY = "Seoul"
URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

def get_weather():
    try:
        res = requests.get(URL)
        if res.status_code == 200:
            data = res.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind = data['wind']['speed']
            rain = data.get('rain', {}).get('1h', 0)
            return [temp, rain, wind, humidity]
        else:
            st.warning("실시간 날씨 불러오기 실패 - 기본값 사용")
            return [20, 0, 1, 50]
    except:
        st.warning("날씨 정보 가져오는 중 오류 발생 - 기본값 사용")
        return [20, 0, 1, 50]

# 🧠 훈련 데이터
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
y = [0, 1, 2, 3, 4, 2, 3, 4, 0]  # 0:반팔, 1:긴팔, 2:가디건, 3:패딩, 4:우비

# 모델 학습
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=1)
model.fit(X, y)

# 👕 옷 이름
clothes = ['반팔', '긴팔', '가디건', '패딩', '우비']

# 🎨 Streamlit UI
st.set_page_config(page_title="오늘 뭐 입지?", page_icon="🧥")
st.title("👕 오늘의 옷차림 추천")
st.write("실시간 날씨를 기반으로 인공지능이 적절한 옷차림을 추천합니다.")

if st.button("AI에게 추천받기"):
    features = get_weather()
    prediction = model.predict([features])[0]

    st.subheader("📍 현재 날씨 정보")
    st.write(f"🌡️ 온도: {features[0]}℃")
    st.write(f"🌧️ 강수량: {features[1]}mm")
    st.write(f"💨 바람: {features[2]}m/s")
    st.write(f"💧 습도: {features[3]}%")

    st.success(f"🤖 오늘은 **{clothes[prediction]}** 입는 걸 추천드려요!")
