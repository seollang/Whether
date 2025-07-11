import streamlit as st
import requests
import numpy as np
from sklearn.linear_model import LogisticRegression

# 📌 도시 선택
city = st.selectbox("📍 도시를 선택하세요", ["Seoul", "Busan", "Incheon", "Jeju", "Daegu"])

# 🔐 API 키 (secrets.toml에 저장)
API_KEY = "0500cef5a15ad8fc7aa30afaa0dc6e84"

# 🌤️ 날씨 정보 가져오기 함수
def get_weather(city_name):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            wind = data['wind']['speed']
            rain = data.get('rain', {}).get('1h', 0)
            return [temp, rain, wind, humidity], None
        else:
            return [20, 0, 1, 50], f"❌ 날씨 정보를 가져오는 데 실패했습니다. 상태코드: {response.status_code}"
    except Exception as e:
        return [20, 0, 1, 50], f"❌ 오류 발생: {e}"

# 🧠 간단한 학습용 데이터 (임의)
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
y = np.array([0, 1, 2, 3, 4, 2, 3, 4, 0])  # 라벨: 0~4

# 👕 라벨 → 옷 종류
clothes = ['반팔', '긴팔', '가디건', '패딩', '우비']

# 모델 학습
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# 🌡️ 날씨 데이터 예측
weather_data, error = get_weather(city)

if error:
    st.error(error)
else:
    temp, rain, wind, humidity = weather_data
    st.subheader(f"📡 {city}의 현재 날씨")
    st.write(f"🌡️ 기온: {temp}℃")
    st.write(f"🌧️ 강수량: {rain}mm")
    st.write(f"💨 바람 속도: {wind}m/s")
    st.write(f"💧 습도: {humidity}%")

    # 예측
    predicted = model.predict([weather_data])[0]
    st.success(f"👕 오늘의 추천 옷: **{clothes[predicted]}**")
