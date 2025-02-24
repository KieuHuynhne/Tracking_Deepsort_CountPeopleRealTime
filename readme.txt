Kiều Ngọc Như Huỳnh - 52100224

Các bước Demo cho hệ điều hành Window:
1. Tạo folder và tạo máy ảo venv
>>> python -m venv myvenv
2. Activate
>>> myvenv\Scripts\activate  
3. Tải source về folder đó và chuyển đến folder source
>> cd source
4. Tải các packages
>>> pip install streamlit
>>> pip install -e '.[dev]'
5. Run app
>>> streamlit run app.py