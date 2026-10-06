# Language Translation Tool

A web app that translates text between languages in real time, built with Python and Streamlit.

## Features
- Translate text between major world languages
- Auto-detect the input language [keep only if your code does this]
- Simple browser interface with Streamlit

## Tech Stack
- Python 3
- Streamlit (UI)
- googletrans (unofficial Google Translate wrapper)

## How It Works
1. The user enters text and selects a target language.
2. The app sends it to Google Translate through googletrans.
3. The translated text is shown instantly.

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Results
Text was translated in real time between multiple languages through the web interface.

## Future Improvements
- Voice input/output
- Official Google Cloud Translation API
- Translation history
