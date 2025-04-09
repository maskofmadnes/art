import json
import streamlit as st


@st.cache_data
def load():
    with open("art/web/translation.json", "r", encoding="utf-8") as f:
        return json.load(f)


def translation(lang: str) -> dict:
    translations = load()
    return translations.get(lang, translations["en"])
