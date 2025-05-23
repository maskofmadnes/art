from web.dashboard import Dashboard
from web.sidebar import Sidebar
import streamlit as st


st.set_page_config(
    page_title="ART",
    page_icon=":musical_note:",
    layout="wide",
)

Sidebar().render()
Dashboard().render()
