from streamlit.runtime.state import SessionState
from web.translation import translation
from web.utils import extract_cover
import tempo
import streamlit as st
import numpy as np
import plotly.express as px


class Dashboard:
    def __init__(self):
        self.td = translation(st.session_state.get("language", "en"))

    def t(self, key):
        return f"{self.td[key]}"

    def render(self):
        st.title(self.t("title"))
        st.subheader(self.t("subtitle"))
        if st.session_state.get("upload", None) is not None:
            audio = tempo.audio(st.session_state.upload)
            dynamic_tempo = tempo.dynamic_tempo(
                audio,
                hop_length=st.session_state.hop_length,
                start_bpm=st.session_state.start_bpm,
                std_bpm=st.session_state.standard_bpm,
                ac_size=st.session_state.ac_size,
                max_tempo=st.session_state.max_bpm,
            )
            time_tempo = tempo.time_tempo(
                audio,
                dynamic_tempo=dynamic_tempo,
                hop_length=st.session_state.hop_length
            )
            segments = tempo.segments(time_tempo, dynamic_tempo)
            static_bpm, dynamic_times = tempo.dynamic_times(
                audio,
                dynamic_tempo,
                hop_length=st.session_state.hop_length,
                start_bpm=st.session_state.start_bpm,
                tightness=st.session_state.tightness,
                trim=st.session_state.trim
            )
            dynamic_clicks = tempo.dynamic_clicks(
                audio,
                dynamic_times,
                hop_length=st.session_state.hop_length,
                click_freq=st.session_state.click_freq,
                click_duration=st.session_state.click_duration
            )
            music_y, music_sr = tempo.music(
                audio,
                dynamic_clicks,
                volume=st.session_state.volume,
                click_freq=st.session_state.click_freq,
                click_duration=st.session_state.click_duration
            )
        else:
            with st.container(border=True):
                st.file_uploader(
                    self.t("choose_file"),
                    type=["wav", "mp3", "flac", "ogg", "m4a", "wma", "aiff", "aif"],
                    key="upload"
                )
            return
        with st.container(border=True):
            st.file_uploader(
                self.t("choose_file"),
                type=["wav", "mp3", "flac", "ogg", "m4a", "wma", "aiff", "aif"],
                key="upload"
            )
            st.audio(music_y, sample_rate=music_sr)
        classic, neural_network, general = st.tabs([self.t("classic"),self.t("neural_network"), self.t("overview")])
        with classic:
            with st.container(border=True):
                x, y = time_tempo, dynamic_tempo
                data = {
                    "x": x,
                    "y": y,
                }
                fig = px.line(
                    data,
                    x="x",
                    y="y",
                    title=self.t("bpm_dynamic"),
                    labels={"x": self.t("time") + " (s)", "y": "BPM"},
                )
                fig.update_traces(line=dict(color="#d85791"))
                st.plotly_chart(fig)
            data = {self.t("start"): [], self.t("end"): [], "BPM": []}
            for start, end, bpm in segments:
                data[self.t("start")] += [f"{start:.3f}".replace(".", ",")]
                data[self.t("end")] += [f"{end:.3f}".replace(".", ",")]
                data["BPM"] += [str(round(bpm, 2))]
            col_average, col_onset = st.columns(2, border=True)
            with col_average:
                st.write(f"{self.t("average")} BPM: {round(np.mean(static_bpm), 2)}")
            with col_onset:
                st.write(f"{self.t("first_onset")}: {round(dynamic_times[0], 3)}")
            st.table(data)
        with general:
            # with st.container(border=True):
            col_info, col_image = st.columns(2, border=True)
            with col_info:
                std_dev = np.std(static_bpm)
                if std_dev < 2:
                    rhythmic_variance = self.t("low")
                elif std_dev < 10:
                    rhythmic_variance = self.t("moderate")
                else:
                    rhythmic_variance = self.t("high")
                duration_seconds = round(segments[-1][1], 3)
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                formatted_duration = f"{minutes}:{seconds:02}"
                min_bpm = round(np.min(static_bpm), 2)
                max_bpm = round(np.max(static_bpm), 2)
                st.write(f"{self.t("average")} BPM: {round(np.mean(static_bpm), 2)}")
                st.divider()
                st.write(f"{self.t("bpm_range")}: {min_bpm} -> {max_bpm}")
                st.divider()
                st.write(f"{self.t("rhythmic_variance")}: {rhythmic_variance}")
                st.divider()
                st.write(f"{self.t("duration")}: {formatted_duration}")
            with col_image:
                cover_data = extract_cover(st.session_state.upload)
                if cover_data:
                    st.image(cover_data)
                else:
                    st.info(self.t("no_track_cover"))
        neural_network.write("In development")
