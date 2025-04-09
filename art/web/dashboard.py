from streamlit.runtime.state import SessionState
from web.translation import translation
from web.utils import extract_cover
import tempo
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression


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
            dynamic_bpm, onset_times = tempo.onset_times(
                audio,
                dynamic_tempo,
                hop_length=st.session_state.hop_length,
                start_bpm=st.session_state.start_bpm,
                tightness=st.session_state.tightness,
                trim=st.session_state.trim
            )
            time_tempo = tempo.time_tempo(
                audio,
                dynamic_tempo=dynamic_bpm,
                hop_length=st.session_state.hop_length
            )
            onset_bpm = tempo.onset_bpm(dynamic_bpm, onset_times, time_tempo)
            intervals = tempo.intervals(onset_bpm, onset_times)
            dynamic_clicks = tempo.dynamic_clicks(
                audio,
                onset_times,
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
        twod_plot, threed_plot, nn, intervals_tab, onset_and_bpm, general = st.tabs([
            self.t("twod_plot"),
            self.t("threed_plot"),
            self.t("nn"), 
            self.t("intervals"),
            self.t("onset_and_bpm"),
            self.t("overview")])
        with twod_plot:
            col_average, col_onset = st.columns(2, border=True)
            with col_average:
                st.write(f"{self.t("average")} BPM: {round(np.mean(dynamic_bpm), 2)}")
            with col_onset:
                st.write(f"{self.t("first_onset")}: {str(round(onset_times[0], 3)).replace(".", ",")}")
            with st.container(border=True):
                x, y = onset_times, onset_bpm
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
                # fig.update_traces(line=dict(color="#d85791"))
                st.plotly_chart(fig)
                
            time_diffs = np.diff(onset_times)
            with st.container(border=True):
                data = {
                    "x": onset_times[1:],  # Начинаем с первого элемента (пропускаем 0)
                    "y": time_diffs,
                }
                fig = px.line(
                    data,
                    x="x",
                    y="y",
                    title="time_intervals_between_onsets",
                    labels={"x": self.t("time") + " (s)", "y": "intervals_between_onsets" + " (s)"},
                )
                st.plotly_chart(fig) 
        with threed_plot:
            with st.container(border=True):
                x = np.array(onset_times)
                y = np.array(onset_bpm)
                z = np.diff(x, prepend=x[0])  # разница между onset_times
            
                fig = go.Figure(data=[go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='lines+markers',
                    marker=dict(size=4, color=z, colorscale='Viridis'),
                    line=dict(color='royalblue', width=2)
                )])
            
                fig.update_layout(
                    scene=dict(
                        xaxis_title=self.t("time") + " (s)",
                        yaxis_title="BPM",
                        zaxis_title=self.t("intervals") + " (s)"
                    ),
                    title="bpm dynamic 3d", #self.t("bpm_dynamic_3d"),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                st.plotly_chart(fig)
        with intervals_tab:
            data = {self.t("start"): [], self.t("end"): [], "BPM": []}
            for start, end, bpm in intervals:
                data[self.t("start")] += [f"{start:.3f}".replace(".", ",")]
                data[self.t("end")] += [f"{end:.3f}".replace(".", ",")]
                data["BPM"] += [str(round(bpm, 2))]
            st.table(data)
        with onset_and_bpm:
            onset_bpm_table = {"Onset (s)": [], "BPM": []}
            for time, bpm in zip(onset_times, onset_bpm):
                onset_bpm_table["Onset (s)"] += [str(round(time, 3)).replace(".", ",")]
                onset_bpm_table["BPM"] += [str(round(bpm, 2))]
            st.table(onset_bpm_table)
        with general:
            col_info, col_image = st.columns(2, border=True)
            with col_info:
                std_dev = np.std(dynamic_bpm)
                if std_dev < 2:
                    rhythmic_variance = self.t("low")
                elif std_dev < 10:
                    rhythmic_variance = self.t("moderate")
                else:
                    rhythmic_variance = self.t("high")
                min_bpm = round(np.min(dynamic_bpm), 2)
                max_bpm = round(np.max(dynamic_bpm), 2)
                score = complexity_score(dynamic_bpm, intervals[-1][1])
                description = interpret_score(score)
                st.metric("Complexity Score", score)
                st.write(description)
                st.divider()
                st.write(f"{self.t("average")} BPM: {round(np.mean(dynamic_bpm), 2)}")
                st.divider()
                st.write(f"{self.t("bpm_range")}: {min_bpm} -> {max_bpm}")
                st.divider()
                st.write(f"{self.t("rhythmic_variance")}: {rhythmic_variance}")
            with col_image:
                cover_data = extract_cover(st.session_state.upload)
                if cover_data:
                    st.image(cover_data)
                else:
                    st.info(self.t("no_track_cover"))
        with nn:
            duration_sec = intervals[-1][1]
            bpm_values = np.array(dynamic_bpm)
            std_bpm = np.std(bpm_values)
            # tempo_changes = len(segments)
            tempo_changes = np.sum(np.abs(np.diff(bpm_values)) > 3)
            change_rate = tempo_changes / duration_sec
            jitter = np.sum(np.diff(np.sign(np.diff(bpm_values))) != 0) / len(bpm_values)
            bpm_range = np.max(bpm_values) - np.min(bpm_values)
            acceleration = bpm_range / duration_sec
            local_var = local_variability(bpm_values)
            entropy_val = bpm_entropy(bpm_values)
            rhythmic_variability = np.mean(np.abs(np.diff(bpm_values)))
            tempo_jumps_threshold = 5  # Порог для скачков темпа
            tempo_jumps = np.sum(np.abs(np.diff(bpm_values)) > tempo_jumps_threshold)
            
            stdk, stdc = 0.2 * std_bpm, std_bpm
            crk, crc = 0.15 * change_rate * 10, change_rate
            jk, jc = 0.15 * jitter * 10, jitter
            brk, brc = 0.05 * bpm_range / 10, bpm_range
            ak, ac = 0.1 * abs(acceleration), acceleration
            lvk, lvc = 0.1 * local_var, local_var
            evk, evc = 0.1 * entropy_val * 10, entropy_val
            rvk, rvc = 0.15 * rhythmic_variability, rhythmic_variability
            tjk, tjc = 0.05 * tempo_jumps, tempo_jumps
            st.write("std_bpm", stdk, stdc)
            st.write("change_rate", crk, crc)
            st.write("jitter", jk, jc)
            st.write("bpm_range", brk, brc)
            st.write("acceleration", ak, ac)
            st.write("local_ver", lvk, lvc)
            st.write("entorpoy_val", evk, evc)
            st.write("rhythmic_variability", rvk, rvc)
            st.write("tempo_jumps", tjk, tjc)
            st.metric("Scorek", stdk + crk + jk + brk + ak + lvk + evk + rvk + tjk)
            st.metric("Scorec", stdc + crc + jc + brc + ac + lvc + evc + rvc + tjc)


def complexity_score(bpm_values, duration_sec):
    bpm_values = np.array(bpm_values)
    std_bpm = np.std(bpm_values)
    tempo_changes = np.sum(np.abs(np.diff(bpm_values)) > 3)
    change_rate = tempo_changes / duration_sec
    jitter = np.sum(np.diff(np.sign(np.diff(bpm_values))) != 0) / len(bpm_values)
    bpm_range = np.max(bpm_values) - np.min(bpm_values)
    acceleration = (bpm_values[-1] - bpm_values[0]) / duration_sec
    local_var = local_variability(bpm_values)
    entropy_val = bpm_entropy(bpm_values)
    score = (
        0.2 * std_bpm +
        0.2 * change_rate * 10 +
        0.15 * jitter * 10 +
        0.1 * bpm_range / 10 +
        0.1 * abs(acceleration) +
        0.1 * local_var +
        0.1 * entropy_val * 10
    )
    return round(score, 2)

def local_variability(bpm_values, window_size=5):
    segments = len(bpm_values) // window_size
    local_std = [np.std(bpm_values[i*window_size:(i+1)*window_size])
                 for i in range(segments) if len(bpm_values[i*window_size:(i+1)*window_size]) > 1]
    return np.mean(local_std) if local_std else 0

def bpm_entropy(bpm_values):
    hist, _ = np.histogram(bpm_values, bins=10, density=True)
    return entropy(hist)

def interpret_score(score):
    if score < 3:
        return "🟢 Very Simple (Loop-like, Minimal Variation)"
    elif score < 5:
        return "🟢🟡 Simple Groove (Some Movement, Still Stable)"
    elif score < 7:
        return "🟡 Moderately Complex (Live Feel, Some Jitter)"
    elif score < 9:
        return "🟠 Complex (Dynamic & Syncopated, Humanized)"
    elif score < 11:
        return "🔴 Highly Complex (Frequent Tempo Changes, Rich Structure)"
    else:
        return "🔴🔴 Ultra Complex (Experimental, Irregular, High Entropy)"
