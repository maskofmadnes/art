import librosa
import numpy as np
import streamlit as st

@st.cache_data
def audio(path): 
    return librosa.load(path, sr=None)
    
    
@st.cache_data
def dynamic_tempo(
        audio,
        hop_length=512,
        start_bpm=120.0,
        std_bpm=1.0,
        ac_size=8.0,
        max_tempo=320.0,
        aggregate=None,
    ):
    y, sr = audio
    return librosa.feature.tempo(
        y=y,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        std_bpm=std_bpm,
        ac_size=ac_size,
        max_tempo=max_tempo,
        aggregate=aggregate,
    )   
    
    
@st.cache_data
def time_tempo(
        audio, 
        dynamic_tempo, 
        hop_length=512
    ):
    y, sr = audio
    return librosa.times_like(
        dynamic_tempo, 
        sr=sr, 
        hop_length=hop_length
    )   
   
@st.cache_data 
def segments(time_tempo, dynamic_tempo):
    segments = []
    tempo_times = time_tempo
    tempo_dynamic = dynamic_tempo
    start = tempo_times[0]
    curr_tempo = tempo_dynamic[0]
    for time, tempo in zip(tempo_times[1:], tempo_dynamic[1:]):
        if round(curr_tempo, 2) != round(tempo, 2):
            segments += [[start, time, curr_tempo]]
            start = time
            curr_tempo = tempo
    segments += [[start, tempo_times[-1], curr_tempo]]
    return segments   
    
@st.cache_data
def music(
        audio,
        dynamic_clicks,
        volume=20, 
        click_freq=660.0, 
        click_duration=0.1,
    ):
    y, sr = audio
    click_dynamic = dynamic_clicks
    volume_reduce = volume / 100
    y_reduced = y * volume_reduce
    combined_audio = y_reduced + click_dynamic
    max_amplitude = np.mean(combined_audio)
    if max_amplitude > 1.0:
        combined_audio = combined_audio / max_amplitude
    return combined_audio, sr
    
@st.cache_data
def dynamic_clicks(
        audio,
        dynamic_times,
        hop_length,
        click_freq,
        click_duration
    ):
    y, sr = audio
    return librosa.clicks(
        times=dynamic_times,
        sr=sr,
        hop_length=hop_length,
        click_freq=click_freq,
        click_duration=click_duration,
        length=len(y),
    )
    
@st.cache_data
def dynamic_times(
        audio,
        dynamic_tempo,
        hop_length=512,
        start_bpm=120.0,
        tightness=100.0,
        trim=True,
        units="time",
    ):
    y, sr = audio
    return librosa.beat.beat_track(
        y=y,
        sr=sr,
        bpm=dynamic_tempo,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=tightness,
        trim=trim,
        units=units,
    )

