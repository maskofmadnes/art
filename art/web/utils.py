from mutagen import File
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4


def extract_cover(uploaded_file):
    audio = File(uploaded_file, easy=False)
    if audio is None:
        return None

    if audio.tags is None:
        return None

    # MP3 (ID3v2)
    if uploaded_file.name.endswith(".mp3") and hasattr(audio, "tags"):
        for tag in audio.tags.values():
            if tag.FrameID == "APIC":  # ID3 Attached Picture
                return tag.data

    # FLAC
    if uploaded_file.name.endswith(".flac") and isinstance(audio, FLAC):
        if audio.pictures:
            return audio.pictures[0].data

    # MP4 / M4A
    if uploaded_file.name.endswith((".m4a", ".mp4")) and isinstance(audio, MP4):
        covr = audio.tags.get("covr")
        if covr:
            return covr[0]

    return None
