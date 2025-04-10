# ART - 🎶 dynamic BPM analyzer
ART provides a web interface to analyze dynamic BPM
* 📈 Charts of BPM
* ⚙️ A lot of parameters of BPM distribution
* 📝 Table of time intervals with BPM
* 📊 Table of time onsets with BPM
  
![app_image](https://github.com/user-attachments/assets/7be4d834-891b-43ae-a340-4a0f90b85a4c)

## Requirements
* git
* python 3.13, 3.12, 3.11 (other versions most likely work but are not tested)
* uv
* make (Linux)
## Installation and Usage
### Linux
```shell
git clone https://github.com/kostya1F634/art.git
cd art
uv sync
make
```
### Windows
```shell
git clone https://github.com/kostya1F634/art.git
cd art
uv sync
uv run streamlit run .\art\app.py
```
