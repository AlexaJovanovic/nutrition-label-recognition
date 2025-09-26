# Nutrition Label Reader 🥗

This project is an algorithm designed to **read and process nutrition label data** from food products.  
The goal is to extract key nutritional information such as:

- Calories
- Macronutrients (protein, carbs, fats)
- Additional values (fiber, sugar, etc.)

## 🚀 Features
- Parse and structure nutrition label data
- Extract key macros for further analysis
- Designed for extensibility (can be adapted to different label formats)

## 🛠 Tech Stack
- **Python 3**  
- Virtual environment (`venv`) for clean dependency management  

## 📦 Setup
Clone the repository and create a virtual environment:
```bash
git clone https://github.com/AlexaJovanovic/nutrition-label-recognition.git
cd nutrition-label-recognition
python -m venv .venv
.venv\Scripts\Activate.ps1 # on Windows
pip install -r requirements.txt
