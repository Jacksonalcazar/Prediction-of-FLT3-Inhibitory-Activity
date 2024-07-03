# streamlit_app.py
import subprocess
import streamlit as st

st.title('FLt3 Inhibitory Activity Prediction')

# Ejecutar los comandos necesarios
subprocess.run(["python", "start.py"])
subprocess.run(["java", "-jar", "PaDEL-Descriptor.jar", "-dir", "./", "-2d", "-detectaromaticity", "-standardizenitro", "-maxruntime", "60000", "-retainorder", "-threads", "-1", "-file", "data/dataFLT3.csv"])
subprocess.run(["java", "-jar", "PaDEL-Descriptor.jar", "-dir", "./", "-fingerprints", "-detectaromaticity", "-standardizenitro", "-maxruntime", "60000", "-retainorder", "-threads", "-1", "-file", "data/Fingerprints.csv"])
subprocess.run(["python", "fp.py"])
subprocess.run(["python", "final.py"])

# Aquí irían los demás pasos del análisis y predicción
st.write("Procesamiento completado")
