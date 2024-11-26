from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
import pandas as pd
import streamlit as st

model = LocalLLM(
    api_base="http://localhost:11434",
    model="llama3.2"
)

st.title("Data analysis with PandasAI")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head(3))

    # Aktifkan Whitelist saat membuat SmartDataframe
    df = SmartDataframe(data, config={"llm": model, "whitelisted_dependencies": ["matplotlib", "pandasai"]})
    
    prompt = st.text_area("Enter your prompt:")

    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                st.write(df.chat(prompt))