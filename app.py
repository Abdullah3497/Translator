import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained translation model
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-ur"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🌍 English ➝ Urdu Translator")
st.write("Type English text below and get its Urdu translation.")

# Text input
text = st.text_area("Enter English text:", "I love learning artificial intelligence.")

if st.button("Translate"):
    if text.strip():
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        # Generate Urdu translation
        outputs = model.generate(**inputs, max_length=100)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Urdu Translation:")
        st.success(translation)
    else:
        st.warning("⚠️ Please enter some text to translate.")
