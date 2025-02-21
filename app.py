import streamlit as st
from transformers import pipeline

translator = pipeline(model="thainq107/en-vi-mbart50")

def main():
    st.title('En-Vi Machine Translation')
    st.header('Model: mBART50. Dataset: IWSLT15-En-Vi')
    text_input = st.text_input("Sentence: ", "The bread is top notch as well")
    pred_sentences = translator(text_input, num_beams=5)
    pred_sentence = pred_sentences[0]['generated_text']
    st.success(f'En Sentence: {text_input} === Vi Sentence: {pred_sentence}')

if __name__ == '__main__':
     main() 
