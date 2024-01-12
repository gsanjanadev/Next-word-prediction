import tensorflow
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import streamlit as st

st.title('Next Word Prediction App')

# Load the model and tokenizer
model_LSTM = load_model('Model_LSTM_Predictor.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

model_BLSTM = load_model('Model_BLSTM_Predictor.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

model_GRU = load_model('Model_GRU_Predictor.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))
def Predict_Next_Words(model, tokenizer, text):
 
  sequence = tokenizer.texts_to_sequences([text])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence))
  predicted_word = ""
   
  for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break
   
  print(predicted_word)
  return predicted_word

def show_prediction(model, model_name, input_text):
    st.subheader(f'Prediction using {model_name}')
    st.write(f'Input Text: {input_text}')
    
    try:
        text = input_text.split(" ")
        text = text[-3:]
        predicted_word = Predict_Next_Words(model, tokenizer, text)
        st.write(f'Predicted Next Word: {predicted_word}')

    except Exception as e:
        st.error(f"Error occurred: {e}")



input_text_lstm = st.text_input('Input Text (LSTM Model)')
submit_button_lstm = st.button('Predict (LSTM)')
if submit_button_lstm:
    show_prediction(model_LSTM, 'LSTM', input_text_lstm)

input_text_blstm = st.text_input('Input Text (Bidirectional LSTM Model)')
submit_button_blstm = st.button('Predict (Bidirectional LSTM)')
if submit_button_blstm:
    show_prediction(model_BLSTM, 'Bidirectional LSTM', input_text_blstm)

input_text_gru = st.text_input('Input Text (GRU Model)')
submit_button_gru = st.button('Predict (GRU)')
if submit_button_gru:
    show_prediction(model_GRU, 'GRU', input_text_gru)