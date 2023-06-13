from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
keras = tf.keras
Tokenizer = keras.preprocessing.text.Tokenizer
import json

app = Flask(__name__)

MAX_LEN = 60
TRUNC_TYPE='post'
PAD_TYPE='pre'

PRED_CLASSES = ['hate_speech','offensive_language','ok']

try:
    model = keras.models.load_model('model/hate_comment_detection_with_nlp.h5')
except Exception as e:
    print(e)

try:
    with open('hate_comment_detection_with_nlp_word_index.json', 'r') as file:
        data = json.load(file)
        # Convert the JSON data to a dictionary
        word_index = dict(data)
        oov_token = '<oov>'

        tokenizer = Tokenizer(oov_token=oov_token)
        tokenizer.word_index = word_index
except Exception as e:
    print(e)

@app.route('/preds',methods=['POST'])
def predictHateComments():
    try:
        # get json data
        req_body = request.json
        text = req_body['comment']

        # tokenize
        encoded = tokenizer.texts_to_sequences([text])
        encoded = keras.utils.pad_sequences(encoded,MAX_LEN,padding=PAD_TYPE,truncating=TRUNC_TYPE)

        preds = model.predict(encoded)
        pred = PRED_CLASSES[np.argmax(preds[0])]

        return jsonify({'pred':pred}),200
    except:
        return jsonify({'pred':None, "err": "Some error"}),500

if __name__ == "__main__":
    app.run(debug=True)