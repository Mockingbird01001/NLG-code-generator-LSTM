from flask import Flask, request, render_template
from visualisation import Visu
import json

top_k = 10
model = Visu()

def get_all_predictions(text_sentence, top_clean=5):
    print(text_sentence, top_clean)
    pred = model.prediction_lstm(text_sentence, top_clean)
    datas = {'tmp__1': pred[0], 'tmp_07': pred[1], 'tmp_04': pred[2], 'tmp_01': pred[3]}
    return datas

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_end_predictions', methods=['post'])
def get_prediction_eos():
    try:
        input_text = request.json['input_text']
        top_k = request.json['top_k']
        res = get_all_predictions(input_text, top_clean=int(top_k))
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run()
