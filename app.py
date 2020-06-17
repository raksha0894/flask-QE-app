from flask import Flask, jsonify, request
import kiwi
from kiwi import constants as const
from flask_cors import CORS


# load model
model = kiwi.load_model('trained_models/estimator_en_de/estimator_en_de.torch')
# app
app = Flask(__name__)
CORS(app)


def make_color(text, color):
    if color == 'red':
        red_array.append(text)
    else:
        green_array.append(text)


def get_color(bad_prob, threshold):
    return 'green' if bad_prob < threshold else 'red'


# routes
@app.route('/', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    threshold = 0.7
    source = data['source']
    mt = data['mt']
    model_out = model.predict({const.SOURCE: [source.lower()], const.TARGET: [mt.lower()]})
    bad_probs = model_out[const.TARGET_TAGS][0]
    global red_array, green_array
    red_array = []
    green_array = []
    for i in range(0, len(mt.split())):
        make_color(str(mt.split()[i]), get_color(bad_probs[i], threshold))
    output = {'green': green_array, 'red': red_array}
    return jsonify(results = output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
