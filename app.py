from flask import Flask
import numpy as np
import pickle

app = Flask(__name__)
model = pickel.load(open(model.pkl, 'rb'))

@app.route('/')
def home():
    return "Flask Heroku app."
    #return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.from.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text = 'Sentiment analysis: $ {}'.format(output))

if __name__=='__main__':
    app.run(debug=True)