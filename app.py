import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('linearmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':

        rate = request.form['rate']
        sales1stmonth = request.form['sales1stmonth']
        sales2ndmonth = request.form['sales2ndmonth']

        data = [rate,sales1stmonth,sales2ndmonth]

        int_features = [int(x) for x in data]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)