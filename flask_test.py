from flask import redirect, url_for, Flask, render_template
# Test flask app to serve up a report from AutoVis

app = Flask(__name__, static_folder="static/")

@app.route('/')
def metrics():
    return render_template('test_report.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)