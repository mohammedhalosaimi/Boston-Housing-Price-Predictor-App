from flask import Flask
from flask import render_template
app = Flask(__name__)


@app.route('/')
# @app.route('/hello/<name>')
def hello(name="Mohammed"):
    # return name
    return render_template('hello.html', username=name)

if __name__ == '__main__':
    app.run()
