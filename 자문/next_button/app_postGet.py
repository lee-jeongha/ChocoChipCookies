from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    name = ""

    if request.method == 'POST':
        name = request.form.get("name", "홍길동")

    name_list = name.split('.')

    return render_template('post_get.html', name=name, name_list=name_list)


if __name__ == "__main__":
    app.run(debug=True)
