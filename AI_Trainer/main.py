from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api", methods=["GET", "POST"])
def qa():
    if request.method == "POST":
        data = {"result": "To make the entire <div> act as a button, you can use the <button> tag instead of the <div>"}
        return jsonify(data)
    data = {"result": "To make the entire <div> act as a button, you can use the <button> tag instead of the <div>"}
    return jsonify(data)

app.run(debug=True)