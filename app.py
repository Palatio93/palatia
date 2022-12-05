# from requests import request
from flask import Flask, render_template, request
from apriori import main as mainApriori
from metricas import main as mainMetricas
from cluster import main as mainCluster

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/presentacion')
def presentacion():
  return render_template('algoritmos/presentacion.html')

@app.route('/apriori', methods=['GET','POST'])
def apriori():
  if request.method == "GET":
    return render_template('algoritmos/apriori_settings.html')
  if request.method == "POST":
    f = request.files['dataset']
    f.save(secure_filename(f.filename))
    soporte = float(request.form['soporte'])
    confi = float(request.form['confi'])
    lifto = float(request.form['lifto'])
    Resultados = mainApriori(soporte,confi,lifto)
    return render_template('algoritmos/apriori.html', resultados=Resultados)


@app.route('/metricas')
def metricas():
  mainMetricas()
  return render_template('algoritmos/metricas.html')


@app.route('/cluster')
def cluster():
  mainCluster()
  return render_template('algoritmos/cluster.html')


