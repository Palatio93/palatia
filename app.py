# from requests import request
from flask import Flask, render_template, request
from apriori import main as mainApriori
from metricas import main as mainMetricas
from cluster import main as mainCluster
from pronostico import main as mainPronostico

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/presentacion')
def presentacion():
  return render_template('algoritmos/presentacion.html')

# @app.route('/apriori', methods=['GET','POST'])
# def apriori():
#   if request.method == "GET":
#     return render_template('algoritmos/apriori_settings.html')
#   if request.method == "POST":
#     f = request.files['dataset']
#     f.save(secure_filename(f.filename))
#     soporte = float(request.form['soporte'])
#     confi = float(request.form['confi'])
#     lifto = float(request.form['lifto'])
#     Resultados = mainApriori(soporte,confi,lifto)
#     return render_template('algoritmos/apriori.html', resultados=Resultados)

@app.route('/apriori', methods=['GET','POST'])
def apriori():
  if request.method == "GET":
    return render_template('algoritmos/apriori_settings.html')
  if request.method == "POST":
    # f = request.files['dataset']
    # f.save(secure_filename(f.filename))
    soporte = float(request.form['soporte'])
    confi = float(request.form['confi'])
    lifto = float(request.form['lifto'])
    Resultados = mainApriori(soporte,confi,lifto)
    return render_template('algoritmos/apriori.html', resultados=Resultados)


@app.route('/metricas', methods=['GET','POST'])
def metricas():
  if request.method == "GET":
    return render_template('algoritmos/metricas.html')
  if request.method == "POST":
    obj1 = request.form['objeto1']
    if obj1 == '':
      obj1 = 0
    else:
      obj1 = int(obj1)
    obj2 = request.form['objeto2']
    if obj2 == '':
      obj2 = 1
    else:
      obj2 = int(obj2)
    dM = mainMetricas(obj1,obj2)
    distanceM = [{'distanceM': dM}]
    return render_template('algoritmos/metricasShow.html', distanceM=distanceM)


@app.route('/cluster')
def cluster():
  mainCluster()
  return render_template('algoritmos/cluster.html')


@app.route('/pronostico')
def pronostico():
  mainPronostico()
  return render_template('algoritmos/pronostico.html')

