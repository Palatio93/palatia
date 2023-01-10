# from requests import request
from flask import Flask, render_template, request, send_file
from apriori import main as mainApriori
from metricas import main as mainMetricas
from cluster import main as mainCluster
# from pronostico import main as mainPronostico
from arboles import main as mainArboles
from bosques import main as mainBosques
from glob import glob
from io import BytesIO
from zipfile import ZipFile
import os

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

@app.route('/aprioriDownload')
def downloadApriori():
  target = './static/images/apriori/'

  stream = BytesIO()
  with ZipFile(stream, 'w') as zf:
      for file in glob(os.path.join(target, '*.pdf')):
          zf.write(file, os.path.basename(file))
  stream.seek(0)

  return send_file(
      stream,
      as_attachment=True,
      download_name='apriori.zip'
  )

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

@app.route('/metricasDownload')
def downloadMetricas():
  target = './static/images/metricas/'

  stream = BytesIO()
  with ZipFile(stream, 'w') as zf:
      for file in glob(os.path.join(target, '*.pdf')):
          zf.write(file, os.path.basename(file))
  stream.seek(0)

  return send_file(
      stream,
      as_attachment=True,
      download_name='metricas.zip'
  )

@app.route('/cluster')
def cluster():
  top10, CentroidesH, CentroidesP, elbow = mainCluster()
  return render_template('algoritmos/cluster.html', top10=top10, CentroidesH=CentroidesH, CentroidesP=CentroidesP, elbow=elbow)

@app.route('/clusterDownload')
def downloadCluster():
  target = './static/images/cluster/'

  stream = BytesIO()
  with ZipFile(stream, 'w') as zf:
      for file in glob(os.path.join(target, '*.pdf')):
          zf.write(file, os.path.basename(file))
  stream.seek(0)

  return send_file(
      stream,
      as_attachment=True,
      download_name='cluster.zip'
  )

@app.route('/arboles')
def arboles():
  acc_score, Matriz_Clasificacion1, criterio, reporte_clasif, ImportanciaMod1 = mainArboles(False)
  return render_template('algoritmos/arboles.html', acc_score=acc_score, Matriz_Clasificacion1=Matriz_Clasificacion1, criterio=criterio, reporte_clasif=reporte_clasif, ImportanciaMod1=ImportanciaMod1)

@app.route('/arbolesDownload')
def downloadArboles():
  target = './static/images/arboles/'

  stream = BytesIO()
  with ZipFile(stream, 'w') as zf:
      for file in glob(os.path.join(target, '*.pdf')):
          zf.write(file, os.path.basename(file))
  stream.seek(0)

  return send_file(
      stream,
      as_attachment=True,
      download_name='arboles.zip'
  )

@app.route('/bosques')
def bosques():
  acc_score, Matriz_Clasificacion2, criterio, reporte_clasif, ImportanciaMod = mainBosques()
  return render_template('algoritmos/bosques.html', acc_score=acc_score, Matriz_Clasificacion2=Matriz_Clasificacion2, criterio=criterio, reporte_clasif=reporte_clasif, ImportanciaMod2=ImportanciaMod)

@app.route('/bosquesDownload')
def downloadBosques():
  target = './static/images/bosques/'

  stream = BytesIO()
  with ZipFile(stream, 'w') as zf:
      for file in glob(os.path.join(target, '*.pdf')):
          zf.write(file, os.path.basename(file))
  stream.seek(0)

  return send_file(
      stream,
      as_attachment=True,
      download_name='bosques.zip'
  )