import pandas as pd 

d = {'Name': ['Derek', 'Kalito', 'Aldirris'],
     'Age': [22,24,21],
     'City': ['Valorant','Minas Gerais','Payday']}
df = pd.DataFrame(data=d, columns=['Name', 'Age', 'City'])

print(d)

# cosas = {"Cluster": [x for x in d[0]]}

for ind in df.index:
  print(df['Name'][ind],"-- vive en --",df['City'][ind],"-- desde hace --",df['Age'][ind],"-- años --")

# reglas = [{
#     "Regla": ", ".join([x for x in res[0]]),
#     "Soporte":res[1]*100,
#     "Confianza":res[2][0][2]*100,
#     "Lift":res[2][0][3]
#   } for res in Resultados]