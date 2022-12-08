import pandas as pd 

d = {'Name': ['Derek', 'Kalito', 'Aldirris','Zuris'],
     'Age': [22,24,21,23],
     'City': ['Valorant','Minas Gerais','Payday','Bohemia Town']}
df = pd.DataFrame(data=d, columns=['Name', 'Age', 'City'])

print(d)

# cosas = {"Cluster": [x for x in d[0]]}

for ind in df.index:
  print(df['Name'][ind],"-- vive en --",df['City'][ind],"-- desde hace --",df['Age'][ind],"-- años --")

print("Deben ser 4, tenemos -> ",len(df))
print("Deben ser 3, tenemos -> ",len(df.columns))

# reglas = [{
#     "Regla": ", ".join([x for x in res[0]]),
#     "Soporte":res[1]*100,
#     "Confianza":res[2][0][2]*100,
#     "Lift":res[2][0][3]
#   } for res in Resultados]