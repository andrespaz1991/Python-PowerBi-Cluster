criterios=['op','co','ex'] #Columnas de criterios a analizar
cantidad_cluster=5 #cantidad de conjuntos
nombre_columna_analizar="usuario" #columna a usar
nombre_nueva_columna="cluester" # Columna nueva
#####
import pandas as pd  #
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
rango_criterios = np.array(dataset[criterios])
rango_criterios =np.ascontiguousarray(rango_criterios, dtype=np.double) #
kmeans = KMeans(n_clusters=cantidad_cluster).fit(rango_criterios)
centroids = kmeans.cluster_centers_
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, rango_criterios)
closest
total=[centroids]
columnas={}
contadortotal=0
nombrecategoria=''
for n in centroids:
  columnas[contadortotal]=n
  contadortotal=contadortotal+1
clusters_finales = pd.DataFrame(columnas)  
res=np.transpose(clusters_finales)
nombrescolumnas=list()
for columna in criterios:
  nombrescolumnas.append(columna)
res.columns = [nombrescolumnas]
##########
columna_de_coeficientes=list()
for i in dataset[nombre_columna_analizar].index:
    coeficientes=list()
    for criterio in criterios:
        coeficientes.append(dataset.loc[i, criterio])   
    conjunto_coeficientes= np.array([coeficientes]) 
    coeficiente_por_registro = kmeans.predict(conjunto_coeficientes)
    columna_de_coeficientes.append(coeficiente_por_registro)    
dataset[nombre_nueva_columna] = pd.DataFrame(columna_de_coeficientes,columns=[nombre_nueva_columna],dtype ='string') 
print(dataset)
