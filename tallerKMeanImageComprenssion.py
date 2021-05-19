# integrantes: GABRIEL GOMEZ, EDUARDO DE LA HOZ, STEPHANIA DE LA HOZ, NEMESYS EVILLA

from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

imagen = io.imread('no mans sky origins w.jpg')
plt.subplot(221)
io.imshow(imagen)

filas = imagen.shape[0]
columnas = imagen.shape[1]

imagen = imagen.reshape(filas*columnas, 3)

kmeans = KMeans(n_clusters=8)
kmeans.fit(imagen)

imagenComprimida = kmeans.cluster_centers_[kmeans.labels_]
imagenComprimida = np.clip(imagenComprimida.astype('uint8'), 0, 255)
imagenComprimida = imagenComprimida.reshape(filas, columnas, 3)


plt.subplot(222)
io.imshow(imagenComprimida)
io.show()


#prueba de ejemplo aSDSDFASDFASDFASDFADF


