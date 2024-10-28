import os
# os.environ['R_HOME'] = 'C:/Users/rmosteiro/AppData/Local/Programs/R/R-4.3.2'
# JMV
# os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

import rpy2.robjects as robjects

from rpy2.robjects import conversion, default_converter

from datetime import datetime
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from cmfrec import CMF, CMF_implicit
import json

from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)

def load_rdata(file):
    env = robjects.environments.Environment()
    robjects.r['load'](file, env)
    return {key: env[key] for key in env.keys()}

# Cargar los archivos .RData
# lista_matrices = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/Productos Recomendados - Py/Lista Matrices - Completa.RData')['lista_matriz_completa']
# info_segmentos = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/Productos Recomendados - Py/Info Segmentos.RData')['info_segmentos']
# exp = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/Productos Recomendados - Py/exp.RData')['exp']

# JMV
lista_matrices = load_rdata('./Lista Matrices - Completa.RData')['lista_matriz_completa']
info_segmentos = load_rdata('./Info Segmentos.RData')['info_segmentos']
exp = load_rdata('./exp.RData')['exp']
# JMV

# Armo DataFrame de exp
exp_df = pd.DataFrame({'predicted_k42': list(exp.rx2['predicted_k42']),
                       'rango_etario': list(exp.rx2['rango_etario']),
                       'sexo': list(exp.rx2['sexo'])})

exp_df['descripcion'] = exp_df['predicted_k42'].astype(str)  + '-' + exp_df['rango_etario'].astype(str) + '-' + exp_df['sexo']

# Armo DataFrame de info_segmentos
info_segmentos_df = pd.DataFrame({'CustomerIdentificationCard': list(info_segmentos.rx2['CustomerIdentificationCard']),
                                  'descripcion': list(info_segmentos.rx2['descripcion'])})

                                

class Query_By_Document(Resource):
    def get(self, document):

        t1 = datetime.now()
        
        cedula = document

        matriz = lista_matrices[int(exp_df[exp_df['descripcion'] == info_segmentos_df.loc[info_segmentos_df['CustomerIdentificationCard'] == cedula, 'descripcion'].values[0]].index[0])]

        # Busco Informacion de Productos y Cedulas dada una cedula concreta
        
        

        a_productos_1 = pd.DataFrame({'PRDCOD': list(robjects.r['colnames'](matriz))})
        a_ci_1 = pd.DataFrame({'Cedula': list(list(robjects.r['rownames'](matriz))[0])})

        # # Genero Productos Recomendados (modelo: ver si accedo a modelos generados o genero uno)
        matriz_coo = coo_matrix((list(robjects.r['slot'](matriz, 'x')), (list(robjects.r['slot'](matriz, 'i')), list(robjects.r['slot'](matriz, 'j')))), shape=(robjects.r['dim'](matriz)[0], robjects.r['dim'](matriz)[1]))

        modelo = CMF_implicit(k=4, alpha=75, verbose = False).fit(matriz_coo)

        # # topN Propio
        def topN_propio(modelo, user, n):
                user_factors = modelo.A_[user]
                item_factors = modelo.B_
                scores = np.dot(item_factors, user_factors)
                items_escogidos = np.argsort(scores)[::-1][:n]
                return items_escogidos

        lista_productos_recomendados = pd.DataFrame({'PRDCOD': a_productos_1.iloc[topN_propio(modelo,  int(a_ci_1.index[a_ci_1['Cedula'] == cedula][0]), 150)]['PRDCOD'].values})
        # # JSon
        lista_json = lista_productos_recomendados.to_json(orient='values') # orient='records', indent=0)

        t2 = datetime.now()
        print(cedula, "time: ", t2-t1)

        # print(lista_json)

        return lista_json

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(Query_By_Document, '/getInfoCI/<string:document>')

api.add_resource(HelloWorld, '/')

app.run(debug=False)

