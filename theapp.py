import os
# os.environ['R_HOME'] = 'C:/Users/rmosteiro/AppData/Local/Programs/R/R-4.3.2'

import rpy2.robjects as robjects
from rpy2.robjects import conversion, default_converter
from rpy2.robjects.conversion import localconverter
from datetime import datetime
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from cmfrec import CMF_implicit
from flask import Flask

app = Flask(__name__)

#Funciones
# Cargo Data
def load_rdata(file):
    env = robjects.environments.Environment()
    robjects.r['load'](file, env)
    return {key: env[key] for key in env.keys()}

# topN Propio
def topN_propio(modelo, user, n):
        user_factors = modelo.A_[user]
        item_factors = modelo.B_
        scores = np.dot(item_factors, user_factors)
        items_escogidos = np.argsort(scores)[::-1][:n]
        return items_escogidos

# Cargar los archivos .RData
# lista_matrices = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/APP/Archivos para App/Lista Matrices - Completa.RData')['lista_matriz_completa']
lista_matrices = load_rdata('./data/Lista Matrices - Completa.RData')['lista_matriz_completa']

# info_segmentos = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/APP/Archivos para App/Info Segmentos.RData')['info_segmentos']
info_segmentos = load_rdata('./data/Info Segmentos.RData')['info_segmentos']

info_segmentos_df = pd.DataFrame({'CustomerIdentificationCard': list(info_segmentos.rx2['CustomerIdentificationCard']),
                                  'descripcion': list(info_segmentos.rx2['descripcion'])})

# exp = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/APP/Archivos para App/exp.RData')['exp']
exp = load_rdata('./data/exp.RData')['exp']

exp_df = pd.DataFrame({'predicted_k42': list(exp.rx2['predicted_k42']),
                       'rango_etario': list(exp.rx2['rango_etario']),
                       'sexo': list(exp.rx2['sexo']), 
                       'k': list(exp.rx2['k']),
                       'alpha': list(exp.rx2['alpha'])})
exp_df['descripcion'] = exp_df['predicted_k42'].astype(str)  + '-' + exp_df['rango_etario'].astype(str) + '-' + exp_df['sexo']

# mercadologico = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/APP/Archivos para App/Mercadologico.RData')['mercadologico']
mercadologico = load_rdata('./data/Mercadologico.RData')['mercadologico']

mercadologico = pd.DataFrame({'ProductAlternateKey': list(mercadologico.rx2['ProductAlternateKey']),
                                  'FamilyName': list(mercadologico.rx2['FamilyName'])})

# lista_topfamilias = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/APP/Archivos para App/Lista Familias Top.RData')['lista_topfamilias']
lista_topfamilias = load_rdata('./data/Lista Familias Top.RData')['lista_topfamilias']

# lista_canasta_regular = load_rdata('//srvfserver/gestioncomercial/Analytics/Segmentacion/APP/Archivos para App/Lista Canasta Regular.RData')['listado_canasta_regular']
lista_canasta_regular = load_rdata('./data/Lista Canasta Regular.RData')['listado_canasta_regular']

@app.route('/')
def index():
    return {'message': 'Welcome to the Flask app!'}

@app.route('/hello')
def hello():
    return {'hello': 'world'}

@app.route('/getInfoCI/<string:document>')
def get_info(document):
    try:    
        # Usar el contexto de conversión de rpy2
        with localconverter(default_converter):
            t1 = datetime.now()
            cedula = document

            # Colocar IF en caso de estar en lista_topfamilias (7 RecSys + 3 Canasta Regular)
            if (info_segmentos_df["CustomerIdentificationCard"] == cedula).any() : 
                matriz = lista_matrices[int(exp_df[exp_df['descripcion'] == info_segmentos_df.loc[info_segmentos_df['CustomerIdentificationCard'] == cedula, 'descripcion'].values[0]].index[0])]

                a_productos_1 = pd.DataFrame({'PRDCOD': list(robjects.r['colnames'](matriz))})
                a_ci_1 = pd.DataFrame({'Cedula': list(list(robjects.r['rownames'](matriz))[0])})

                # Genero Productos Recomendados Modelo
                matriz_coo = coo_matrix((list(robjects.r['slot'](matriz, 'x')), (list(robjects.r['slot'](matriz, 'i')), list(robjects.r['slot'](matriz, 'j')))), shape=(robjects.r['dim'](matriz)[0], robjects.r['dim'](matriz)[1]))

                k_opt = int(exp_df.loc[exp_df["descripcion"] == info_segmentos_df.loc[info_segmentos_df['CustomerIdentificationCard'] == 
                                    cedula, 'descripcion'].iloc[0], "k"].iloc[0])
                alpha_opt = int(exp_df.loc[exp_df["descripcion"] == info_segmentos_df.loc[info_segmentos_df['CustomerIdentificationCard'] == 
                                    cedula, 'descripcion'].iloc[0], "alpha"].iloc[0])
                
                modelo = CMF_implicit(k=k_opt, alpha=alpha_opt, w_user = 1, w_item = 1, verbose = False).fit(matriz_coo)

                lista_productos_recomendados_modelo = pd.DataFrame({'PRDCOD': a_productos_1.iloc[topN_propio(modelo,  int(a_ci_1.index[a_ci_1['Cedula'] == cedula][0]), 200)]['PRDCOD'].values})

                
                lista_productos_recomendados_modelo = lista_productos_recomendados_modelo.merge(mercadologico, left_on='PRDCOD', right_on='ProductAlternateKey', how='left')
                lista_productos_recomendados_modelo = lista_productos_recomendados_modelo[lista_productos_recomendados_modelo['FamilyName'].isin(list(lista_topfamilias.rx2[cedula].rx2['FamilyName']))].groupby("FamilyName").nth(0).head(7)
                    
                # Genero Productos Recomendados Canasta Regular
                lista_productos_recomendados_canasta  =  pd.DataFrame({
                    'PRDCOD': lista_canasta_regular.rx2[cedula].rx2["ProductAlternateKey"]})

                # 'PRDCOD': lista_canasta_regular.rx2[cedula].rx2["ProductAlternateKey"],
                # 'ProductAlternateKey': lista_canasta_regular.rx2[cedula].rx2["ProductAlternateKey"],
                # 'FamilyName': lista_canasta_regular.rx2[cedula].rx2["FamilyName"]}).head(3)
                
                # Genero Listado Productos Recomendados
                lista_productos_recomendados = pd.concat([lista_productos_recomendados_modelo, lista_productos_recomendados_canasta])
                
            # Else (canasta regular (top 10))
            else :
                lista_productos_recomendados  =  pd.DataFrame({
                    'PRDCOD': lista_canasta_regular.rx2[cedula].rx2["ProductAlternateKey"]})
                # 'PRDCOD': lista_canasta_regular.rx2[cedula].rx2["ProductAlternateKey"],
                # 'ProductAlternateKey': lista_canasta_regular.rx2[cedula].rx2["ProductAlternateKey"],
                # 'FamilyName': lista_canasta_regular.rx2[cedula].rx2["FamilyName"]}).head(10)
                
            # JSon
            # lista_json = lista_productos_recomendados.to_json(orient='records', indent=4)
            lista_json = lista_productos_recomendados.to_json(orient='values')

            t2 = datetime.now()
            print(cedula, "time: ", t2 - t1)
            
            return lista_json

    except Exception as e:
        print(f"Error procesando la solicitud: {e}")
        return []

if __name__ == "__main__":
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)



