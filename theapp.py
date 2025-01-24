import os
import rpy2.robjects as robjects
from rpy2.robjects import conversion, default_converter
from rpy2.robjects.conversion import localconverter
from datetime import datetime
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from cmfrec import CMF, CMF_implicit
from flask import Flask

app = Flask(__name__)

def load_rdata(file):
    print("load_rdata. YASA")
    env = robjects.environments.Environment()
    robjects.r['load'](file, env)
    return {key: env[key] for key in env.keys()}

# Cargar los archivos .RData
lista_matrices = load_rdata('./Lista Matrices - Completa.RData')['lista_matriz_completa']
info_segmentos = load_rdata('./Info Segmentos.RData')['info_segmentos']
exp = load_rdata('./exp.RData')['exp']

# Armo DataFrame de exp
exp_df = pd.DataFrame({
    'predicted_k42': list(exp.rx2['predicted_k42']),
    'rango_etario': list(exp.rx2['rango_etario']),
    'sexo': list(exp.rx2['sexo'])
})
exp_df['descripcion'] = exp_df['predicted_k42'].astype(str) + '-' + exp_df['rango_etario'].astype(str) + '-' + exp_df['sexo']

# Armo DataFrame de info_segmentos
info_segmentos_df = pd.DataFrame({
    'CustomerIdentificationCard': list(info_segmentos.rx2['CustomerIdentificationCard']),
    'descripcion': list(info_segmentos.rx2['descripcion'])
})

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

            matriz = lista_matrices[int(exp_df[exp_df['descripcion'] == info_segmentos_df.loc[
                info_segmentos_df['CustomerIdentificationCard'] == cedula, 'descripcion'].values[0]].index[0])]

            # Busco información de productos y cédulas
            a_productos_1 = pd.DataFrame({'PRDCOD': list(robjects.r['colnames'](matriz))})
            a_ci_1 = pd.DataFrame({'Cedula': list(list(robjects.r['rownames'](matriz))[0])})

            # Genero matriz de productos recomendados
            matriz_coo = coo_matrix(
                (list(robjects.r['slot'](matriz, 'x')),
                 (list(robjects.r['slot'](matriz, 'i')),
                  list(robjects.r['slot'](matriz, 'j')))),
                shape=(robjects.r['dim'](matriz)[0], robjects.r['dim'](matriz)[1])
            )

            modelo = CMF_implicit(k=4, alpha=75, verbose=False).fit(matriz_coo)

            # Función topN
            def topN_propio(modelo, user, n):
                user_factors = modelo.A_[user]
                item_factors = modelo.B_
                scores = np.dot(item_factors, user_factors)
                items_escogidos = np.argsort(scores)[::-1][:n]
                return items_escogidos

            lista_productos_recomendados = pd.DataFrame({
                'PRDCOD': a_productos_1.iloc[
                    topN_propio(modelo, int(a_ci_1.index[a_ci_1['Cedula'] == cedula][0]), 150)
                ]['PRDCOD'].values
            })

            # Convertir a JSON
            lista_json = lista_productos_recomendados.to_json(orient='values')

            t2 = datetime.now()
            print(cedula, "time: ", t2 - t1)

            return lista_json

    except Exception as e:
        print(f"Error procesando la solicitud: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
