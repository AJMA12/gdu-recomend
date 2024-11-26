from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

data = {
    '36489070': [135790, 555555, 777777, 999999],
    '12345678': [987654, 321012, 345678],
}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class Query_By_Document(Resource):
    def get(self, document):
        if document in data:
            return {
                'document': document,
                'data': data[document]
            }
        else:
            return {
                'document': document,
                'data': "no hay"
            }

api.add_resource(HelloWorld, '/')

api.add_resource(Query_By_Document, '/getInfoCI/<string:document>')

# if __name__ == '__main__':
app.run(debug=False)
# app.run(host="0.0.0.0", port=8700) #, ssl_context=('cert.pem', 'key.pem'))