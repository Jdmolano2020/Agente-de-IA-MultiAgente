from flask import Blueprint

user_bp = Blueprint('user', __name__)

@user_bp.route('/test', methods=['GET'])
def test():
    return {'message': 'Ruta de usuario funcionando'}
