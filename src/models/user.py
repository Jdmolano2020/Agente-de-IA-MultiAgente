from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Puedes definir aquí tu modelo de usuario si lo necesitas
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     ...
