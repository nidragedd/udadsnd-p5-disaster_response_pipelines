from flask.app import Flask

app = Flask(__name__)

# Trick: need to put this line after app initialization otherwise app is not yet defined
import src.webapp.routes
