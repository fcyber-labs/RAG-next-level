
import os
import flask_session
path = os.path.dirname(flask_session.__file__)
print('Files in flask_session:', os.listdir(path))
