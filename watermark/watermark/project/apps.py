from django.apps import AppConfig
from custom_layers import SliceLayer , IRDWTLayer , RDWTLayer , FullModel
from tensorflow.keras.models import load_model

class ProjectConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'project'

    model = None

    def ready(self):
        global model
        print("loading_model")
        self.model = load_model('main_model_downloaded_xray.keras' , custom_objects={'RDWTLayer': RDWTLayer, 'IRDWTLayer': IRDWTLayer, 'SliceLayer': SliceLayer , 'FullModel' : FullModel})
        print("model downloaded bro")