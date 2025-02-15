from PIL import Image
from src.backbone import Backbone

backbone = Backbone()

dog_img = Image.open("dog.jpg")
dog_img = dog_img.resize((224, 224))

