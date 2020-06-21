# handwritten-letters-and-digits-predictiong-model-using-keras
This model takes black and white images of handwritten letters and digits and predicts the correct char written in it.
How to use the model:
load an image using the PIL module (black and white only, one char per image!) as follows:
img = Image.open(*image-path*)
then:
pixel_array = numpy.array(img).
From the required methods file:
  *activate the create_image_for_prediction method.
  *load the model using the get_model method.
  *get the predicted character using the predict_character method.
  
  
  Image for example is attached.
  
