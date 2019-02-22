from keras.applications.vgg19 import VGG19
model = VGG19(include_top=False)
model.save('vgg19.h5')