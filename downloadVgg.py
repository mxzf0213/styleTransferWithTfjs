from keras.applications.vgg16 import VGG16
model = VGG16(include_top=False)
model.save('vgg16.h5')
