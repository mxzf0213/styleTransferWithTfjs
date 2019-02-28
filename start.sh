python3 downloadVgg.py
rm -rf model
mkdir model
tensorflowjs_converter --input_format keras vgg16.h5 model/
firefox index.html

