//从model.json参数结构文件加载VGG16模型
console.log('Loading Model...');
async function loadModel() {
    vgg16_model = await tf.loadLayersModel('./model/model.json')
    //warm up 
    /*
    const temp = get_wb('block1_conv1')
    temp[1].print()
    */
    console.log('Loaded!')

    enableStylizeButtons()
}

loadModel()


//取vgg16模型layer_name层参数(w,b)
function get_wb(layer_name) {
    const kernel = vgg16_model.getLayer(layer_name)['kernel'].val
    const bias = vgg16_model.getLayer(layer_name)['bias'].val
    kernel.trainable = false
    bias.trainable = false
    return [kernel, bias]
}

//卷积层实现
function conv_relu(input, wb) {
    return tf.tidy(() => {
        const conv = tf.conv2d(input, wb[0], strides = [1, 1], pad = 'same')
        const relu = tf.relu(tf.add(conv, wb[1]))
        return relu
    }
    )
}

//池化层实现->最大值池化
function pool(input) {
    return tf.maxPool(input, filterSize = [2, 2], strides = [2, 2], pad = 'same')
}

//根据内容图片产生噪声图片
function get_random_img() {
    return tf.tidy(() => {
        const noise_image = tf.randomUniform([1, IMAGE_HEIGHT, IMAGE_WEIGHT, 3], -20, 20, dtype = 'float32')
        const random_image = tf.mul(noise_image, NOISE).add(tf.mul(content_img, tf.sub(tf.scalar(1), NOISE)))
        return tf.variable(random_image)
    }
    )
}


// Helper function for setting an image
function setImage(element, selectedValue) {
    if (selectedValue === 'file') {
        //console.log('file selected');
        fileSelect.onchange = (evt) => {
            const f = evt.target.files[0];
            const fileReader = new FileReader();
            fileReader.onload = ((e) => {
                element.src = e.target.result;
            });
            fileReader.readAsDataURL(f);
        }
        fileSelect.click();
    } else if (selectedValue === 'pic') {
        openModal(element);
    } else {
        element.src = 'images/' + selectedValue + '.jpg';
    }
}

//建立模型，只用到了前若干层，所以不用取原模型的所有层
function buildModel(input) {
    net_input = input
    net_conv1_1 = conv_relu(net_input, block1_conv1)
    net_conv1_2 = conv_relu(net_conv1_1, block1_conv2)
    net_pool1 = pool(net_conv1_2)
    net_conv2_1 = conv_relu(net_pool1, block2_conv1)
    net_conv2_2 = conv_relu(net_conv2_1, block2_conv2)
    net_pool2 = pool(net_conv2_2)
    net_conv3_1 = conv_relu(net_pool2, block3_conv1)
}

//计算损失函数，这里仅计算风格损失
function loss() {
    return tf.tidy(() => {

        buildModel(gen_img)

        var style_loss = tf.scalar(0.)

        //computing style loss
        //conv1_1
        M = net_conv1_1.shape[1] * net_conv1_1.shape[2];
        N = net_conv1_1.shape[3];
        weight = 0.2
        const net_conv1_1_gram = gram(net_conv1_1, M, N)
        const style_conv1_1_gram = gram(style_conv1_1, M, N)
        style_loss = style_loss.add(tf.scalar(weight).div(tf.scalar(4 * M * M * N * N)).mul(tf.sum(tf.pow(tf.sub(net_conv1_1_gram, style_conv1_1_gram), 2))));

        //conv2_1
        M = net_conv2_1.shape[1] * net_conv2_1.shape[2];
        N = net_conv2_1.shape[3];
        weight = 0.2
        const net_conv2_1_gram = gram(net_conv2_1, M, N)
        const style_conv2_1_gram = gram(style_conv2_1, M, N)
        style_loss = style_loss.add(tf.scalar(weight).div(tf.scalar(4 * M * M * N * N)).mul(tf.sum(tf.pow(tf.sub(net_conv2_1_gram, style_conv2_1_gram), 2))));

        //conv3_1
        M = net_conv3_1.shape[1] * net_conv3_1.shape[2];
        N = net_conv3_1.shape[3];
        weight = 0.2
        const net_conv3_1_gram = gram(net_conv3_1, M, N)
        const style_conv3_1_gram = gram(style_conv3_1, M, N)
        style_loss = style_loss.add(tf.scalar(weight).div(tf.scalar(4 * M * M * N * N)).mul(tf.sum(tf.pow(tf.sub(net_conv3_1_gram, style_conv3_1_gram), 2))));

        return style_loss;
    }
    )
}

//计算feature矩阵的gram(相关性)矩阵
function gram(x, size, deep) {
    return tf.tidy(() => {
        const y = x.reshape([size, deep])
        const z = tf.matMul(tf.transpose(y), y)
        return z
    }
    )
}

//预处理图片张量以符合输入网络的size
function preprocess(image) {
    console.log(image.width, image.height)
    return tf.tidy(
        () => {
            const tensor = tf.browser.fromPixels(image).toFloat()
            const resized = tf.image.resizeBilinear(tensor, [IMAGE_HEIGHT, IMAGE_WEIGHT])
            const reshape = resized.expandDims(0)
            const changed = reshape.sub(IMAGE_MEAN_VALUE)
            return changed
        }
    )
}

//将输出张量转化为图片
function topixel(image) {
    const tensor = image.add(IMAGE_MEAN_VALUE)
    const resized = tensor.reshape([IMAGE_HEIGHT, IMAGE_WEIGHT, 3])
    const modified = tf.image.resizeBilinear(resized, [IMAGE_HEIGHT, contentImage.width])
    const changed = modified.div(tf.scalar(255.)).clipByValue(0., 1.)
    tensor.dispose()
    resized.dispose()
    modified.dispose()
    tf.browser.toPixels(changed, genImage)
}

//按钮激活
function enableStylizeButtons() {
    styleButton.disabled = false;
    styleButton.textContent = 'Stylize';
}
function disableStylizeButtons() {
    styleButton.disabled = true;
}

//加载模型参数以及得到风格图片特征矩阵
async function initContentStyle() {
    block1_conv1 = get_wb('block1_conv1');
    block1_conv2 = get_wb('block1_conv2');
    block2_conv1 = get_wb('block2_conv1');
    block2_conv2 = get_wb('block2_conv2');
    block3_conv1 = get_wb('block3_conv1');

    await buildModel(style_img);
    style_conv1_1 = net_conv1_1;
    style_conv2_1 = net_conv2_1;
    style_conv3_1 = net_conv3_1;
}

//训练模型，采用adam算法优化，learning_rate = 1.0
function train() {
    const optimizer = tf.train.adam(1.0)
    for (let i = 0; i < TRAIN_STEPS; i++) {
        optimizer.minimize(
            () => {
                return loss()
            }
        )
    }
    topixel(gen_img)
}

var vgg16_model;

var content_img;
var contentImage = document.getElementById("content-img");
var style_img;
var styleImage = document.getElementById("style-img");
var gen_img;
var genImage = document.getElementById("gen-img");

var fileSelect = document.getElementById("fileSelect");

var styleButton = document.getElementById('style-button');

// Initialize selectors
contentSelect = document.getElementById('content-select');
contentSelect.onchange = (evt) => setImage(contentImage, evt.target.value);
contentSelect.onclick = () => contentSelect.value = '';
styleSelect = document.getElementById('style-select');
styleSelect.onchange = (evt) => setImage(styleImage, evt.target.value);
styleSelect.onclick = () => styleSelect.value = '';

styleButton.onclick = () => {
    disableStylizeButtons()
    test().finally(() => {
        enableStylizeButtons();
    })
};

//初始化web摄像头
function initalizeWebcamVariables() {
    camModal = $('#cam-modal');

    snapButton = document.getElementById('snap-button');
    webcamVideoElement = document.getElementById('webcam-video');

    navigator.getUserMedia = navigator.getUserMedia ||
        navigator.webkitGetUserMedia || navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;

    camModal.on('hidden.bs.modal', () => {
        stream.getTracks()[0].stop();
    })

    camModal.on('shown.bs.modal', () => {
        navigator.getUserMedia(
            {
                video: true
            },
            (stream) => {
                stream = stream;
                webcamVideoElement.srcObject = stream;
                webcamVideoElement.play();
            },
            (err) => {
                console.error(err);
            }
        );
    })
}

initalizeWebcamVariables();

//打开摄像头
function openModal(element) {
    camModal.modal('show');
    snapButton.onclick = () => {
        const hiddenCanvas = document.getElementById('hidden-canvas');
        const hiddenContext = hiddenCanvas.getContext('2d');
        hiddenCanvas.width = webcamVideoElement.width;
        hiddenCanvas.height = webcamVideoElement.height;
        hiddenContext.drawImage(webcamVideoElement, 0, 0,
            hiddenCanvas.width, hiddenCanvas.height);
        const imageDataURL = hiddenCanvas.toDataURL('image/jpg');
        element.src = imageDataURL;
        camModal.modal('hide');
    };
}

const IMAGE_HEIGHT = 224;
const IMAGE_WEIGHT = 224;
const NOISE = tf.scalar(0.5);
const IMAGE_MEAN_VALUE = tf.scalar(128.)
const TRAIN_STEPS = 150

contentImage.height = IMAGE_HEIGHT;
styleImage.height = IMAGE_HEIGHT;

var net_input;
var net_conv1_1;
var net_conv1_2;
var net_pool1;
var net_conv2_1;
var net_conv2_2;
var net_pool2;
var net_conv3_1;

var style_conv1_1;
var style_conv2_1;
var style_conv3_1;

var block1_conv1;
var block1_conv2;
var block2_conv1;
var block2_conv2;
var block3_conv1;

//训练的一系列过程
async function test() {
    await tf.nextFrame();
    styleButton.textContent = 'Stylizing image...';
    await tf.nextFrame();
    content_img = preprocess(contentImage)
    style_img = preprocess(styleImage)
    gen_img = get_random_img()
    await initContentStyle();
    await train()

    console.log(tf.memory())
}

