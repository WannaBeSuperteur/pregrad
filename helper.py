import tensorflow as tf
import math

# CNN of generator
def make_gen_CNN():
    model = tf.keras.Sequential()
    model_input = model
    
    model.add(layers.Conv2d(64, (3, 3), strides=(1, 1), padding='same', input_shape=[256, 256, 3]))
    model.add(layers.LeakyReLU(alpha=0.1)) # PReLU -> LeakyReLU

    for i in range(16):
        before_model = model
        model.add(layers.Conv2d(64, (3, 3), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))
        model.add(layers.BatchNorm2d(model))
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.Conv2d(64, (3, 3), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))
        model.add(layers.BatchNorm2d(model))
        model = Elementwise(tf.add)([before_model, model])

    before_model2 = model
    model.add(layers.Conv2d(64, (3, 3), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU(alpha=0.1)) # PReLU -> LeakyReLU
    model.add(layers.BatchNorm2d(model))
    model = Elementwise(tf.add)([before_model2, model])

    model.add(layers.Conv2d(64, (3, 3), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))
    model = SubpixelConv2d(scale=2)(model)
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv2d(64, (3, 3), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))
    model = SubpixelConv2d(scale=2)(model)
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv2d(3, (1, 1), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.tanh())

    return Model(inputs=model_input, outputs=model, name='gen')
    
def generator_CNN(image):

    # 1

# CNN of discriminator:
def make_dis_CNN():
def discriminator_CNN(image):

    model = tf.keras.Sequential()
    model_input = model
    
    model.add(layers.Conv2d(64, (3, 3), strides=(1, 1), padding='same', input_shape=[256, 256, 3]))
    model.add(layers.LeakyReLU(alpha=0.1)) # PReLU -> LeakyReLU
    
    model.add(layers.Conv2d(128, (3, 3), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.BatchNorm2d(model))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv2d(256, (3, 3), strides=(1, 1), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.BatchNorm2d(model))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv2d(512, (3, 3), strides=(1, 1), padding='same', input_shape=[256, 256, 3]))
    model.add(layers.BatchNorm2d(model))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv2d(1024, (3, 3), strides=(1, 1), padding='same', input_shape=[512, 512, 3]))
    model.add(layers.BatchNorm2d(model))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.LeakyReLU(alpha=0.1)) # PReLU -> LeakyReLU
    model.add(layers.BatchNorm2d(model))
    model = Elementwise(tf.add)([before_model2, model])

    model.add(layers.Conv2d(1, (3, 3), strides=(1, 1), padding='same', input_shape=[1024, 1, 3]))
    model.add(layers.sigmoid())

    return Model(inputs=model_input, outputs=model, name='dis')

# Min-Max problem
def minMaxLoss(imageHR, imageLR):

    # 3

# VGG19 net layer before i-th maxpooling and j-th convolution
# For obtaining final result (use EVERY part of CNN), set i = j = -1
def VGG19_layer(image, i, j):

    # 4

# L-layer deep network obtained by optimizing SR-specific loss function l^SR
def LNetParameter(imagesLR, imagesHR, thetaG):
    # finding thetaG where value of Sum(n=1,N)l^SR(Gen(I^LR_n), I^HR_n)/N is smallest
    N = len(imagesLR) # number of images

    # 5

# MSE Loss (L_SR MSE)
def MSEloss(imageHR, imageLR, r):
    gCNNLR = generator_CNN(imageLR)
    result = 0

    H = len(imageLR) # height of LR image
    W = len(imageLR[0]) # width of LR image
    
    for y in range(r*H):
        for x in range(r*W):
            result += (imageHR[y][x][0] - gCNNLR[y][x][0])
    result /= (r*r*W*H)

    return result

# VGG Loss (L_SR VGG)
def VGGloss(imageHR, imageLR, r, i, j):
    gCNNLR = generator_CNN(imageLR)
    piHR = VGG19_layer(imageHR, i, j) # pi_i,j(I^HR)x,y
    piCNNLR = VGG19_layer(gCNNLR, i, j) # pi_i,j(generator_CNN(I^LR))x,y
    result = 0

    H = len(piHR) # height of VGG19(HR) image
    W = len(piHR[0]) # width of VGG19(HR) image

    for y in range(H):
        for x in range(W):
            result += (imageHR[y][x][0] - gCNNLR[y][x][0])
    result /= (W*H)

    return result

# Adversarial loss
def adversarialLoss(imageLRs, N):
    result = 0

    for i in range(N):
        gCNNLR = generator_CNN(imageLRs[i]) # generator(LR[i])
        dgCNNLR = discriminator_CNN(gCNNLR) # discriminator(generator(LR))
        result -= math.log2(dgCNNLR)
        
    return result

# Calculate loss (perceptual loss function)
def loss(option, imageLRs, k, imageHR, r):

    N = len(imageLRs) # number of images in imageLR
    imageLR = imageLRs[k] # using k-th image in imageLR

    # content loss    
    contentLoss = 0
    if option == 0: # SRGAN-MSE
        contentLoss = MSEloss(imageHR, imageLR, r)
    elif option == 1: # SRGAN-VGG22
        contentLoss = VGGloss(imageHR, imageLR, r, 2, 2)
    elif option == 2: # SRGAN-VGG54
        contentLoss = VGGloss(imageHR, imageLR, r, 5, 4)

    # adversarial loss
    advLoss = adversarialLOss(imageLRs, N)

    return contentLoss + 0.001 * advLoss
