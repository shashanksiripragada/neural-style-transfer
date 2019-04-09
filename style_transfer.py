import torch
import torchvision
import torchvision.transforms as T
import PIL
import numpy as np

from image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def features_from_img(imgpath, imgsize):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = img.type(dtype)
    return extract_features(img_var, cnn), img_var

dtype = torch.FloatTensor

cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.type(dtype)


for param in cnn.parameters():
    param.requires_grad = False

def extract_features(x, cnn):
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def content_loss(content_weight, content_current, content_original):
    loss = content_weight * ((content_current-content_original).pow(2).sum())    
    return loss 

def gram_matrix(features, normalize=True):
    N,C,H,W = features.shape
    F1 = torch.reshape(features,(N,C,H*W)) #N,C,H*W
    F2 = F1.permute(0,2,1) #N,H*W,C
    gram = F1.matmul(F2)
    if normalize:
        gram = gram/(H*W*C)    
    return gram

def style_loss(feats, style_layers, style_targets, style_weights):
    loss = torch.zeros([1])
    for i in range(len(style_layers)):
        curr_style = feats[style_layers[i]]
        w_i = style_weights[i]
        
        loss += w_i*((gram_matrix(curr_style)-style_targets[i]).pow(2).sum())
     
    return loss

def tv_loss(img, tv_weight):
    vertical = tv_weight * ((img[:,:,1:,:]-img[:,:,:-1,:]).pow(2).sum())
    horiz = tv_weight * ((img[:,:,:,1:]-img[:,:,:,:-1]).pow(2).sum())
    
    return (vertical+horiz)


def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = False):
    
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_feats = extract_features(content_img, cnn)
    content_target = content_feats[content_layer].clone()

    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_feats = extract_features(style_img, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(style_feats[idx].clone()))

    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1).type(dtype)
    else:
        img = content_img.clone().type(dtype)

    img.requires_grad_()

    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    optimizer = torch.optim.Adam([img], lr=initial_lr)

    for t in range(200):
        if t < 190:
            img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        img_feats = extract_features(img, cnn)
        
        c_loss = content_loss(content_weight, img_feats[content_layer], content_target)
        s_loss = style_loss(img_feats, style_layers, style_targets, style_weights)
        t_loss = tv_loss(img, tv_weight) 
        loss = c_loss + s_loss + t_loss
        
        loss.backward()

        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img], lr=decayed_lr)
        optimizer.step()

    img = deprocess(img.data.cpu())
    img.save('style_transfer_out.png')
        
params1 = {
    'content_image' : 'content.jpg',
    'style_image' : 'style.jpg',
    'image_size' : 192,
    'style_size' : 512,
    'content_layer' : 3,
    'content_weight' : 5e-2, 
    'style_layers' : (1, 4, 6, 7),
    'style_weights' : (20000, 500, 12, 1),
    'tv_weight' : 5e-2
}

style_transfer(**params1)    
    

