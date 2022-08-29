# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from pascal_voc_parser import get_data
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the reclassification from Places365 to Land use
file_name = 'places365_landuse.csv'
if not os.access(file_name, os.W_OK):
    raise ValueError('Error: no reclassification from Places365 to Land Use')
places365_landuse = list()
with open(file_name) as class_file:
    for line in class_file:
        places365_landuse.append(line.strip().split(' ')[1][:])
places365_landuse = tuple(places365_landuse)

# load Land Use classes
file_name = 'landuse_classes.csv'
if not os.access(file_name, os.W_OK):
    raise ValueError('Error: no Land Use classes')
landuse_classes = list()
with open(file_name) as class_file:
    for line in class_file:
        landuse_classes.append(line.strip().split(' ')[0][:])
landuse_classes = tuple(landuse_classes)

all_imgs, _, _ = get_data(options.test_path)
test_imgs = [s for s in all_imgs if s['imageset'] == 'test']

all_maps = []
all_topks = []
T = {}
P = {}
P_topk = {}
for idx, img_data in enumerate(test_imgs):
    print('{}/{}'.format(idx+1, len(test_imgs)))
    img_name = img_data['filepath']

    # load the test image
    #img_name = '0027_lado_b.jpg'
    if not os.access(img_name, os.W_OK):
        img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + img_url)

    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    print('{} prediction on {}'.format(arch,img_name))
    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    print('{} prediction on {}'.format(arch,img_name))
    # output the prediction of Land Use
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], landuse_classes[int(places365_landuse[idx[i]])-1]))
