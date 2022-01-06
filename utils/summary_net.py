from torchsummary import summary
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.models as models
import sys
sys.path.append('/home/lab325/gry/gry-graduation/style-binary-classification/')

from model import *

# Load the pretrained model
model = models.vgg16(num_classes=2)
net = VGG16BinaryNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device('cpu')
net = net.to(device)
# img = Image.open('../get_AVA_dataset/AVA_analysis/wh_ratio_hist.png')
# img = img.convert('RGB')
# img = transform(img)
# img = img.unsqueeze(dim=0)
# img = img.to(device)
# with torch.no_grad():
#     out = net(img)
# print(out)


summary(net, (3, 224, 224))
print("net: ", net)
