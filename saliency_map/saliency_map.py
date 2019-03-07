#------------------------------------------------------------------------------
#   Implementation of the Saliency map section from the paper
#   https://arxiv.org/pdf/1312.6034v2.pdf
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------
import cv2, torch
import numpy as np
from torchvision import models
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt


#------------------------------------------------------------------------------
#	Main execution
#------------------------------------------------------------------------------
# # Create model
# model = models.resnet18(pretrained=True)
# model.cuda()
# model.eval()

# # Freeze trained weights
# for param in model.parameters():
#     param.requires_grad = False

# Read and Pre-process an image
img = cv2.imread("../images/dog.jpg")[...,::-1]
image = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
X = F.to_tensor(image)
X = F.normalize(X, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
X = torch.unsqueeze(X, dim=0)
X.requires_grad = True
X = X.cuda()