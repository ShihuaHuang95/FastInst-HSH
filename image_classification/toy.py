import os
import os.path as osp
import argparse
import math
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from utils import AverageMeter, accuracy
from resnet import ResNet50

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="The path of imagenet", type=str, default="/data8022/huangshihua/Datasets/ImageNet1k")
parser.add_argument("--gpu", help="The gpu(s) to use", type=str, default="1,2,3,4,5")
parser.add_argument("--batch-size", help="The batch on every device for validation", type=int, default=100)
parser.add_argument("--workers", help="Number of workers", type=int, default=20)
parser.add_argument("--pt-path", help='the path of pretrained weights', type=str, default="ResNet50D-Params@17.40-FLOPs@1924M.pth")

args = parser.parse_args()
if args.gpu == "all":
    device_list = range(torch.cuda.device_count())
    args.gpu = ",".join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(",")]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

arch_config = {'d': [2, 0, 1, 1, 1], 
'e': [0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.35, 0.2, 0.2, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.2, 0.25], 
'w': [0, 1, 0, 0, 1, 2]
}
net = ResNet50(
        depth_list=arch_config['d'],
        expand_ratio_list=arch_config['e'],
        width_mult_list=arch_config['w'],
)
# print(net)
state_dicts = torch.load(args.pt_path, map_location="cpu")
net.load_state_dict(state_dicts, strict=True)
print("     ## Pretrained Weights Loading Successful!!! ###")
args.batch_size = args.batch_size * max(len(device_list), 1)

data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        osp.join(args.path, "val"),
        transforms.Compose(
            [
                transforms.Resize(int(math.ceil(224 / 0.875))),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True,
    drop_last=False,
)

net = torch.nn.DataParallel(net).cuda()
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()

net.eval()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

with torch.no_grad():
    with tqdm(total=len(data_loader), desc="Validate") as t:
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            # compute output
            output = net(images)
            loss = criterion(output, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            t.set_postfix(
                {
                    "loss": losses.avg,
                    "top1": top1.avg,
                    "top5": top5.avg,
                    "img_size": images.size(2),
                }
            )
            t.update(1)

print("Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f" % (losses.avg, top1.avg, top5.avg))

