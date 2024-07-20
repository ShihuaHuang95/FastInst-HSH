# FastInst + OFA

### New Updates

* [ImageNet1k evaluation](image_classification)
* [OFA subnet: top1@78.1](checkpoints/ResNet50D-Params@17.40-FLOPs@1924M.pth)
* [OFA subnet: top1@76.1](checkpoints/ResNet50D-Params@9.58-FLOPs@1284M.pth)
* [Backbone Register](fastinst/modeling/backbone/resnas.py)
* [New Backbone Config](./configs/coco/instance-segmentation/fastinst_NASR50-vd-dcn_ppm-fpn_x3_640.yaml)

***

## Get Started
* ImageNet1k evaluation
  ```sh
  cd imgage_classification

  python toy.py -path /data8022/huangshihua/Datasets/ImageNet1k --gpu 0 --pt-path ./ResNet50D-Params@17.40-FLOPs@1924M.pth  

  ```
  Finally you should acchieve Top1@78.1 and Top5@94.0
  Another smaller backbone 
  ```sh
  python toy.py -path /data8022/huangshihua/Datasets/ImageNet1k --gpu 0 --pt-path ./ResNet50D-Params@9.58-FLOPs@1284M.pth  
  ```
  Finally you should acchieve Top1@76.1 and Top5@92.8


* Convert Pretrained weight into Detectron2
  ```sh
  python tools/convert-timm-to-d2.py ./image_classification/ResNet50D-Params@17.40-FLOPs@1924M.pth ./checkpoints/ResNet50D-Params@17.40-FLOPs@1924M.pth

  python tools/convert-timm-to-d2.py ./image_classification/ResNet50D-Params@9.58-FLOPs@1284M.pth ./checkpoints/ResNet50D-Params@9.58-FLOPs@1284M.pth
  ```

* Train the FastInst with new OFA models
  ``` sh
  Do as usually.
  ```
