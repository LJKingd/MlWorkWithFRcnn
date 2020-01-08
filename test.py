# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#.txt-->.xml
#! /usr/bin/python
# -*- coding:UTF-8 -*-
###########################
##获取labels路径，并转为xml存入Annotations
##############
import os, sys
import glob
from PIL import Image

import os
import random
import shutil
#



import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import shutil

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



#将图片复制到JPEGImages
def moveImg(pathimg,pathimgmt):
  print("开始复制")
  for file in os.listdir(pathimg):
        source_file = os.path.join(pathimg, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, pathimgmt)

  print("复制图片文件成功！")


#生成xml数据，将其保存在一个Main下单独的文件夹
def vocxml(src_img_dir,src_txt_dir,src_xml_dir):
  if not os.path.exists(src_xml_dir):
    os.makedirs(src_xml_dir)
  ##删除旧数据xml
  filelisto=os.listdir(src_xml_dir)                #列出该目录下的所有文件名
  for f in filelisto:
      filepatho = os.path.join( src_xml_dir, f )   #将文件名映射成绝对路劲
      os.remove(filepatho)                 #若为文件，则直接删除

  print("开始生成XML\n") 
  img_Lists = glob.glob(src_img_dir + '/*.jpg')
   
  img_basenames = [] # e.g. 100.jpg
  for item in img_Lists:
      img_basenames.append(os.path.basename(item))
   
  img_names = [] # e.g. 100
  for item in img_basenames:
      temp1, temp2 = os.path.splitext(item)
      img_names.append(temp1)
   
  for img in img_names:
      im = Image.open((src_img_dir + '/' + img + '.jpg'))
      width, height = im.size
   
      # open the crospronding txt file
      gt = open(src_txt_dir + '/' + img.replace('img','label',1) + '.txt',encoding='UTF-8').read().splitlines()
      #gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
   
      # write in xml file
      #os.mknod(src_xml_dir + '/' + img + '.xml')
      xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
      xml_file.write('<annotation>\n')
      xml_file.write('    <folder>VOC2007</folder>\n')
      xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
      xml_file.write('    <source>\n')
      xml_file.write('        <database>VOC Database</database>\n')
      xml_file.write('        <annotation>battery_VOC</annotation>\n')
      xml_file.write('        <image>lking</image>\n')
      xml_file.write('        <flickrid>0</flickrid>\n')
      xml_file.write('    </source>\n')
      xml_file.write('    <owner>\n')
      xml_file.write('        <flickrid>lking</flickrid>\n')
      xml_file.write('        <name>lking</name>\n')
      xml_file.write('    </owner>\n')

      xml_file.write('    <size>\n')
      xml_file.write('        <width>' + str(width) + '</width>\n')
      xml_file.write('        <height>' + str(height) + '</height>\n')
      xml_file.write('        <depth>3</depth>\n')
      xml_file.write('    </size>\n')
      xml_file.write('    <segmented>0</segmented>\n')
   
      # write the region of image on xml file
      for img_each_label in gt:
          spt = img_each_label.split(' ') #这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
          #判断边界小于1，让它加一
          if int(spt[2])<1:
              spt[2]=1
          if int(spt[3])<1:
              spt[3]=1

          if str(spt[1])=='带电芯充电宝':
              nname = "core_battery"
          elif str(spt[1])=='不带电芯充电宝':
              nname="coreless_battery"
          else:
              continue    

          xml_file.write('    <object>\n')
          xml_file.write('        <name>' + str(nname) + '</name>\n')
          xml_file.write('        <pose>Unspecified</pose>\n')
          xml_file.write('        <truncated>0</truncated>\n')
          xml_file.write('        <difficult>0</difficult>\n')
          xml_file.write('        <bndbox>\n')
          xml_file.write('            <xmin>' + str(int(spt[2])) + '</xmin>\n')
          xml_file.write('            <ymin>' + str(int(spt[3])) + '</ymin>\n')
          xml_file.write('            <xmax>' + str(int(spt[4])) + '</xmax>\n')
          xml_file.write('            <ymax>' + str(int(spt[5])) + '</ymax>\n')
          xml_file.write('        </bndbox>\n')
          xml_file.write('    </object>\n')
   
      xml_file.write('</annotation>')
  print("生成xml成功\n")


#生成test.txt, core_battery_test.txt,core_battery_test.txt
def createTest(xmlfilepath,txtsavepath):

  
  total_xml = os.listdir(xmlfilepath)
   
  num = len(total_xml)
  list = range(num)
  tn = int(num) #测试集的个数
  Pathmain = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit2007','VOC2007','ImageSets','Main') #main文件路径
  if not os.path.exists(Pathmain):
    os.makedirs(Pathmain)
##删除旧数据
  #Mfilelisto=os.listdir(Pathmain)                #列出该目录下的所有文件名
  #for f in Mfilelisto:
      #Mfilepatho = os.path.join( Pathmain, f )   #将文件名映射成绝对路劲
    #  os.remove(Mfilepatho)                 #若为文件，则直接删除

  ftest =  open(Pathmain+'/test.txt', 'w')  # 生成测试集
  
  #生成test.txt
  for i in list:
      name = total_xml[i][:-4] + '\n'
      ftest.write(name) 

  ftest.close()


  #生成生成core_battery_test.txt,core_battery_test.txt
  
  #ftcore_battery = open(Pathmain+'/core_battery_test.txt', 'w')  # 生成测试集

  #ftcoreless_battery = open(Pathmain+'/coreless_battery_test.txt', 'w')     # 生成测试集

  #for i in list:
     # name = total_xml[i][:-4] 
    #  if len(name) < 22:
    #    name=name+ ' 1'+'\n'        
   #   else:
     #   name=name+ ' -1'+'\n'
     # ftcore_battery.write(name)

 # for i in list:
  #    name = total_xml[i][:-4] 
    #  if len(name) > 22:
    #    name=name+ ' 1'+'\n'        
    #  else:
   #     name=name+ ' -1'+'\n'
   #   ftcoreless_battery.write(name)


  #ftcore_battery.close()
 # ftcoreless_battery.close()


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=2, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=9899, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')

###########新加
  parser.add_argument('--img_dir', dest='img_dir',
                      help='directory to load img',default='/home/taotao/lking/testfile/Image_test',
                      type=str)
  parser.add_argument('--label_dir', dest='label_dir',
                      help='directory to load label_dir',default='/home/taotao/lking/testfile/Anno_test',
                      type=str)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':
  #lking:delete cache


  cache=os.path.join(cfg.ROOT_DIR,'data','cache' )
  if os.path.exists(cache):
     shutil.rmtree(cache)
  Anno_cache=os.path.join(cfg.ROOT_DIR,'data','VOCdevkit2007','annotations_cache' )
  if os.path.exists(Anno_cache):
     shutil.rmtree(Anno_cache)

  args = parse_args()
#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
  pathimg = args.img_dir
#####复制图片
  pathimglink=os.path.join(cfg.ROOT_DIR,'data','VOCdevkit2007','VOC2007','JPEGImages' )
  # moveImg(pathimg,pathimglink)#！！！！！！！！！！！！！！！！！！！！
  ##########！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
  #set ruanlianjie
  rename = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit2007','VOC2007',str(time.time())+'JPEGImages', )
  if os.path.exists(pathimglink):
    #os.remove(pathimglink)
    os.rename(pathimglink,rename)
  os.symlink(pathimg,pathimglink)
  #生成xml
  # VEDAI 图像存储位置lking
  src_img_dir = pathimg
  # VEDAI 图像的 ground truth 的 txt 文件存放位置
  src_txt_dir = args.label_dir
  src_xml_dir = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit2007','VOC2007','Annotations' )
  vocxml(src_img_dir,src_txt_dir,src_xml_dir)

  ##
  #生成test文件lking
  xmlfilepath = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit2007','VOC2007','Annotations' ) # xml文件,需要改！！！！！！！！！！！！！！！！！！！！！！！！！！
  txtsavepath = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit2007','VOC2007','ImageSets','Main') # 生成的训练集，测试集，验证集位置
  createTest(xmlfilepath,txtsavepath)
  ###################################lking end


  if torch.cuda.is_available() and not args.cuda:
     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

 

  print('Called with args:')
  print(args)

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset#加载模型目录
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name,map_location=torch.device('cpu'))#只运行cpu的torch
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i,args.img_dir))##lking
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
  os.remove(pathimglink)
