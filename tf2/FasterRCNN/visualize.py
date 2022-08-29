#
# Faster R-CNN in PyTorch and TensorFlow 2 w/ Keras
# tf2/FasterRCNN/visualize.py
# Copyright 2021-2022 Bart Trzynadlowski
#
# Routines for visualizing model results and debug information.
#

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import xml.etree.ElementTree as ET
import os

def _draw_rectangle(ctx, corners, color, thickness = 4):
  y_min, x_min, y_max, x_max = corners
  ctx.rectangle(xy = [(x_min, y_min), (x_max, y_max)], outline = color, width = thickness)

def _draw_text(image, text, position, color, scale = 1.0, offset_lines = 0):
  """
  Parameters
  ----------
  image : PIL.Image
    Image object to draw on.
  text : str
    Text to render.
  position : Tuple[float, float]
    Location of top-left corner of text string in pixels.
  offset_lines : float
    Number of lines to offset the vertical position by, where a line is the
    text height.
  """
  font = ImageFont.load_default()
  text_size = font.getsize(text)
  text_image = Image.new(mode = "RGBA", size = text_size, color = (0, 0, 0, 0))
  ctx = ImageDraw.Draw(text_image)
  ctx.text(xy = (0, 0), text = text, font = font, fill = color)
  scaled = text_image.resize((round(text_image.width * scale), round(text_image.height * scale)))
  position = (round(position[0]), round(position[1] + offset_lines * scaled.height))
  image.paste(im = scaled, box = position, mask = scaled)

def _class_to_color(class_index):
  return list(ImageColor.colormap.values())[class_index + 1]

def show_anchors(output_path, image, anchor_map, anchor_valid_map, gt_rpn_map, gt_boxes, display = False):
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  
  # Draw all ground truth boxes with thick green lines
  for box in gt_boxes:
    _draw_rectangle(ctx, corners = box.corners, color = (0, 255, 0))

  # Draw all object anchor boxes in yellow
  for y in range(anchor_valid_map.shape[0]):
    for x in range(anchor_valid_map.shape[1]):
      for k in range(anchor_valid_map.shape[2]):  
        if anchor_valid_map[y,x,k] <= 0 or gt_rpn_map[y,x,k,0] <= 0:
          continue  # skip anchors excluded from training
        if gt_rpn_map[y,x,k,1] < 1:
          continue  # skip background anchors
        height = anchor_map[y,x,k*4+2]
        width = anchor_map[y,x,k*4+3]
        cy = anchor_map[y,x,k*4+0]
        cx = anchor_map[y,x,k*4+1]
        corners = (cy - 0.5 * height, cx - 0.5 * width, cy + 0.5 * height, cx + 0.5 * width)
        _draw_rectangle(ctx, corners = corners, color = (255, 255, 0), thickness = 3)
 
  image.save(output_path)
  if display:
    image.show()

def show_detections(output_path, show_image, image, scored_boxes_by_class_index, class_index_to_name):
  # Draw all results
  ctx = ImageDraw.Draw(image, mode = "RGBA")
  color_idx = 0
  for class_index, scored_boxes in scored_boxes_by_class_index.items():
    for i in range(scored_boxes.shape[0]):
      scored_box = scored_boxes[i,:]
      class_name = class_index_to_name[class_index]
      text = "%s %1.2f" % (class_name, scored_box[4])
      color = _class_to_color(class_index = class_index)
      _draw_rectangle(ctx = ctx, corners = scored_box[0:4], color = color, thickness = 2)
      _draw_text(image = image, text = text, position = (scored_box[1], scored_box[0]), color = color, scale = 1.5, offset_lines = -1)

  # Output
  if show_image:
    image.show()
  if output_path is not None:
    image.save(output_path)
    print("Wrote detection results to '%s'" % output_path)

def write_detections(output_path, image, image_file, scored_boxes_by_class_index, class_index_to_name, options):
  project_database = os.path.basename(options.dataset_dir)
  project_image = 'JPEGImages'
  project_annotation = options.annot_dir
  image_src = 'gsv'

  project_folder = os.path.dirname(options.dataset_dir)
  project_path = project_folder + '/' + project_database + '/'
  project_image_path = project_path + project_image + '/'
  project_annotation_path = project_path + project_annotation + '/'

  # create the file structure
  data = ET.Element('annotation')
  folder = ET.SubElement(data, 'folder')
  filename = ET.SubElement(data, 'filename')
  source = ET.SubElement(data, 'source')
  database = ET.SubElement(source, 'database')
  annotation = ET.SubElement(source, 'annotation')
  image = ET.SubElement(source, 'image')
  size = ET.SubElement(data, 'size')
  width = ET.SubElement(size, 'width')
  height = ET.SubElement(size, 'height')
  depth = ET.SubElement(size, 'depth')
  segmented = ET.SubElement(data, 'segmented')
  folder.set('name', project_folder)
  image_filename = os.path.basename(image_file)
  filename.text = image_filename
  database.text = project_database
  annotation.text = project_database
  image.text = image_src
  img_file = project_image_path + image_filename
  im = Image.open(img_file)
  w, h = im.size
  width.text = str(w)
  height.text = str(h)
  depth.text = '3'
  segmented.text = '0'

  # print(image_filename)
  for class_index, scored_boxes in scored_boxes_by_class_index.items():
    for i in range(scored_boxes.shape[0]):
      scored_box = scored_boxes[i,:]
      class_name = class_index_to_name[class_index]

#  for reg in img_metadata[img]['regions']:
      object_ = ET.SubElement(data, 'object')
      name = ET.SubElement(object_, 'name')
      pose = ET.SubElement(object_, 'pose')
      truncated = ET.SubElement(object_, 'truncated')
      difficult = ET.SubElement(object_, 'difficult')
      bndbox = ET.SubElement(object_, 'bndbox')
      name.text = class_name
      pose.text = 'Unspecified'
      truncated.text = 'Unspecified'
      difficult.text = 'Unspecified'
      xmin = ET.SubElement(bndbox, 'xmin')
      ymin = ET.SubElement(bndbox, 'ymin')
      xmax = ET.SubElement(bndbox, 'xmax')
      ymax = ET.SubElement(bndbox, 'ymax')
      xmin.text = str(int(scored_box[1]/2))
      ymin.text = str(int(scored_box[0]/2))
      xmax.text = str(int(scored_box[3]/2))
      ymax.text = str(int(scored_box[2]/2))

  xmlfile = os.path.splitext(image_filename)[0]
  xmlfile = project_annotation_path + xmlfile + '.xml'
  # create a new XML file with the results
  mydata = ET.tostring(data, encoding='unicode', method='xml')
  myfile = open(xmlfile, 'w')
  myfile.write(mydata)
