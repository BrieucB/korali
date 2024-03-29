#!/usr/bin/env python3
import sys
import os
import json
import shutil
import copy
import glob
import argparse

def processFeature(source, destination, featureRelPath, featureName):
  featurePath = os.path.join(source, featureRelPath)
  featureReadmeFile = featurePath + '/README.rst'
  featureOutputDir = os.path.abspath(os.path.join(destination + featureRelPath, os.pardir))

  print('Processing file: ' + featureReadmeFile)

  featureReadmeString = '.. _feature_' + featureRelPath.lower().replace('./', '').replace('/', '-').replace(' ', '') + ':\n\n'

  # Creating subfolder list
  subFolderList = []
  list_dir = os.listdir(featurePath)
  for f in list_dir:
    fullPath = os.path.join(featurePath, f)
    if not os.path.isfile(fullPath):
      if (not '.o/' in fullPath and not '.d/' in fullPath and not '/_' in fullPath):
        subFolderList.append(f)

  # Creating feature's folder, if not exists
  if not os.path.exists(featureOutputDir):
    os.mkdir(featureOutputDir)

  # Determining if its a parent or leaf feature
  isParentFeature = True
  if (subFolderList == []):
    isParentFeature = False

  # If its leaf, link to source code
  if (isParentFeature == False):
    featureReadmeString += '.. hint::\n\n'
    featureReadmeString += '   Example code: `https://github.com/cselab/korali/tree/master/examples/features/' + featureRelPath.replace(
        './', ''
    ) + '/ <https://github.com/cselab/korali/tree/master/examples/features/' + featureRelPath.replace(
        './', '') + '/>`_\n\n'

  # Copying any images in the source folder
  for file in glob.glob(r'' + featurePath + '/*.png'):
    shutil.copy(file, featureOutputDir)

  # Reading original rst
  with open(featureReadmeFile, 'r') as file:
    featureReadmeString += file.read() + '\n\n'

  # If its parent, construct children features
  if (isParentFeature == True):
    featureReadmeString += '**Sub-Categories**\n\n'
    featureReadmeString += '.. toctree::\n'
    featureReadmeString += '   :titlesonly:\n\n'

    for f in subFolderList:
      subFeatureFullPath = os.path.join(featurePath, f)
      if (not '/_' in subFeatureFullPath):
        featureReadmeString += '   ' + featureName + '/' + f + '\n'
        subPath = os.path.join(featureRelPath, f)
        processFeature(source, destination, subPath, f)

  # Saving Feature's readme file
  featureReadmeString += '\n\n'
  with open(featureOutputDir + '/' + featureName + '.rst', 'w') as file:
    file.write(featureReadmeString)


def build_features(source, destination):
  shutil.rmtree(destination, ignore_errors=True, onerror=None)
  os.makedirs(destination)
  
  list_dir = os.listdir(source)
  for f in list_dir:
    fullPath = os.path.join(source, f)
    if not os.path.isfile(fullPath):
      if (not '.o/' in fullPath and not '.d/' in fullPath and not '/_' in fullPath):
        processFeature(source, destination, f, f)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--destination", help="Save the generated files in this folder")
  parser.add_argument("--source", help="Read files from this folder")
  args = parser.parse_args()

  build_features(args.source, args.destination) 
