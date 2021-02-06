Training normal:

python normal_training.py

Incremntal training:

python training_ditilled.py

Preparing dataset:

In tools run oid_to_pascal_voc.py and then run XML_to_YOLOv3.py .. you will get the train and test lables with paths in yolov3format::

For dataset preperation =, make two folders in coustom_data train and test. put train images and labels in trian and test in test respectively. 
Then run XML_to_YOLOv3.py to convert ot yolov3 format. 


The labdma value is .007 and lambda 2 value is .02, we get good reslts for these both. 

