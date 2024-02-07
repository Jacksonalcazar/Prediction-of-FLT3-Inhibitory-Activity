@echo off
start.py
@java -jar PaDEL-Descriptor.jar -dir ./ -2d -detectaromaticity -standardizenitro -maxruntime 60000 -retainorder -threads -1 -file data/dataFLT3.csv
@java -jar PaDEL-Descriptor.jar -dir ./ -fingerprints -detectaromaticity -standardizenitro -maxruntime 60000 -retainorder -threads -1 -file data/Fingerprints.csv
fp.py
final.py