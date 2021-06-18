# Modified Pytorch Multitag Skeleton code

After install dependencies via requirements.txt, initiate via running train.py to train and then inference.py to generate predictions on the last 10 images in the csv file. Load images into /input/photos/Images and update trainNew.csv with image ids and labels as needed. Some example predictions are included in outputs as well as a loss plot.

# NOTES
inference.py takes the last 10 cells of the csv file and runs inference on them

Run shuffle.py and then inferenceRandom.py to run inferences on 10 randomly selected images in the dataset rather than the last 10 listed in the csv file.

To run inferences on an entire csv file, run inferenceWhole.py.
