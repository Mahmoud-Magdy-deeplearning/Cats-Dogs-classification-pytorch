import argparse
from src.train import train
from src.test import test
from src.inference import inference

import sys
sys.path.insert(1, ".")
parser = argparse.ArgumentParser()
parser.add_argument("-train_dir", default=rf"F:\Machine learning projects\pytorch Dogs vs Cats\train", type=str)
parser.add_argument("-test_dir", default=rf"F:\Machine learning projects\pytorch Dogs vs Cats\test1", type=str)
parser.add_argument("-inf_dir", default=rf"F:\Machine learning projects\pytorch Dogs vs Cats\simple\inference", type=str)
parser.add_argument("-model_dir", default=rf"F:\Machine learning projects\pytorch Dogs vs Cats\simple\parameters", type=str)
parser.add_argument("-tensorboard_dir", default=rf"F:\Machine learning projects\pytorch Dogs vs Cats\simple\tensorboard", type=str)
parser.add_argument("-mode",default="train", type=str , choices=['train', 'test' , "inf"])
parser.add_argument("-mode",default="train", type=str , choices=['train', 'test' , "inf"])
parser.add_argument('-cuda', action='store_true')


args = parser.parse_args()

if(args.mode=="train"):
    train(args)

elif(args.mode=="test"):
    test(args)

elif(args.mode=="inf"):
    test(args)
