## library
import argparse

## argparse
parser = argparse.ArgumentParser(description= 'argparse Example for Tistory')

## add argument
parser.add_argument('--a',
                    type= int,
                    default= 0,
                    help= 'first number')
parser.add_argument('--b',
                    type= int,
                    default= 0,
                    help= 'second number')
args = parser.parse_args()

## result
print(args.a + args.b)