import os
import argparse
import tensorflow as tf
import models.model as model


def parse_args():
    parser = argparse.ArgumentParser(description='Deblur arguments')
    parser.add_argument('--phase', type=str, default='test', help='Determine whether to train or test')
    parser.add_argument('--datalist', type=str, default='./datalist_gopro.txt', help='Training datalist')
    parser.add_argument('--model', type=str, default='color', help='Model type: [lstm | gray | color]')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epoch', type=int, default=4000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='Initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use, set to -1 for CPU')
    parser.add_argument('--height', type=int, default=720,
                        help='Height for the TensorFlow placeholder, should be multiples of 16')
    parser.add_argument('--width', type=int, default=1280,
                        help='Width for the TensorFlow placeholder, should be multiples of 16 for 3 scales')
    parser.add_argument('--input_path', type=str, default='./testing_set',
                        help='Input path for testing images')
    parser.add_argument('--output_path', type=str, default='./testing_res',
                        help='Output path for testing images')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU/CPU mode
    if int(args.gpu) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Set up deblur model
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        deblur.test(args.input_path, args.output_path)
    elif args.phase == 'train':
        deblur.train()
    else:
        print('Phase should be set to either "test" or "train".')


if __name__ == '__main__':
    main()