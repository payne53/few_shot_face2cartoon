import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from models import ResnetGenerator
from dataset import ImageFolder
from utils import *
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model_path', type=str, help='model path')
parser.add_argument('--remark', type=str, default='')
parser.add_argument('--label', type=int, default=0)
args = parser.parse_args()

args.result_dir = './results/{}'.format(args.remark)
os.makedirs(args.result_dir, exist_ok=True)
print(args.result_dir)

class Photo2Cartoon:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.genA2B = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        self.genB2A = ResnetGenerator(ngf=32, img_size=256, light=True).to(self.device)
        # params = torch.load(args.model_path, map_location=self.device)
        # params = load_hdfs(args.model_path)
        params = load_params(args.model_path)
        load_params(self.genA2B, params['genA2B'])
        load_params(self.genB2A, params['genB2A'])
        # self.genA2B.load_state_dict(params['genA2B'])
        # self.genB2A.load_state_dict(params['genB2A'])
        print("Load model success. ")
        self.img_size = 256
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.testA = ImageFolder(args.dataset, test_transform, label=args.label)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)

    def test(self):
        self.genA2B.eval(), self.genB2A.eval()
        with torch.no_grad():
            for n, (real_A, label_A) in tqdm(enumerate(self.testA_loader)):
                real_A = real_A.to(self.device)
                label_A = label_A.to(self.device)

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A, label_A)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B, label_A)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A, label_A)

                # A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                #                       # cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                #                       # RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                #                       # cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                #                       RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                #                       # cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                #                       RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 1)
                #
                # cv2.imwrite(os.path.join(args.result_dir, 'A2B_%d.png' % (n + 1)), A2B * 255.0)

                real_A = RGB2BGR(tensor2numpy(denorm(real_A[0])))
                cv2.imwrite(os.path.join(args.result_dir, '%d_real_A.png' % (n + 1)), real_A * 255.0)
                fake_A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
                cv2.imwrite(os.path.join(args.result_dir, '%d_fake_A2B.png' % (n + 1)), fake_A2B * 255.0)
                fake_A2B2A = RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))
                cv2.imwrite(os.path.join(args.result_dir, '%d_fake_A2B2A.png' % (n + 1)), fake_A2B2A * 255.0)

                fake_A2A_heatmap = cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size)
                cv2.imwrite(os.path.join(args.result_dir, '%d_A2A_heatmap.png' % (n + 1)), fake_A2A_heatmap * 255.0)
                fake_A2B_heatmap = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                cv2.imwrite(os.path.join(args.result_dir, '%d_A2B_heatmap.png' % (n + 1)), fake_A2B_heatmap * 255.0)
                fake_A2B2A_heatmap = cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size)
                cv2.imwrite(os.path.join(args.result_dir, '%d_A2B2A_heatmap.png' % (n + 1)), fake_A2B2A_heatmap * 255.0)

if __name__ == '__main__':
    c2p = Photo2Cartoon()
    c2p.test()