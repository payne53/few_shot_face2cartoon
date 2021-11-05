import sys
import time
import itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .networks import *
from utils import *
from glob import glob
from .face_features import FaceFeatures

from config import *
from pdb import set_trace

class UgatitSadalinHourglass(object):
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.experiment_name = self.result_dir.split('/')[-1]
        self.dataset = args.dataset
        self.hdfs_root = os.path.join(HDFS_HOME, 'experiment')

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.faceid_weight = args.faceid_weight
        self.cls_weight = args.cls_weight

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.rho_clipper = args.rho_clipper
        self.w_clipper = args.w_clipper
        self.pretrained_weights = args.pretrained_weights

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# faceid_weight : ", self.faceid_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)
        print("# cls_weight", self.cls_weight)
        print("# rho_clipper: ", self.rho_clipper)
        print("# w_clipper: ", self.w_clipper)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.trainA_0 = ImageFolder(os.path.join('dataset', 'woman', 'trainA'), train_transform, label=0)
        self.trainB_0 = ImageFolder(os.path.join('dataset', 'woman', 'trainB'), train_transform, label=0)
        self.testA_0 = ImageFolder(os.path.join('dataset', 'woman', 'testA'), test_transform, label=0)
        self.testB_0 = ImageFolder(os.path.join('dataset', 'woman', 'testB'), test_transform, label=0)

        self.trainA_loader_0 = DataLoader(self.trainA_0, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader_0 = DataLoader(self.trainB_0, batch_size=self.batch_size, shuffle=True)
        self.testA_loader_0 = DataLoader(self.testA_0, batch_size=1, shuffle=False)
        self.testB_loader_0 = DataLoader(self.testB_0, batch_size=1, shuffle=False)

        self.trainA_1 = ImageFolder(os.path.join('dataset', 'man', 'trainA'), train_transform, label=1)
        self.trainB_1 = ImageFolder(os.path.join('dataset', 'man', 'trainB'), train_transform, label=1)
        self.testA_1 = ImageFolder(os.path.join('dataset', 'man', 'testA'), test_transform, label=1)
        self.testB_1 = ImageFolder(os.path.join('dataset', 'man', 'testB'), test_transform, label=1)

        self.trainA_loader_1 = DataLoader(self.trainA_1, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader_1 = DataLoader(self.trainB_1, batch_size=self.batch_size, shuffle=True)
        self.testA_loader_1 = DataLoader(self.testA_1, batch_size=1, shuffle=False)
        self.testB_loader_1 = DataLoader(self.testB_1, batch_size=1, shuffle=False)

        self.trainA_2 = ImageFolder(os.path.join('dataset', 'kids', 'trainA'), train_transform, label=2)
        self.trainB_2 = ImageFolder(os.path.join('dataset', 'kids', 'trainB'), train_transform, label=2)
        self.testA_2 = ImageFolder(os.path.join('dataset', 'kids', 'testA'), test_transform, label=2)
        self.testB_2 = ImageFolder(os.path.join('dataset', 'kids', 'testB'), test_transform, label=2)

        self.trainA_loader_2 = DataLoader(self.trainA_2, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader_2 = DataLoader(self.trainB_2, batch_size=self.batch_size, shuffle=True)
        self.testA_loader_2 = DataLoader(self.testA_2, batch_size=1, shuffle=False)
        self.testB_loader_2 = DataLoader(self.testB_2, batch_size=1, shuffle=False)

        self.trainA_3 = ImageFolder(os.path.join('dataset', 'old', 'trainA'), train_transform, label=3)
        self.trainB_3 = ImageFolder(os.path.join('dataset', 'old', 'trainB'), train_transform, label=3)
        self.testA_3 = ImageFolder(os.path.join('dataset', 'old', 'testA'), test_transform, label=3)
        self.testB_3 = ImageFolder(os.path.join('dataset', 'old', 'testB'), test_transform, label=3)

        self.trainA_loader_3 = DataLoader(self.trainA_3, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader_3 = DataLoader(self.trainB_3, batch_size=self.batch_size, shuffle=True)
        self.testA_loader_3 = DataLoader(self.testA_3, batch_size=1, shuffle=False)
        self.testB_loader_3 = DataLoader(self.testB_3, batch_size=1, shuffle=False)


        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(ngf=self.ch, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(ngf=self.ch, img_size=self.img_size, light=self.light).to(self.device)

        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        self.facenet = FaceFeatures('pretrained_weights/model_mobilefacenet.pth', self.device)


        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.CE_loss = nn.CrossEntropyLoss().to(self.device)

        """ Trainer """
        genA2B_specific_params, genA2B_shared_params = self.genA2B.get_params()
        genB2A_specific_params, genB2A_shared_params = self.genB2A.get_params()

        self.G_optim_specific = torch.optim.Adam(
            itertools.chain(genA2B_specific_params, genB2A_specific_params),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001
        )
        self.G_optim_share = torch.optim.Adam(
            itertools.chain(genA2B_shared_params, genB2A_shared_params),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001
        )

        self.D_optim = torch.optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters(),
                            self.disLA.parameters(), self.disLB.parameters()),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001
        )

        """ Define Rho clipper to constraint the value of rho in AdaLIN and LIN"""
        self.Rho_clipper = RhoClipper(0, self.rho_clipper)
        self.W_Clipper = WClipper(0, self.w_clipper)

        """Summary writer"""
        self.writer = SummaryWriter(os.path.join(self.result_dir, 'logs'))


    def train(self):
        self.genA2B.train(), self.genB2A.train()
        self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim_share.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.G_optim_specific.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        if self.pretrained_weights:
            params = torch.load(self.pretrained_weights, map_location=self.device)
            load_params(self.genA2B, params['genA2B'])
            load_params(self.genB2A, params['genB2A'])
            # self.genA2B.load_state_dict(params['genA2B'])
            # self.genB2A.load_state_dict(params['genB2A'])
            # self.disGA.load_state_dict(params['disGA'])
            # self.disGB.load_state_dict(params['disGB'])
            # self.disLA.load_state_dict(params['disLA'])
            # self.disLB.load_state_dict(params['disLB'])
            print(" [*] Load {} Success".format(self.pretrained_weights))
            self.genA2B.copy_params()
            self.genB2A.copy_params()
            print(" [*] Init new branches Success")

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim_share.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.G_optim_specific.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            # Woman
            try:
                real_A_0, label_A_0 = trainA_iter_0.next()
            except:
                trainA_iter_0 = iter(self.trainA_loader_0)
                real_A_0, label_A_0 = trainA_iter_0.next()

            try:
                real_B_0, label_B_0 = trainB_iter_0.next()
            except:
                trainB_iter_0 = iter(self.trainB_loader_0)
                real_B_0, label_B_0 = trainB_iter_0.next()

            # Man
            try:
                real_A_1, label_A_1 = trainA_iter_1.next()
            except:
                trainA_iter_1 = iter(self.trainA_loader_1)
                real_A_1, label_A_1 = trainA_iter_1.next()

            try:
                real_B_1, label_B_1 = trainB_iter_1.next()
            except:
                trainB_iter_1 = iter(self.trainB_loader_1)
                real_B_1, label_B_1 = trainB_iter_1.next()

            # Kids
            try:
                real_A_2, label_A_2 = trainA_iter_2.next()
            except:
                trainA_iter_2 = iter(self.trainA_loader_2)
                real_A_2, label_A_2 = trainA_iter_2.next()

            try:
                real_B_2, label_B_2 = trainB_iter_2.next()
            except:
                trainB_iter_2 = iter(self.trainB_loader_2)
                real_B_2, label_B_2 = trainB_iter_2.next()

            # Old
            try:
                real_A_3, label_A_3 = trainA_iter_3.next()
            except:
                trainA_iter_3 = iter(self.trainA_loader_3)
                real_A_3, label_A_3 = trainA_iter_3.next()

            try:
                real_B_3, label_B_3 = trainB_iter_3.next()
            except:
                trainB_iter_3 = iter(self.trainB_loader_3)
                real_B_3, label_B_3 = trainB_iter_3.next()

            real_A_0, real_B_0 = real_A_0.to(self.device), real_B_0.to(self.device)
            label_A_0, label_B_0 = label_A_0.to(self.device), label_B_0.to(self.device)

            real_A_1, real_B_1 = real_A_1.to(self.device), real_B_1.to(self.device)
            label_A_1, label_B_1 = label_A_1.to(self.device), label_B_1.to(self.device)

            real_A_2, real_B_2 = real_A_2.to(self.device), real_B_2.to(self.device)
            label_A_2, label_B_2 = label_A_2.to(self.device), label_B_2.to(self.device)

            real_A_3, real_B_3 = real_A_3.to(self.device), real_B_3.to(self.device)
            label_A_3, label_B_3 = label_A_3.to(self.device), label_B_3.to(self.device)


            # Update D
            self.D_optim.zero_grad()

            fake_A2B_0, fake_A2B_1, fake_A2B_2, fake_A2B_3, _ = \
                self.genA2B.forward_train(real_A_0, real_A_1, real_A_2, real_A_3)
            fake_B2A_0, fake_B2A_1, fake_B2A_2, fake_B2A_3, _ = \
                self.genB2A.forward_train(real_B_0, real_B_1, real_B_2, real_B_3)

            real_A = torch.cat([real_A_0, real_A_1, real_A_2, real_A_3], dim=0)
            real_B = torch.cat([real_B_0, real_B_1, real_B_2, real_B_3], dim=0)

            label_A = torch.cat([label_A_0, label_A_1, label_A_2, label_A_3], dim=0)
            label_B = torch.cat([label_B_0, label_B_1, label_B_2, label_B_3], dim=0)
            label_A = self.label2onehot(label_A, 4)
            label_B = self.label2onehot(label_B, 4)

            real_GA_logit, real_GA_cls, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cls, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cls, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cls, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_B2A = torch.cat([fake_B2A_0, fake_B2A_1, fake_B2A_2, fake_B2A_3], dim=0)
            fake_A2B = torch.cat([fake_A2B_0, fake_A2B_1, fake_A2B_2, fake_A2B_3], dim=0)

            fake_GA_logit, _, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, _, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, _, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, _, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + \
                           self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))

            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + \
                               self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))

            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + \
                           self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))

            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + \
                               self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))

            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + \
                           self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))

            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + \
                               self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))

            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + \
                           self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))

            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + \
                               self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_cls_loss_GA = self.MSE_loss(real_GA_cls, label_A)
            D_cls_loss_LA = self.MSE_loss(real_LA_cls, label_A)
            D_cls_loss_GB = self.MSE_loss(real_GB_cls, label_B)
            D_cls_loss_LB = self.MSE_loss(real_LB_cls, label_B)

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA) + \
                       (D_cls_loss_GA + D_cls_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB) + \
                       (D_cls_loss_GB + D_cls_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            # forward
            fake_A2B_0, fake_A2B_1, fake_A2B_2, fake_A2B_3, fake_A2B_cam_logit = \
                self.genA2B.forward_train(real_A_0, real_A_1, real_A_2, real_A_3)
            fake_B2A_0, fake_B2A_1, fake_B2A_2, fake_B2A_3, fake_B2A_cam_logit = \
                self.genB2A.forward_train(real_B_0, real_B_1, real_B_2, real_B_3)

            fake_A2B2A_0, fake_A2B2A_1, fake_A2B2A_2, fake_A2B2A_3, _ = \
                self.genB2A.forward_train(fake_A2B_0, fake_A2B_1, fake_A2B_2, fake_A2B_3)
            fake_B2A2B_0, fake_B2A2B_1, fake_B2A2B_2, fake_B2A2B_3, _ = \
                self.genA2B.forward_train(fake_B2A_0, fake_B2A_1, fake_B2A_2, fake_B2A_3)

            fake_A2A_0, fake_A2A_1, fake_A2A_2, fake_A2A_3, fake_A2A_cam_logit = \
                self.genB2A.forward_train(real_A_0, real_A_1, real_A_2, real_A_3)
            fake_B2B_0, fake_B2B_1, fake_B2B_2, fake_B2B_3, fake_B2B_cam_logit = \
                self.genA2B.forward_train(real_B_0, real_B_1, real_B_2, real_B_3)

            fake_A2B = torch.cat([fake_A2B_0, fake_A2B_1, fake_A2B_2, fake_A2B_3], dim=0)
            fake_B2A = torch.cat([fake_B2A_0, fake_B2A_1, fake_B2A_2, fake_B2A_3], dim=0)

            fake_A2B2A = torch.cat([fake_A2B2A_0, fake_A2B2A_1, fake_A2B2A_2, fake_A2B2A_3], dim=0)
            fake_B2A2B = torch.cat([fake_B2A2B_0, fake_B2A2B_1, fake_B2A2B_2, fake_B2A2B_3], dim=0)

            fake_A2A = torch.cat([fake_A2A_0, fake_A2A_1, fake_A2A_2, fake_A2A_3], dim=0)
            fake_B2B = torch.cat([fake_B2B_0, fake_B2B_1, fake_B2B_2, fake_B2B_3], dim=0)

            fake_GA_logit, fake_GA_cls, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cls, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cls, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cls, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            # loss G
            # base domain
            G_ad_loss_GA_base = self.MSE_loss(fake_GA_logit[0], torch.ones_like(fake_GA_logit[0]).to(self.device))
            G_ad_cam_loss_GA_base = self.MSE_loss(fake_GA_cam_logit[0], torch.ones_like(fake_GA_cam_logit[0]).to(self.device))
            G_ad_loss_LA_base = self.MSE_loss(fake_LA_logit[0], torch.ones_like(fake_LA_logit[0]).to(self.device))
            G_ad_cam_loss_LA_base = self.MSE_loss(fake_LA_cam_logit[0], torch.ones_like(fake_LA_cam_logit[0]).to(self.device))
            G_ad_loss_GB_base = self.MSE_loss(fake_GB_logit[0], torch.ones_like(fake_GB_logit[0]).to(self.device))
            G_ad_cam_loss_GB_base = self.MSE_loss(fake_GB_cam_logit[0], torch.ones_like(fake_GB_cam_logit[0]).to(self.device))
            G_ad_loss_LB_base = self.MSE_loss(fake_LB_logit[0], torch.ones_like(fake_LB_logit[0]).to(self.device))
            G_ad_cam_loss_LB_base = self.MSE_loss(fake_LB_cam_logit[0], torch.ones_like(fake_LB_cam_logit[0]).to(self.device))

            G_recon_loss_A_base = self.L1_loss(fake_A2B2A[0], real_A[0])
            G_recon_loss_B_base = self.L1_loss(fake_B2A2B[0], real_B[0])

            G_identity_loss_A_base = self.L1_loss(fake_A2A[0], real_A[0])
            G_identity_loss_B_base = self.L1_loss(fake_B2B[0], real_B[0])

            G_id_loss_A_base = self.facenet.cosine_distance(real_A[0].unsqueeze(0), fake_A2B[0].unsqueeze(0)).mean()
            G_id_loss_B_base = self.facenet.cosine_distance(real_B[0].unsqueeze(0), fake_B2A[0].unsqueeze(0)).mean()

            G_loss_A_base = self.adv_weight * (G_ad_loss_GA_base + G_ad_cam_loss_GA_base + G_ad_loss_LA_base + G_ad_cam_loss_LA_base) + \
                            self.cycle_weight * G_recon_loss_A_base + self.identity_weight * G_identity_loss_A_base + \
                            self.faceid_weight * G_id_loss_A_base
            G_loss_B_base = self.adv_weight * (G_ad_loss_GB_base + G_ad_cam_loss_GB_base + G_ad_loss_LB_base + G_ad_cam_loss_LB_base) + \
                            self.cycle_weight * G_recon_loss_B_base + self.identity_weight * G_identity_loss_B_base + \
                            self.faceid_weight * G_id_loss_B_base

            Generator_loss_base = G_loss_A_base + G_loss_B_base

            # new domain
            G_ad_loss_GA_new = self.MSE_loss(fake_GA_logit[1:], torch.ones_like(fake_GA_logit[1:]).to(self.device))
            G_ad_cam_loss_GA_new = self.MSE_loss(fake_GA_cam_logit[1:], torch.ones_like(fake_GA_cam_logit[1:]).to(self.device))
            G_ad_loss_LA_new = self.MSE_loss(fake_LA_logit[1:], torch.ones_like(fake_LA_logit[1:]).to(self.device))
            G_ad_cam_loss_LA_new = self.MSE_loss(fake_LA_cam_logit[1:], torch.ones_like(fake_LA_cam_logit[1:]).to(self.device))
            G_ad_loss_GB_new = self.MSE_loss(fake_GB_logit[1:], torch.ones_like(fake_GB_logit[1:]).to(self.device))
            G_ad_cam_loss_GB_new = self.MSE_loss(fake_GB_cam_logit[1:], torch.ones_like(fake_GB_cam_logit[1:]).to(self.device))
            G_ad_loss_LB_new = self.MSE_loss(fake_LB_logit[1:], torch.ones_like(fake_LB_logit[1:]).to(self.device))
            G_ad_cam_loss_LB_new = self.MSE_loss(fake_LB_cam_logit[1:], torch.ones_like(fake_LB_cam_logit[1:]).to(self.device))

            G_recon_loss_A_new = self.L1_loss(fake_A2B2A[1:], real_A[1:])
            G_recon_loss_B_new = self.L1_loss(fake_B2A2B[1:], real_B[1:])

            G_identity_loss_A_new = self.L1_loss(fake_A2A[1:], real_A[1:])
            G_identity_loss_B_new = self.L1_loss(fake_B2B[1:], real_B[1:])

            G_id_loss_A_new = self.facenet.cosine_distance(real_A[1:], fake_A2B[1:]).mean()
            G_id_loss_B_new = self.facenet.cosine_distance(real_B[1:], fake_B2A[1:]).mean()

            G_loss_A_new = self.adv_weight * (G_ad_loss_GA_new + G_ad_cam_loss_GA_new + G_ad_loss_LA_new + G_ad_cam_loss_LA_new) + \
                           self.cycle_weight * G_recon_loss_A_new + self.identity_weight * G_identity_loss_A_new + \
                           self.faceid_weight * G_id_loss_A_new
            G_loss_B_new = self.adv_weight * (G_ad_loss_GB_new + G_ad_cam_loss_GB_new + G_ad_loss_LB_new + G_ad_cam_loss_LB_new) + \
                           self.cycle_weight * G_recon_loss_B_new + self.identity_weight * G_identity_loss_B_new + \
                           self.faceid_weight * G_id_loss_B_new

            Generator_loss_new = G_loss_A_new + G_loss_B_new

            # cls loss
            G_cls_loss_GA = self.MSE_loss(fake_GA_cls, label_A)
            G_cls_loss_LA = self.MSE_loss(fake_LA_cls, label_A)
            G_cls_loss_GB = self.MSE_loss(fake_GB_cls, label_B)
            G_cls_loss_LB = self.MSE_loss(fake_LB_cls, label_B)

            # cam loss
            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + \
                           self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + \
                           self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_cam_loss = 0.5 * (G_cam_loss_A + G_cam_loss_B) * self.cam_weight
            G_cls_loss = 0.25 * (G_cls_loss_GA + G_cls_loss_LA + G_cls_loss_GB + G_cls_loss_LB) * self.cls_weight

            self.G_optim_share.zero_grad()
            self.G_optim_specific.zero_grad()
            loss_shared = G_cam_loss + Generator_loss_base
            loss_shared.backward(retain_graph=True)
            self.G_optim_share.step()
            self.G_optim_specific.step()

            self.G_optim_share.zero_grad()
            self.G_optim_specific.zero_grad()
            loss_specific = G_cls_loss + Generator_loss_new
            loss_specific.backward()
            self.G_optim_specific.step()

            # clip parameter of Soft-AdaLIN and LIN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)
            
            self.genA2B.apply(self.W_Clipper)
            self.genB2A.apply(self.W_Clipper)

            if step % 10 == 0:
                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss_base: %.8f, g_loss_new: %.8f,cam_loss: %.8f, cls_loss: %.8f" \
                      % (step, self.iteration, time.time() - start_time, \
                         Discriminator_loss, Generator_loss_base, Generator_loss_new, G_cam_loss, G_cls_loss))
                self.writer.add_scalar('Generator loss base', Generator_loss_base.item(), step)
                self.writer.add_scalar('Generator loss new', Generator_loss_new.item(), step)
                self.writer.add_scalar('Discriminator loss', Discriminator_loss.item(), step)
                self.writer.add_scalar('CAM loss', G_cam_loss.item(), step)
                self.writer.add_scalar('Classification loss', G_cls_loss.item(), step)

            if step % self.print_freq == 0:
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval()
                self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()

                with torch.no_grad():
                    for _ in range(test_sample_num):
                        try:
                            real_A_1, label_A_1 = testA_iter.next()
                        except:
                            testA_iter = iter(self.testA_loader_1)
                            real_A_1, label_A_1 = testA_iter.next()

                        try:
                            real_B_1, label_B_1 = testB_iter.next()
                        except:
                            testB_iter = iter(self.testB_loader_1)
                            real_B_1, label_B_1 = testB_iter.next()
                        real_A_1, real_B_1 = real_A_1.to(self.device), real_B_1.to(self.device)
                        label_A_1, label_B_1 = label_A_1.to(self.device), label_B_1.to(self.device)

                        fake_A2B_man, _, fake_A2B_heatmap = self.genA2B(real_A_1, label_A_1)
                        fake_B2A_man, _, fake_B2A_heatmap = self.genB2A(real_B_1, label_B_1)

                        fake_A2B2A_man, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B_man, label_B_1)
                        fake_B2A2B_man, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A_man, label_A_1)

                        fake_A2A_man, _, fake_A2A_heatmap = self.genB2A(real_A_1, label_A_1)
                        fake_B2B_man, _, fake_B2B_heatmap = self.genA2B(real_B_1, label_B_1)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A_1[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A_man[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B_man[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A_man[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B_1[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B_man[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A_man[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B_man[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train()
                self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.hdfs_root, self.experiment_name, 'model'), step)


    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        save_hdfs(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))


    def load(self, dir, step):
        params = load_hdfs(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        # self.disGA.load_state_dict(params['disGA'])
        # self.disGB.load_state_dict(params['disGB'])
        # self.disLA.load_state_dict(params['disLA'])
        # self.disLB.load_state_dict(params['disLB'])

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim).type_as(labels)
        out[np.arange(batch_size), labels.long()] = 1
        return out.float()