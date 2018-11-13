from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, edges2shoes_A_loader, edges2shoes_B_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.edges2shoes_A_loader = edges2shoes_A_loader
        self.edges2shoes_B_loader = edges2shoes_B_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['edges2shoes']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='edges2shoes'):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'edges2shoes':
                c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='edges2shoes'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'edges2shoes':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def train(self):
        # Set data loader.
        if self.dataset == 'edges2shoes':
            data_loader_A = self.edges2shoes_A_loader
            data_loader_B = self.edges2shoes_B_loader

        # Fetch fixed inputs for debugging.

        if self.dataset == 'edges2shoes':  # data_iter_A : edge img ,data_laoder_B : shoes img
            data_iter_A = iter(data_loader_A)
            data_iter_B = iter(data_loader_B)
            x_edges_, c_org_edges = next(data_iter_A)
            x_shoes_, c_org_shoes = next(data_iter_B)
            x_edges_ = x_edges_.to(self.device)
            x_shoes_ = x_shoes_.to(self.device)
            c_edges_list = self.create_labels(c_org_edges, self.c_dim, self.dataset)  # 선택한 cat의 label list
            c_shoes_list = self.create_labels(c_org_shoes, self.c_dim, self.dataset)  # 선택한 cat의 label list

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            if self.dataset == 'edges2shoes':
                try:
                    x_edges, label_org_edges = next(data_iter_A)
                    x_shoes, label_org = next(data_iter_B)
                except:
                    data_iter_A = iter(data_loader_A)
                    data_iter_B = iter(data_loader_B)
                    x_edges, label_org_edges = next(data_iter_A)
                    x_shoes, label_org = next(data_iter_B)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))  # randperm : 크기만큼의 수를 랜덤하게 배치 ex) 10이면 0~9가 10개의 배열에 랜덤하게 들어감
            label_trg = label_org[rand_idx]  # label_trg : label_org에 있는 label list들 중에서 random하게 뿌린 label
            # print(label_org)

            if self.dataset == 'edges2shoes':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            #print(c_org)
            # 이 부분의 D와 G의 Input을 나눠줘야함
            if self.dataset == 'edges2shoes':  # G에는 x_edges, D에는 x_shoes
                x_edges = x_edges.to(self.device)
                x_shoes = x_shoes.to(self.device)

            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(self.device)  # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # Compute loss with real images.
            if self.dataset == 'edges2shoes':  # G에는 x_edges, D에는 x_shoes
                out_src, out_cls = self.D(x_shoes)
                d_loss_real = - torch.mean(out_src)  # real or fake 판별
                d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)  # class 분류 판별
                # print(x_edges.shape)
                x_fake = self.G(x_edges, c_trg)
                out_src, out_cls = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                alpha = torch.rand(x_shoes.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_shoes.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if self.dataset == 'edges2shoes':  # G에는 x_edges, D에는 x_shoes
                if (i + 1) % self.n_critic == 0:
                    x_fake = self.G(x_edges, c_trg)  # edge 이미지로 fake 이미지 생성
                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = -torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, c_org, self.dataset)

                    x_reconst = self.G(x_fake, c_org)  # 다시 edge로 만들때는 label을 0,0,0,0,0
                    g_loss_rec = torch.mean(torch.abs(x_edges - x_reconst))  # reconstruct이미지는 edge이미지가 나와야함

                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i + 1)

            if self.dataset == 'edges2shoes':
                if (i + 1) % self.sample_step == 0:
                    with torch.no_grad():
                        x_edges_list = [x_edges_]
                        for c_edges in c_edges_list:
                            x_edges_list.append(self.G(x_edges_, c_edges))
                        x_concat = torch.cat(x_edges_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            '''
            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - 150000):
                g_lr /= float(self.num_iters_decay)
                d_lr /= float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            '''

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - 150000):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'edges2shoes':
            data_loader = self.edges2shoes_A_loader
            print('finished loader')

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):  # x_real은 이미지 c_org는 라벨

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
