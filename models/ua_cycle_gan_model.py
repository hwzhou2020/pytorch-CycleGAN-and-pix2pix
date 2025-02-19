import os
import torch
import itertools
from util.image_pool import ImagePool
from util.util import cal_uncertainty, cal_ua_range
from .base_model import BaseModel
from . import networks


class UACycleGANModel(BaseModel):
    """
    This class implements the Uncertainty-aware UA-CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        # parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'fake_alpha1_B', 'fake_alpha2_B', 'fake_beta_B', 'rec_A', 'rec_alpha1_A', 'rec_alpha2_A', 'rec_beta_A']
        visual_names_B = ['real_B', 'fake_A', 'fake_alpha1_A', 'fake_alpha2_A', 'fake_beta_A', 'rec_B', 'rec_alpha1_B', 'rec_alpha2_B', 'rec_beta_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        visual_names_A.append('ua_B')
        visual_names_B.append('ua_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        load_pretrain = opt.load_pretrain  
        if load_pretrain == True:
            model_init_path = opt.pretrain_path
            # Initialize the weights for netG_A
            Init_netG_A_state_dict = torch.load(os.path.join(model_init_path, 'Init_net_G_A.pth'))
            netG_A_dict = self.netG_A.state_dict()
            filtered_Init_netG_A_dict = {('module.' + k): v for k, v in Init_netG_A_state_dict.items() if ('module.' + k) in netG_A_dict}
            for k, v in Init_netG_A_state_dict.items():
                k_parse = k.split('.')
                if k_parse[-3] == 'conv_block' and int(k_parse[-2]) == 5:
                    k_parse[-2] = '6'
                    k = '.'.join(k_parse)
                    filtered_Init_netG_A_dict['module.' + k] = v

            filtered_Init_netG_A_dict['module.image.0.weight'] = Init_netG_A_state_dict['model.26.weight']
            filtered_Init_netG_A_dict['module.image.0.bias'] = Init_netG_A_state_dict['model.26.bias']
            netG_A_dict.update(filtered_Init_netG_A_dict)
            self.netG_A.load_state_dict(netG_A_dict)
            # Initialize the weights for netG_B
            Init_netG_B_state_dict = torch.load(os.path.join(model_init_path, 'Init_net_G_B.pth'))
            netG_B_dict = self.netG_B.state_dict()
            filtered_Init_netG_B_dict = {('module.' + k): v for k, v in Init_netG_B_state_dict.items() if ('module.' + k) in netG_B_dict}
            for k, v in Init_netG_B_state_dict.items():
                k_parse = k.split('.')
                if k_parse[-3] == 'conv_block' and int(k_parse[-2]) == 5:
                    k_parse[-2] = '6'
                    k = '.'.join(k_parse)
                    filtered_Init_netG_B_dict['module.' + k] = v

            filtered_Init_netG_B_dict['module.image.0.weight'] = Init_netG_B_state_dict['model.26.weight']
            filtered_Init_netG_B_dict['module.image.0.bias'] = Init_netG_B_state_dict['model.26.bias']
            netG_B_dict.update(filtered_Init_netG_B_dict)
            self.netG_B.load_state_dict(netG_B_dict)

            # print(Init_netG_A_state_dict.keys())
            # print(' ')
            # print(netG_A_dict.keys())
            # print(' ')
            # print(filtered_Init_netG_A_dict.keys())
            # print(filtered_Init_netG_A_dict['module.model.11.conv_block.6.weight'])
            print('Message: pretrained weights loaded')

            self.freeze = opt.freeze
            if self.freeze == True:
                # Freeze the weights for netG_A except for alpha1, alpha2, and beta layers
                for name, param in self.netG_A.named_parameters():
                    if 'alpha1' in name or 'alpha2' in name or 'beta' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                # Freeze the weights for netG_B except for alpha1, alpha2, and beta layers
                for name, param in self.netG_B.named_parameters():
                    if 'alpha1' in name or 'alpha2' in name or 'beta' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                # for name, param in self.netG_A.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.size())
                print('Message: pretrained weights frozen')

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            if load_pretrain == True:
                model_init_path = opt.pretrain_path
                # Initialize the weights for netD_A
                Init_netD_A_state_dict = torch.load(os.path.join(model_init_path, 'Init_net_D_A.pth'))
                netD_A_dict = self.netD_A.state_dict()
                filtered_Init_netD_A_dict = {('module.' + k): v for k, v in Init_netD_A_state_dict.items() if ('module.' + k) in netD_A_dict}
                netD_A_dict.update(filtered_Init_netD_A_dict)
                self.netD_A.load_state_dict(netD_A_dict)
                # Initialize the weights for netD_B
                Init_netD_B_state_dict = torch.load(os.path.join(model_init_path, 'Init_net_D_B.pth'))
                netD_B_dict = self.netD_B.state_dict()
                filtered_Init_netD_B_dict = {('module.' + k): v for k, v in Init_netD_B_state_dict.items() if ('module.' + k) in netD_B_dict}
                netD_B_dict.update(filtered_Init_netD_B_dict)
                self.netD_B.load_state_dict(netD_B_dict)

                if self.freeze == True:
                    # Freeze the weights for netD_A
                    for param in self.netD_A.parameters():
                        param.requires_grad = False
                    # Freeze the weights for netD_B
                    for param in self.netD_B.parameters():
                        param.requires_grad = False
            
            self.up_bound = opt.up_bound
            self.low_bound = opt.low_bound
            self.ua_range, self.aggd_max = cal_ua_range(self.up_bound, self.low_bound)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = networks.UncertaintyLoss().to(self.device) # torch.nn.L1Loss()
            if self.freeze == True:
                self.criterionIdt = networks.UncertaintyLoss().to(self.device) 
            else:
                self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.freeze == False:
                self.optimizers.append(self.optimizer_D)


        if self.isTrain and self.freeze:
            self.mc_dropout = opt.mc_dropout
            self.mc_dropout_mode = opt.mc_dropout_mode
            self.mc_dropout_rate = opt.mc_dropout_rate

            if self.mc_dropout and self.mc_dropout_mode == 'constant':
                for layer in self.netG_A.modules():
                    if isinstance(layer, torch.nn.Dropout):
                        layer.p = self.mc_dropout_rate
                for layer in self.netG_B.modules():
                    if isinstance(layer, torch.nn.Dropout):
                        layer.p = self.mc_dropout_rate

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.freeze:
            if self.mc_dropout:
                # Activate dropout layers in pretrained generators
                self.netG_A.apply(lambda m: m.train() if isinstance(m, torch.nn.Dropout) else None)
                self.netG_B.apply(lambda m: m.train() if isinstance(m, torch.nn.Dropout) else None)

                # for module in self.netG_A.modules():
                #     # print(module)
                #     if isinstance(module, torch.nn.Dropout):
                #         print(module)
                #         print("Dropout p =", module.p)
                #         print("Training mode =", module.training) 
 

        self.fake_B, self.fake_alpha1_B, self.fake_alpha2_B, self.fake_beta_B = self.netG_A(self.real_A)   # G_A(A)
        self.rec_A, self.rec_alpha1_A, self.rec_alpha2_A, self.rec_beta_A = self.netG_B(self.fake_B)       # G_B(G_A(A))
        self.fake_A, self.fake_alpha1_A, self.fake_alpha2_A, self.fake_beta_A = self.netG_B(self.real_B)   # G_B(B)
        self.rec_B, self.rec_alpha1_B, self.rec_alpha2_B, self.rec_beta_B = self.netG_A(self.fake_A)       # G_A(G_B(B))

        self.ua_B = cal_uncertainty(self.fake_alpha1_B, self.fake_alpha2_B, self.fake_beta_B, self.ua_range)
        self.ua_A = cal_uncertainty(self.fake_alpha1_A, self.fake_alpha2_A, self.fake_beta_A, self.ua_range)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        if self.freeze:
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.idt_A, idt_alpha1_A, idt_alpha1_B, idt_beta_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, idt_alpha1_A, idt_alpha1_B, idt_beta_A, self.real_B, 
                                                    up_bound = self.up_bound, low_bound = self.low_bound, aggd_max = self.aggd_max) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B, idt_alpha1_A, idt_alpha1_B, idt_beta_A = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, idt_alpha1_A, idt_alpha1_B, idt_beta_A, self.real_A,
                                                    up_bound = self.up_bound, low_bound = self.low_bound, aggd_max = self.aggd_max) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0

            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.rec_alpha1_A, self.rec_alpha2_A, self.rec_beta_A, self.real_A,
                                                    up_bound = self.up_bound, low_bound = self.low_bound, aggd_max = self.aggd_max) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.rec_alpha1_B, self.rec_alpha2_B, self.rec_beta_B, self.real_B,
                                                    up_bound = self.up_bound, low_bound = self.low_bound, aggd_max = self.aggd_max) * lambda_B 
            # combined loss and calculate gradients
            self.loss_G = self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

            self.loss_G_A = 0
            self.loss_G_B = 0
            self.loss_D_A = 0
            self.loss_D_B = 0
            self.loss_G.backward()

        else: 
                
            # Identity loss
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                self.idt_A, _, _, _ = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B, _, _, _ = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0

            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
            # Forward cycle loss || G_B(G_A(A)) - A||
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.rec_alpha1_A, self.rec_alpha2_A, self.rec_beta_A, self.real_A,
                                                    up_bound = self.up_bound, low_bound = self.low_bound) * lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.rec_alpha1_B, self.rec_alpha2_B, self.rec_beta_B, self.real_B,
                                                    up_bound = self.up_bound, low_bound = self.low_bound) * lambda_B 
            # combined loss and calculate gradients
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
            self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        if self.freeze:
            self.forward()      # compute fake images and reconstruction images.
            # G_A and G_B
            # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights
        else:
            # forward
            self.forward()      # compute fake images and reconstruction images.
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A()      # calculate gradients for D_A
            self.backward_D_B()      # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights
