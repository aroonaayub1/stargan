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
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
import cv2

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, CompCars_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader      #Dataloader for CelebA dataset
        self.rafd_loader = rafd_loader          #Dataloader for RaFD dataset
        self.CompCars_loader = CompCars_loader  #Dataloader for CompCars dataset

        # Model configurations.
        self.c_dim = config.c_dim            # dimension of domain labels (1st dataset) 
        self.c2_dim = config.c2_dim          # dimension of domain labels (2nd dataset)
        self.image_size = config.image_size  # image resolution
        self.g_conv_dim = config.g_conv_dim  # number of conv filters in the first layer of G
        self.d_conv_dim = config.d_conv_dim  # number of conv filters in the first layer of D
        self.g_repeat_num = config.g_repeat_num # number of residual blocks in G
        self.d_repeat_num = config.d_repeat_num # number of residual blocks in D
        self.lambda_cls = config.lambda_cls  # weight for domain classification loss
        self.lambda_rec = config.lambda_rec  # weight for reconstruction loss
        self.lambda_gp = config.lambda_gp    # weight for gradient penalty

        # Training configurations.
        self.dataset = config.dataset        # Dataset name
        self.batch_size = config.batch_size  # Mini-Batch Size
        self.num_iters = config.num_iters    # Number of total iterations for training D
        self.num_iters_decay = config.num_iters_decay # number of iterations for decaying lr
        self.g_lr = config.g_lr              # learning rate for G
        self.d_lr = config.d_lr              # learning rate for D
        self.n_critic = config.n_critic      # number of D updates per each G update
        self.beta1 = config.beta1            # beta1 for Adam optimizer
        self.beta2 = config.beta2            # beta2 for Adam optimizer
        self.resume_iters = config.resume_iters # resume training from this step
        self.selected_attrs = config.selected_attrs # selected attributes for the dataset

        # Test configurations.
        self.test_iters = config.test_iters  # test model from this step
        self.attr_path = config.attr_path

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard       #Use tensorboard to display results
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    #Select device GPU-0 or CPU

        # Directories.
        self.log_dir = config.log_dir               # Directory for saving training logs  
        self.sample_dir = config.sample_dir         # Directory for saving training time samples
        self.model_save_dir = config.model_save_dir # Directory for saving training model checkpoints
        self.result_dir = config.result_dir         # Directory for saving test results

        # Step size.
        self.log_step = config.log_step             # Step size for saving logs
        self.sample_step = config.sample_step       # Step size for saving samples
        self.model_save_step = config.model_save_step# Step size for saving model checkpoints
        self.lr_update_step = config.lr_update_step # Step size for update learning rate

        # Build the model and tensorboard.
        if self.use_tensorboard:
            self.build_tensorboard()    # Save tensorboard graph (For information display on tensorboard)
        self.build_model()              # Build the G and D model
        
    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD', "CompCars"]:  #Used for single dataset
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)                      # Generator Network
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) # Discriminator Network
            
        elif self.dataset in ['Both']:                      # Used for both, CelebA and RaFD dataset
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        # if True:
        #     self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        #     self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
            
   
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])   # Optimizer for Generator
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])   # Optimizer for Discriminator
        self.print_network(self.G, 'G') # Print Generator Network Details
        self.print_network(self.D, 'D') # Print Discriminator Network Details
            
        self.G.to(self.device)  # Move the Generator network to the specified device (GPU-0 or CPU)
        self.D.to(self.device)  # Move the Discriminator network to the specified device (GPU-0 or CPU)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():    # Run for all parameters in the model
            num_params += p.numel()     # Accumulate the number of parameters in whole model
        print(model)                    # Print Model network details
        print(name)                     # Print Model name
        print("The number of parameters: {}".format(num_params)) # Print total number of parameters

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        # Used if we are resuming training from pre-trained model.
        print('Loading the trained models from step {}...'.format(resume_iters))
        
        # Defining the path for models
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))    # Path for loading Generator network 
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))    # Path for loading Discriminator network 
        
        # Loading networks frrom specified paths
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage)) # Loading the Generator network
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage)) # Loading the Discriminator network

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)
        
        
        # #For saving Tensorboard Graph
        # from torch.utils.tensorboard import SummaryWriter
        # self.tf_writerD=SummaryWriter(self.log_dir+"_summaryWriter_G") 
        # self.tf_writerG=SummaryWriter(self.log_dir+"_summaryWriter_D") 

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr    # Update the learning rate for Generator using its optimizer 
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr    # Update the learning rate for Discriminator using its optimizer 

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()    # Resets the gradient for Generator to [ ]. (Initialization)
        self.d_optimizer.zero_grad()    # Resets the gradient for Discriminator to [ ]. (Initialization)

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
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)
                    
        if dataset == 'CompCars':
            car_side_indices = []
            for i, attr_name in enumerate(selected_attrs):
                # if attr_name in ["uncertain", "front", "rear", "side", "frontside", "rearside"]:
                if attr_name in ["uncertain","front","rear","side","frontside","rearside","Uncertain","MPV","SUV","sedan","hatchback","minibus","fastback","estate","pickup","hardtop","sports","crossover","convertible","metallic_black","gray","blue","cyan","silver","green","red","orange","pink","purple","yellow","white"]:
                    car_side_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
                    
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
                
            elif dataset == 'CompCars':
                c_trg = c_org.clone()
                if i in car_side_indices:  # Set one side to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in car_side_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA' or 'CompCars':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'CompCars':
            data_loader = self.CompCars_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)    # Returns Images, Labels (Fixed for making same samples at certain iterations)
        x_fixed = x_fixed.to(self.device)   # Move images to GPU
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs) #Define labels in 0,1 format

        # Learning rate cache for decaying.
        g_lr = self.g_lr    # Initial learning rate for Generator
        d_lr = self.d_lr    # Initial learning rate for Discrimintaor

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters         # Resume training from this iteration no. Defined by user.
            self.restore_model(self.resume_iters)   # Restore model state dictionary from saved checkpoint

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)     # Return 4 images and their corresponding labels
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))    # Shuffles the input images index sequence (inside the mini-batch)
            label_trg = label_org[rand_idx]                 # Shuffle the labels for input images according to new sequence

            if self.dataset == 'CelebA' or "CompCars":      # CompCars attributes were defined in same way as CelebA
                c_org = label_org.clone()
                c_trg = label_trg.clone()                   # Clone labels to create an independent copy of labels
            elif self.dataset == 'RaFD':                    # Sometimes changing in assigned variable changes values in org variable as well
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.                               # Move input images to GPU
            
            c_org = c_org.to(self.device)             # Original domain labels.                     # Cloned copy of input labels  - Move to GPU
            c_trg = c_trg.to(self.device)             # Target domain labels.                       # Cloned copy of target labels - Move to GPU
            
            label_org = label_org.to(self.device)     # Labels for computing classification loss.   # input labels  - used for computing loss
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.   # target labels - used for computing loss
            

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)   # Input image -> Discrimintaor ---> Return src and class for given No of Images in batch
            d_loss_real = - torch.mean(out_src) # Calculate real loss for discrimintaor
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset) #Calculate classification loss for discrimintaor

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)      # Input Images , target labels -> Generator -----> Returns fake images
            out_src, out_cls = self.D(x_fake.detach())  # Discriminator tries to discriminate fake images. Returns src and class for fake images.
            d_loss_fake = torch.mean(out_src)   # calculate fake loss for discrimintaor

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)     #Gradient multiplier
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)  # some image
            out_src, _ = self.D(x_hat)  # Discrimnator tries to discrimintate some image
            d_loss_gp = self.gradient_penalty(out_src, x_hat)   # Calculate gradient policy loss

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp  #Total Discriminator Loss
            self.reset_grad()   # Resets gradients - Otherwise the gradient keeps on accumulating
            d_loss.backward()   # Builtin function - Computes the gradient according to loss value
            self.d_optimizer.step() # performs a parameter update based on the current gradient 

            # Logging all losses
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:      # n_critic = number of D updates per each G update
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)      # Generate fake images using real images and target labels
                out_src, out_cls = self.D(x_fake)   # Discrimators discriminates fake images and returns src and class of fake images
                g_loss_fake = - torch.mean(out_src) # Calculate fake image loss
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset) #Calculate fake image's class loss

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)   # Use fake image as input and generate the original image - Reconstruction
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))  # Calculate reconstruction loss

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls  #Calculate total loss
                self.reset_grad()   # Resets gradients - Otherwise the gradient keeps on accumulating
                g_loss.backward()   # Builtin function - Computes the gradient according to loss value
                self.g_optimizer.step() # performs a parameter update based on the current gradient 

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log) #Print information for iteration

                if self.use_tensorboard: #Log results on tensorboard
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
                        
                        #For saving Tensorboard Graph
                        # self.tf_writer.add_scalar(tag, value, i+1)

            # Translate fixed images for debugging. Save real and their corresponding fake images using fixed/same images.
            if (i+1) % self.sample_step == 0:
            # if (i+1) % 50 == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))
                    
                    # #For saving Tensorboard Graph
                    # import torchvision
                    # grid = torchvision.utils.make_grid(x_concat.data.cpu())
                    # self.tf_writerD.add_image('x_fixed c_fixed', grid, 0)
                    # self.tf_writerD.add_graph(self.G , (x_fixed, c_fixed), 0)
                    # self.tf_writerG.add_graph(self.D , x_real, 0)
                    
                    
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay/Update learning rates. 
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

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
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
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

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
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

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)
                            #For saving Tensorboard Graph
                            # self.tf_writer.add_scalar(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'CompCars':
            data_loader = self.CompCars_loader
        
        with torch.no_grad():
            for i, (x_real, c_org, fname) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                
                debug=False
                if debug:
                    from PIL import Image
                    a = Image.fromarray(x_real[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    a.save("test2.jpg")

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))
                    
                # # Save the translated images.
                # x_concat = torch.cat(x_fake_list, dim=3) # dimension = channel
                # result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                # print('Saved real and fake images into {}...'.format(result_path))
                
                # Modified Code - Aroona
                y = {"0" : "metallic_black", "1" : "blue" , "2" : "silver", "3" : "green", "4" : "red", "5" : "yellow", "6" : "white" }   
                for x in range(len(x_fake_list)):
                    if x==0:
                        continue
                    key = x
                    
                    # Save patches
                    
                    #Path for outputs
                    if "Val" in self.attr_path:
                        pdata = "/data/BDD/BDD100k/images/100k/val"
                        odir = os.path.join(self.result_dir,'Val',y[str(key-1)])
                        outPath = os.path.join(self.result_dir, "generated_patches","Val")
                    if "Train" in self.attr_path:
                        pdata = "/data/BDD/BDD100k/images/100k/train"
                        odir = os.path.join(self.result_dir,'Train',y[str(key-1)])
                        outPath = os.path.join(self.result_dir, "generated_patches","Train")
                    if not os.path.exists(outPath): os.makedirs(outPath) 
                    if not os.path.exists(odir): os.makedirs(odir) 
                    
                    # Patch file
                    result_path = os.path.join(outPath, fname[0][:-4] + '-' + y[str(key-1)] +'.jpg')
                    
                    # Patch 
                    x_concat = torch.cat(x_fake_list[x:x+1], dim=3) # dimension = channel 
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))
                    
                    # Check if more than 1 object in an image
                    pfile = fname[0].split('_')[0]+".jpg"
                    if os.path.exists(os.path.join(odir, pfile)):
                        pimg = os.path.join(odir, pfile)
                    else:
                        pimg = os.path.join(pdata, pfile)
                    
                    # Output Image (Modified)
                    ofile = os.path.join(odir,pfile)
                    
                    # Read image
                    orgIm = cv2.imread(pimg)
                    
                    # Read labels
                    plab = os.path.join( os.path.dirname(self.attr_path) , fname[0][:-4]+'.txt')
                    [X,Y,W,H] = [int(xx) for xx in open(plab,'r').readlines()[0].split(' ')]
                    
                    # Read patch
                    patch = cv2.imread(result_path)
                    
                    # Resize patch
                    hh,ww,_ = orgIm.shape
                    [Y1,Y2,X1,X2] = [Y, min(hh,Y+H), X, min(ww,X+W)]
                    resiz = cv2.resize(patch,(X2-X1,Y2-Y1))
                    
                    # Replace patch in the image
                    nImg = orgIm
                    nImg[Y1:Y2,X1:X2] = resiz
                    
                    # Save new image
                    cv2.imwrite(ofile,nImg)
                    
                    
                

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))