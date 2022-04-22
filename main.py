import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings


        
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader. Loads images and attributes
    celeba_loader   = None
    rafd_loader     = None
    CompCars_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    if config.dataset in ['CompCars']:
        CompCars_loader = get_loader(config.CompCars_image_dir, config.attr_path, config.selected_attrs,
                                 config.CompCars_crop_size, config.image_size, config.batch_size,
                                 'CompCars', config.mode, config.num_workers)
    
            
    # Create an instance of "Solver" for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, CompCars_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD', 'CompCars']:
            solver.train()          # Train on single dataset
        elif config.dataset in ['Both']:
            solver.train_multi()    # Train on multi dataset (CelebA and RaFD)
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD', 'CompCars']:
            solver.test()           # Test on single dataset
        elif config.dataset in ['Both']:
            solver.test_multi()     # Test on multi dataset (CelebA and RaFD)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--CompCars_crop_size', type=int, default=178, help='crop size for the CompCars dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both', 'CompCars'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=2000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=1900000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--CompCars_image_dir', type=str, default='/data/CompCars_Images')
    parser.add_argument('--celeba_image_dir', type=str, default='/home/msis/Aroona/Codes/stargan/data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='/home/msis/Aroona/Codes/stargan/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    
    config = parser.parse_args()
    
    # parser.add_argument('--custom-data', type=str, default='CompCars')
    
    
    # if config.custom_data == 'CompCars' and config.dataset=="CompCars":
    #     config.CompCars_image_dir   = "/data/CompCars_Images"
    #     config.attr_path            = "/home/msis/Aroona/Codes/stargan/data/CompCars/list_attr_CompCars.txt"
    # elif config.custom_data == 'BDD_Val_Cropped6':
    #     config.CompCars_image_dir   = "/home/msis/Aroona/Codes/stargan/data/BDD_Val_Cropped6/crops/car"
    #     config.attr_path            = "/home/msis/Aroona/Codes/stargan/data/BDD_Val_Cropped6/crops/car/list.txt"
    # elif config.custom_data == 'HACKATHON':
    #     config.CompCars_image_dir   = "/home/msis/Aroona/Codes/stargan/data/HACKATHON/images"
    #     config.attr_path            = "/home/msis/Aroona/Codes/stargan/data/HACKATHON/list_images.txt"
    # elif config.custom_data == 'CompCars_svData':
    #     config.CompCars_image_dir   = "/home/msis/Aroona/Codes/stargan/data/CompCars_SvData/images"
    #     config.attr_path            = "/home/msis/Aroona/Codes/stargan/data/CompCars_SvData/list.txtt"
        
        
        
    # if config.dataset=="CompCars":
    #     parser.add_argument('--CompCars_crop_size', type=int, default=1024, help='crop size for the CelebA dataset')
    #     parser.add_argument('--image_size', type=int, default=256, help='image resolution')
        
    #     # parser.add_argument('--CompCars_image_dir', type=str, default='/data/CompCars_Images')
    #     # parser.add_argument('--attr_path', type=str, default='data/CompCars/list_attr_CompCars.txt')
        
    #     parser.add_argument('--CompCars_image_dir', type=str, default='/home/msis/Aroona/Codes/stargan/data/BDD_Val_Cropped6/crops/car')
    #     parser.add_argument('--attr_path', type=str, default='/home/msis/Aroona/Codes/stargan/data/BDD_Val_Cropped6/crops/car/list.txt')
        
    #     # parser.add_argument('--CompCars_image_dir', type=str, default='/home/msis/Aroona/Codes/stargan/data/HACKATHON/images')
    #     # parser.add_argument('--attr_path', type=str, default='/home/msis/Aroona/Codes/stargan/data/HACKATHON/list_images.txt')
        
    #     # parser.add_argument('--CompCars_image_dir', type=str, default='/home/msis/Aroona/Codes/stargan/data/CompCars_SvData/images')
    #     # parser.add_argument('--attr_path', type=str, default='/home/msis/Aroona/Codes/stargan/data/CompCars_SvData/list.txt')
        
    #     config = parser.parse_args()
    # else: 
    #     parser.add_argument('--attr_path', type=str, default='/home/msis/Aroona/Codes/stargan/data/celeba/list_attr_celeba.txt')
    #     config = parser.parse_args()
        
    print(config)
    main(config)