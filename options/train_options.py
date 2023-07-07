from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # visualization parameters
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--autoresume', default=False, action='store_true', help='continue training: load and keep track of epochs')
        parser.add_argument('--path2epoch', type=str, default='', help='loads in preference networks weights stored at: /path/to/folder/epoch. It loads at the end /path/to/folder/epoch_net_$(name).pth.')

        # training parameters
        parser.add_argument('--niter', type=int, default=70, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.00016, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay as a parameter to the optimizer")
        parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer to use for the optimization", choices=["adam", "adamw", "ranger", "adabelief", "sgd"])
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        # parameters for validation during training
        parser.add_argument('--val_epoch_freq', type=int, default=1, help='frequency (in epochs) for computing the validation')
        parser.add_argument('--val_dataroot', type=str, default='./datasets/validation_dataset', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--val_dataset_mode', type=str, default='infer4rec', help='chooses how datasets are loaded.')
        parser.add_argument('--val_videos',  type=str, default='000,001,002,003,004', help='which videos to use for testing.') #Only used in inferaxel.
        parser.add_argument('--no_val', action='store_true', default=False, help='Don\'t test on val data while training.')

        self.isTrain = True
        return parser
