"""
Configuration and hyperparameters for FedProto + SDAR attack.
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="FedProto with SDAR Attack")

    # ── Federated learning arguments ──
    parser.add_argument('--rounds', type=int, default=100,
                        help="number of global communication rounds")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of federated clients")
    parser.add_argument('--train_ep', type=int, default=1,
                        help="number of local training epochs per round")
    parser.add_argument('--local_bs', type=int, default=4,
                        help="local batch size for client training")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="client learning rate")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'], help="optimizer type")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="SGD momentum")

    # ── FedProto-specific arguments ──
    parser.add_argument('--ways', type=int, default=5,
                        help="average number of classes per client")
    parser.add_argument('--shots', type=int, default=100,
                        help="average number of samples per class per client")
    parser.add_argument('--train_shots_max', type=int, default=110,
                        help="max samples per class per client")
    parser.add_argument('--test_shots', type=int, default=15,
                        help="test samples per class per client")
    parser.add_argument('--stdev', type=int, default=2,
                        help="standard deviation for ways/shots sampling")
    parser.add_argument('--ld', type=float, default=0.1,
                        help="weight of prototype loss in FedProto")

    # ── Model arguments ──
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['cnn_mnist', 'cnn_cifar', 'resnet18'],
                        help="client model architecture")
    parser.add_argument('--num_channels', type=int, default=3,
                        help="number of input image channels")
    parser.add_argument('--out_channels', type=int, default=20,
                        help="output channels for CNN models")

    # ── Dataset arguments ──
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100'],
                        help="dataset name")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="number of classes in dataset")
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help="directory for dataset storage")
    parser.add_argument('--server_data_frac', type=float, default=0.3,
                        help="fraction of training data reserved for server auxiliary dataset")

    # ── SDAR attack arguments ──
    parser.add_argument('--attack', action='store_true', default=False,
                        help="enable SDAR attack on the server")
    parser.add_argument('--attack_lr', type=float, default=0.001,
                        help="learning rate for simulator and decoder training")
    parser.add_argument('--attack_epochs', type=int, default=5,
                        help="number of attack training epochs per round")
    parser.add_argument('--lambda1', type=float, default=0.02,
                        help="weight of simulator discriminator GAN loss")
    parser.add_argument('--lambda2', type=float, default=1e-5,
                        help="weight of decoder discriminator GAN loss")
    parser.add_argument('--conditional', action='store_true', default=True,
                        help="use conditional generation (label embedding)")
    parser.add_argument('--no_conditional', dest='conditional', action='store_false')
    parser.add_argument('--use_e_dis', action='store_true', default=True,
                        help="use simulator discriminator")
    parser.add_argument('--no_e_dis', dest='use_e_dis', action='store_false')
    parser.add_argument('--use_d_dis', action='store_true', default=True,
                        help="use decoder discriminator")
    parser.add_argument('--no_d_dis', dest='use_d_dis', action='store_false')
    parser.add_argument('--attack_batch_size', type=int, default=64,
                        help="batch size for SDAR attack training")
    parser.add_argument('--diff_simulator', action='store_true', default=False,
                        help="use a completely different architecture for the simulator (black-box setting)")
    parser.add_argument('--no_proto_avg', action='store_true', default=False,
                        help="ablation: send individual per-image prototypes to the "
                             "attacker instead of class-averaged ones")

    # ── General arguments ──
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID (-1 for CPU)")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed")
    parser.add_argument('--verbose', type=int, default=1,
                        help="verbose output (0=silent, 1=verbose)")
    parser.add_argument('--results_dir', type=str, default='./results/',
                        help="directory to save results")

    args = parser.parse_args()
    return args
