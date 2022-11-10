from __future__ import print_function

from utils.utils import seed_torch, get_time, adjust_learning_rate_D

# fix random seeds
seed_torch(2019)

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.res_unet_cbam import Shared_Res_unet_cbam_insn
from models.gan_model import NLayerDiscriminator, GANLoss
from models.gan_model import NetC, criterion_gan, FCDiscriminator
from utils.loss import DICELoss_LV
from config.config import random_list_all
from data.data import PancreasCancerDatasetAugmentationIterFold

def get_args():
    parser = argparse.ArgumentParser(description="Pancreatic cancer Dataset")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for testing (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        metavar="N",
        help="number of epochs to train (default: 15)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=0.0002,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="enables CUDA training (default: true)",
    )
    parser.add_argument("--size", type=int, default=256, metavar="N", help="imsize")
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        metavar="str",
        help="weight file to load (default: None)",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="3",
        metavar="N",
        help="input visible devices for training (default: 3)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        metavar="str",
        help="Optimizer (default: SGD)",
    )
    parser.add_argument(
        "--dis",
        type=str,
        default="NL",
        metavar="str",
        help="Discriminator (default: NL)",
    )
    parser.add_argument("--fold", type=int, default=0, metavar="N", help="fold(0-3)")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_discriminators = 5
    unsup_weights = [0.1, 0.1, 0.1, 0.1, 0.6]
    lambda_adv_tgts = [0.1, 0.1, 0.1, 0.1, 0.6]

    test_list = []
    random_list = random_list_all
    if args.fold == 0:
        test_list = random_list[:16]
    else:
        test_list = random_list[args.fold * 17 - 1 : (args.fold + 1) * 17 - 1]

    NetS = Shared_Res_unet_cbam_insn(mode=1)

    # create a list of discriminators
    discriminators = []
    if args.dis == "FCD":
        for dis_idx in range(n_discriminators):
            discriminators.append(FCDiscriminator())
            discriminators[dis_idx].train()
            discriminators[dis_idx].cuda()
    elif args.dis == "NL":
        criterion_gan = GANLoss(use_lsgan=True)
        for dis_idx in range(n_discriminators):
            discriminators.append(NLayerDiscriminator())
            discriminators[dis_idx].train()
            discriminators[dis_idx].cuda()

    e_discriminators = []
    for dis_idx in range(n_discriminators):
        e_discriminators.append(NetC())
        e_discriminators[dis_idx].train()
        e_discriminators[dis_idx].cuda()

    args.cuda = args.cuda and torch.cuda.is_available()

    if args.cuda:
        NetS.cuda()

    print("##########Building model done!##########")
    criterion = DICELoss_LV()
    if args.optimizer == "SGD":
        optimizerG = optim.SGD(NetS.parameters(), lr=args.lr, momentum=0.99)
    else:
        optimizerG = optim.Adam(NetS.parameters(), lr=args.adam_lr, betas=(0.5, 0.999))

    d_optimizers = []
    for idx in range(n_discriminators):
        optimizer = optim.Adam(
            discriminators[idx].parameters(), lr=args.adam_lr, betas=(0.9, 0.99)
        )
        d_optimizers.append(optimizer)

    e_optimizers = []
    for idx in range(n_discriminators):
        optimizer = optim.Adam(
            e_discriminators[idx].parameters(), lr=args.adam_lr, betas=(0.9, 0.99)
        )
        e_optimizers.append(optimizer)

    dset_train = PancreasCancerDatasetAugmentationIterFold(
        fold=args.fold
    )
    # change num_workers to >0 according your CPU to speed up training
    train_loader = DataLoader(
        dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    cur_time = get_time()
    print("Training Data : ", len(train_loader.dataset))
    print("#################################")
    print("epoch:", args.epochs)
    print("lr:", args.lr)
    print("batch_size:", args.batch_size)
    print("optimizer:", args.optimizer)
    print("#################################")

    for epoch in range(args.epochs):

        loss_list = []
        lossD_list = []

        with tqdm(train_loader) as t:
            for batch_idx, (image_t1, mask_t1, image_t2, mask_t2) in enumerate(t):
                t.set_description("epoch %s" % epoch)

                for d_optimizer in d_optimizers:
                    d_optimizer.zero_grad()
                for e_optimizer in e_optimizers:
                    e_optimizer.zero_grad()
                optimizerG.zero_grad()

                if args.cuda:
                    image_t1, mask_t1, image_t2, mask_t2 = (
                        image_t1.cuda(),
                        mask_t1.cuda(),
                        image_t2.cuda(),
                        mask_t2.cuda(),
                    )
                image_t1, mask_t1, image_t2, mask_t2 = (
                    image_t1.float(),
                    mask_t1.float(),
                    image_t2.float(),
                    mask_t2.float(),
                )
                image_t1, mask_t1, image_t2, mask_t2 = (
                    Variable(image_t1),
                    Variable(mask_t1),
                    Variable(image_t2),
                    Variable(mask_t2),
                )

                NetS.mode = 1
                NetS.train()
                sup_pred_t1_all = list(NetS(image_t1))
                sup_pred_t1 = sup_pred_t1_all[:5]
                sup_encoder_t1 = sup_pred_t1_all[5:]

                NetS.mode = 2
                NetS.train()
                sup_pred_t2_all = list(NetS(image_t2))
                sup_pred_t2 = sup_pred_t2_all[:5]
                sup_encoder_t2 = sup_pred_t2_all[5:]

                input_t1 = image_t1.clone()
                target_masked_t1 = input_t1 * mask_t1
                input_t2 = image_t2.clone()
                target_masked_t2 = input_t2 * mask_t2

                if epoch % 5 == 0:
                    d_losses = []
                    for idx in range(n_discriminators):
                        discriminator = discriminators[idx]
                        for param in discriminator.parameters():
                            param.requires_grad = True

                        temp = sup_encoder_t1[idx].detach()
                        d_outs_t1 = discriminators[idx](temp)
                        d_losses.append(criterion_gan(d_outs_t1, True))
                        d_losses[idx] = d_losses[idx] / 2
                        d_losses[idx].backward()

                    for idx in range(n_discriminators):
                        temp = sup_encoder_t2[idx].detach()
                        d_outs_t2 = discriminators[idx](temp)
                        d_losses[idx] = criterion_gan(d_outs_t2, False)
                        d_losses[idx] = d_losses[idx] / 2
                        d_losses[idx].backward()

                    for d_optimizer in d_optimizers:
                        d_optimizer.step()

                    # clip parameters in D
                    for discriminator in discriminators:
                        for p in discriminator.parameters():
                            p.data.clamp_(-0.05, 0.05)

                    for idx in range(n_discriminators):
                        discriminator = e_discriminators[idx]
                        for param in discriminator.parameters():
                            param.requires_grad = True

                        temp = sup_pred_t1[idx].detach()
                        d_outs_t1 = e_discriminators[idx](input_t1 * temp)
                        d_target_t1 = e_discriminators[idx](target_masked_t1)
                        d_losses = -torch.mean(torch.abs(d_outs_t1 - d_target_t1))
                        d_losses = d_losses / 2
                        d_losses.backward()

                    for idx in range(n_discriminators):
                        temp = sup_pred_t2[idx].detach()
                        d_outs_t2 = e_discriminators[idx](input_t2 * temp)
                        d_target_t2 = e_discriminators[idx](target_masked_t2)
                        d_losses = -torch.mean(torch.abs(d_outs_t2 - d_target_t2))
                        d_losses = d_losses / 2
                        d_losses.backward()

                    for e_optimizer in e_optimizers:
                        e_optimizer.step()

                    # clip parameters in D
                    for discriminator in e_discriminators:
                        for p in discriminator.parameters():
                            p.data.clamp_(-0.05, 0.05)

                for discriminator in discriminators:
                    for param in discriminator.parameters():
                        param.requires_grad = False

                for discriminator in e_discriminators:
                    for param in discriminator.parameters():
                        param.requires_grad = False

                seg_losses, total_seg_loss = [], 0
                for idx in range(len(sup_pred_t1)):
                    seg_loss = 0.5 * (
                        criterion(sup_pred_t1[idx], mask_t1)
                        + criterion(sup_pred_t2[idx], mask_t2)
                    )
                    seg_losses.append(seg_loss)
                    total_seg_loss += seg_loss * unsup_weights[idx]

                total_adv_loss, total_adv_loss_e = (0, 0)
                for idx in range(n_discriminators):
                    d_outs_t1 = discriminators[idx](sup_encoder_t1[idx])
                    d_outs_t2 = discriminators[idx](sup_encoder_t2[idx])
                    adv_tgt_loss = 0.5 * (
                        criterion_gan(d_outs_t1, False) + criterion_gan(d_outs_t2, True)
                    )
                    total_adv_loss += lambda_adv_tgts[idx] * adv_tgt_loss

                for idx in range(n_discriminators):
                    sup_t1 = sup_pred_t1[idx]
                    sup_t2 = sup_pred_t2[idx]
                    d_pred_t1 = e_discriminators[idx](input_t1 * sup_t1)
                    d_pred_t2 = e_discriminators[idx](input_t2 * sup_t2)
                    d_target_t1 = e_discriminators[idx](target_masked_t1)
                    d_target_t2 = e_discriminators[idx](target_masked_t2)
                    adv_tgt_loss = 0.5 * (
                        torch.mean(torch.abs(d_pred_t1 - d_target_t1))
                        + torch.mean(torch.abs(d_pred_t2 - d_target_t2))
                    )
                    total_adv_loss_e += lambda_adv_tgts[idx] * adv_tgt_loss

                loss_list.append(total_seg_loss.item())
                lossD_list.append(total_adv_loss.item())

                total_loss = total_seg_loss + total_adv_loss + total_adv_loss_e
                total_loss.backward()
                optimizerG.step()

                t.set_postfix(
                    ave_Gloss=np.mean(loss_list),
                    ave_Dloss=np.mean(lossD_list),
                    cur_loss=loss_list[-1],
                )
        for d_optimizer in d_optimizers:
            adjust_learning_rate_D(d_optimizer, args.adam_lr, epoch, args.epochs)
        for e_optimizer in e_optimizers:
            adjust_learning_rate_D(e_optimizer, args.adam_lr, epoch, args.epochs)
        adjust_learning_rate_D(optimizerG, args.lr, epoch, args.epochs, power=2)

        with open(
            "logs/MMSA_fold%s_%s.txt" % (args.fold, cur_time), "a"
        ) as f:
            f.write(
                "ave_loss=%s, ave_Dloss=%s \n"
                % (np.mean(loss_list), np.mean(lossD_list))
            )

    torch.save(
        NetS.state_dict(),
        "checkpoints/MMSA_fold%s_%s.pth" % (args.fold, cur_time),
    )

    print("#################################")
    print("epoch:", args.epochs)
    print("lr:", args.lr)
    print("batch_size:", args.batch_size)
    print("optimizer:", args.optimizer)
    print("cur_time:", cur_time)
    print("#################################")
