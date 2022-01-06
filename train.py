# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from dataset.dataset import AVADataset

from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()


def main(config):

    writer = SummaryWriter()

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    model = VGG16BinaryNet()

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pth' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pth' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=[0, 1])
        model = model.to(device)
    else:
        model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.img_path, transform=train_transform)
        valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.img_path, transform=val_transform)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
            shuffle=False, num_workers=config.num_workers)
        # for early stopping
        count = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        for epoch in range(config.warm_start_epoch, config.epochs):
            batch_losses = []
            batch_acc = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['style_label'].to(device).float()
                outputs = model(images)
                optimizer.zero_grad()

                # loss, acc = emd_loss(labels, outputs)
                loss = criterion(outputs, labels)
                batch_losses.append(loss.item())
                # batch_acc.append(acc)

                loss.backward()
                optimizer.step()

                print('Epoch: %d/%d | Step: %d/%d | Training CrossEntropyLoss loss: %.4f' % (epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1, loss.item()))
                writer.add_scalar('batch train loss', loss.item(), i + epoch * (len(trainset) // config.train_batch_size + 1))

                if (i + 1) % 4 == 0:
                    batch_val_losses = []
                    for j, val_data in enumerate(val_loader):
                        images = val_data['image'].to(device)
                        labels = val_data['style_label'].to(device).float()
                        with torch.no_grad():
                            outputs = model(images)
                        val_loss = criterion(outputs, labels)
                        batch_val_losses.append(val_loss.item())
                    avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
                    val_losses.append(avg_val_loss)
                    print('Epoch %d completed. Mean CrossEntropyLoss loss on val set: %.4f.' % ((i + 1) // 6, avg_val_loss))
                    if avg_val_loss < init_val_loss:
                        init_val_loss = avg_val_loss
                        # save model weights if val loss decreases
                        print('Saving model...')
                        if not os.path.exists(config.ckpt_path):
                            os.makedirs(config.ckpt_path)
                        torch.save(model.state_dict(),
                                   os.path.join(config.ckpt_path, 'epoch-%d-%f.pth' % ((i + 1) // 6, avg_val_loss)))
                        print('Done.\n')


            avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size + 1)
            # avg_acc = sum(batch_acc) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            print('Epoch %d mean training CrossEntropyLoss loss: %.4f' % (epoch + 1, avg_loss))


            # do validation after each epoch
            batch_val_losses = []
            for i, data in enumerate(val_loader):
                images = data['image'].to(device)
                labels = data['style_label'].to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                val_loss = criterion(outputs, labels)
                batch_val_losses.append(val_loss.item())
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)
            print('Epoch %d completed. Mean CrossEntropyLoss loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
            writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

            # Use early stopping to monitor training
            torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d-%f.pth' % (epoch + 1, avg_val_loss)))

            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                if not os.path.exists(config.ckpt_path):
                    os.makedirs(config.ckpt_path)
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print('Val CrossEntropyLoss loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')

        '''
        # use tensorboard to log statistics instead
        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            plt.plot(epochs, train_losses, 'b-', label='train loss')
            plt.plot(epochs, val_losses, 'g-', label='val loss')
            plt.title('CrossEntropyLoss loss')
            plt.legend()
            plt.savefig('./loss.png')
        '''

    if config.test:
        model.eval()
        # compute mean score
        test_transform = val_transform
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.img_path, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        for data in test_loader:
            image = data['image'].to(device)
            output = model(image)
            output = output.view(10, 1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            predicted_std = predicted_std ** 0.5
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--img_path', type=str, default='/home/lab325/gry/Innovative/AVA_dataset/all_images/')
    parser.add_argument('--train_csv_file', type=str, default='./data/2_style_binary_train.csv')
    parser.add_argument('--val_csv_file', type=str, default='./data/2_style_binary_val.csv')
    parser.add_argument('--test_csv_file', type=str, default='./data/2_style_binary_val.csv')

    # training parameters
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_batch_size', type=int, default=156)
    parser.add_argument('--val_batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/2')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true')

    config = parser.parse_args()

    main(config)

