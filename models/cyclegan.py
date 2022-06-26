
import sys
sys.path.append('.')
print(sys.path)
import time
from CycleGAN.options.train_options import TrainOptions
from CycleGAN.data import create_dataset
from CycleGAN.models import create_model
from CycleGAN.util.visualizer import Visualizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from tqdm import tqdm

def transform_mnist():
    transform_list = []
    transform_list.append(transforms.Resize((32,32)))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.RandomVerticalFlip())
    transform_list.append(transforms.RandomRotation(90))
    transform_list.append(transforms.Grayscale(num_output_channels=3))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)

def transform_svhn():
    transform_list = []
    transform_list.append(transforms.Resize((32,32)))
    transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.append(transforms.RandomVerticalFlip())
    transform_list.append(transforms.RandomRotation(90))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)



if __name__ == '__main__':
    os.system('pwd')
    datasetA = datasets.MNIST('./datasets/mnist',train=True, download=False, transform=transform_mnist())
    datasetB  = datasets.SVHN('./datasets/svhn', split='train', download=False, transform=transform_svhn())
    loader_args = dict(batch_size=64,num_workers=8, pin_memory=True)
    iter_A = DataLoader(datasetA, shuffle=True, **loader_args)
    iter_B = DataLoader(datasetB, shuffle=True, **loader_args)

    dataset_size = min(len(datasetA), len(datasetB))

    opt = TrainOptions().parse()   # get training options
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        loss = 0
        for (dA, _), (dB, _) in tqdm(zip(iter_A, iter_B), total=dataset_size/loader_args['batch_size']):  # inner loop within one epoch

            data = {'A':dA, 'B':dB}
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            loss += model.optimize_parameters().item()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                #visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    pass
                    #visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        loss /= len(iter_A)
        print('loss: ', loss)
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
