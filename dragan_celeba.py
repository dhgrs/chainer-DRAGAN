# coding: UTF-8
import argparse
import os
import glob
import random

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert


class Generator(chainer.Chain):
    def __init__(self, noise=128, dim=512):
        super(Generator, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, dim * 8 * 8)
            self.dc1 = L.Deconvolution2D(None, dim // 2, 4, 2, 1)
            self.dc2 = L.Deconvolution2D(None, dim // 4, 4, 2, 1)
            self.dc3 = L.Deconvolution2D(None, dim // 8, 4, 2, 1)
            self.dc4 = L.Deconvolution2D(None, 3, 3, 1, 1)
        self.dim = dim
        self.noise = noise

    def __call__(self, z):
        h = F.relu(self.fc(z)).reshape((-1, self.dim, 8, 8))
        h = F.relu(self.dc1(h))
        h = F.relu(self.dc2(h))
        h = F.relu(self.dc3(h))
        x = F.tanh(self.dc4(h))
        return x

    def make_z(self, n):
        z = self.xp.random.normal(size=(n, self.noise)).astype(self.xp.float32)
        return z


class Discriminator(chainer.Chain):
    def __init__(self, dim=512):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(None, dim // 8, 4, 2, 1)
            self.c2 = L.Convolution2D(None, dim // 4, 4, 2, 1)
            self.c3 = L.Convolution2D(None, dim // 2, 4, 2, 1)
            self.c4 = L.Convolution2D(None, dim, 3, 1, 1)
            self.fc = L.Linear(None, 1)
        self.dim = dim

    def __call__(self, x):
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c3(h))
        h = F.leaky_relu(self.c4(h))
        y = self.fc(h)
        return y


class DRAGANUpdater(training.StandardUpdater):
    def __init__(self, iterator, l, opt_g, opt_d, device):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.generator = opt_g.target
        self.discriminator = opt_d.target
        self.l = l
        self._optimizers = {'generator': opt_g, 'discriminator': opt_d}
        self.device = device
        self.converter = convert.concat_examples
        self.iteration = 0

    def update_core(self):
        # read data
        batch = self._iterators['main'].next()
        x_real = self.converter(batch, self.device)
        x_real = F.resize_images(x_real, (64, 64))
        m = x_real.shape[0]
        xp = chainer.cuda.get_array_module(x_real)
        x_real_std = xp.std(x_real.data, axis=0, keepdims=True)
        x_random = xp.random.uniform(0, 1, x_real.shape).astype(xp.float32)
        x_perturb = chainer.Variable(x_real.data + 0.5 * x_random * x_real_std)

        # generate
        z = self.generator.make_z(m)
        x_fake = self.generator(z)

        # discriminate
        y_real = self.discriminator(x_real)
        y_fake = self.discriminator(x_fake)

        # compute loss
        loss_gan = (F.average(F.softplus(-y_real)) +
                    F.average(F.softplus(y_fake)))
        grad, = chainer.grad([self.discriminator(x_perturb)], [x_perturb],
                             enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = self.l * F.mean_squared_error(grad,
                                                  xp.ones_like(grad.data))
        loss_discriminator = loss_gan + loss_grad
        loss_generator = F.average(-y_fake)

        # update discriminator
        self.discriminator.cleargrads()
        loss_discriminator.backward()
        self._optimizers['discriminator'].update()

        # update generator
        self.generator.cleargrads()
        loss_generator.backward()
        self._optimizers['generator'].update()

        # report
        chainer.reporter.report({
            'loss/gan/dis': loss_gan, 'loss/grad': loss_grad,
            'loss/gan/gen': loss_generator})


def main():
    parser = argparse.ArgumentParser(description='DRAGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', '-i', type=int, default=200000,
                        help='Number of iteration')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--directory', '-d', default='.',
                        help='root directory of CelebA Dataset')
    args = parser.parse_args()

    generator = Generator()
    discriminator = Discriminator()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        discriminator.to_gpu()

    opt_g = chainer.optimizers.Adam(1e-4, beta1=0.5, beta2=0.9)
    opt_g.setup(generator)

    opt_c = chainer.optimizers.Adam(1e-4, beta1=0.5, beta2=0.9)
    opt_c.setup(discriminator)

    def preprocess(x):
        # crop 128x128 and flip
        top = random.randint(0, 218 - 128)
        left = random.randint(0, 178 - 128)
        bottom = top + 128
        right = left + 128
        # flip
        x = x[:, top:bottom, left:right]
        if random.randint(0, 1):
            x = x[:, :, ::-1]
        # to [-1, 1]
        x = x * (2 / 255) - 1
        return x

    train = chainer.datasets.TransformDataset(
        chainer.datasets.ImageDataset(glob.glob(
            os.path.join(args.directory, 'Img/img_align_celeba_png/*.png'))),
        preprocess)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = DRAGANUpdater(train_iter, 10,
                            opt_g, opt_c, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'),
                               out=args.out)

    def out_generated_image(generator, H, W, rows, cols, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            # generate
            z = generator.make_z(rows * cols)
            with chainer.using_config('enable_backprop', False):
                # with chainer.using_config('train', False):
                x = generator(z)
            x = chainer.cuda.to_cpu(x.data)
            x = (x + 1) * (255 / 2)

            # convert to image
            x = np.asarray(np.clip(x, 0.0, 255.0), dtype=np.uint8)
            channels = x.shape[1]
            x = x.reshape((rows, cols, channels, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, channels))
            x = np.squeeze(x)

            # save
            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>6}.png'.format(trainer.updater.iteration)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x).save(preview_path)
        return make_image

    trainer.extend(extensions.dump_graph('loss/gan/dis'))
    trainer.extend(extensions.snapshot(filename='snapshot'),
                   trigger=(1000, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    trainer.extend(
        extensions.PlotReport(['loss/gan/gen', 'loss/gan/dis'],
                              'iteration', file_name='gan.png',
                              trigger=(100, 'iteration')))
    trainer.extend(
        extensions.PlotReport(
            ['loss/grad'], 'iteration', file_name='grad.png',
            trigger=(100, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'loss/gan/gen', 'loss/gan/dis', 'loss/grad',
         'elapsed_time']), trigger=(100, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=100))
    trainer.extend(out_generated_image(generator, 64, 64, 4, 4, args.out),
                   trigger=(1000, 'iteration'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        trainer.stop_trigger = chainer.training.trigger.get_trigger(
            (args.iteration, 'iteration'))
    trainer.run()


if __name__ == '__main__':
    main()
