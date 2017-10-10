# coding: UTF-8
import argparse
import os

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
    def __init__(self, ):
        super(Generator, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 28 * 28)
            )

    def __call__(self, z):
        h = F.relu(self.fc1(z))
        y = F.reshape(F.sigmoid(self.fc2(h)), (-1, 1, 28, 28))
        return y

    def make_z(self, n):
        z = self.xp.random.normal(size=(n, 10)).astype(self.xp.float32)
        return z


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            fc1=L.Linear(None, 800),
            fc2=L.Linear(None, 1)
            )

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        y = self.fc2(h)
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
        m = x_real.shape[0]
        xp = chainer.cuda.get_array_module(x_real)
        x_real_std = xp.std(x_real, axis=0, keepdims=True)
        x_random = xp.random.uniform(0, 1, x_real.shape).astype(xp.float32)
        x_perturb = chainer.Variable(x_real + 0.5 * x_random * x_real_std)

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
            'loss/discriminator': loss_discriminator, 'loss/grad': loss_grad,
            'loss/generator': loss_generator})


def main():
    parser = argparse.ArgumentParser(description='DRAGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    generator = Generator()
    discriminator = Discriminator()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        discriminator.to_gpu()

    opt_g = chainer.optimizers.Adam()
    opt_g.setup(generator)

    opt_d = chainer.optimizers.Adam()
    opt_d.setup(discriminator)

    train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = DRAGANUpdater(train_iter, 10,
                            opt_g, opt_d, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    def out_generated_image(generator, H, W, rows, cols, dst):
        @chainer.training.make_extension()
        def make_image(trainer):
            n_images = rows * cols
            xp = generator.xp
            z = generator.make_z(rows * cols)
            with chainer.using_config('enable_backprop', False):
                with chainer.using_config('train', False):
                    x = generator(z)
            x = chainer.cuda.to_cpu(x.data)

            x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
            channels = x.shape[1]
            x = x.reshape((rows, cols, channels, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, channels))
            x = np.squeeze(x)

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image{:0>5}.png'.format(trainer.updater.epoch)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x).save(preview_path)
        return make_image

    trainer.extend(extensions.dump_graph('loss/discriminator'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport(['loss/discriminator', 'loss/grad'],
                              'epoch', file_name='discriminator.png'))
    trainer.extend(
        extensions.PlotReport(
            ['loss/generator'], 'epoch', file_name='generator.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'loss/discriminator', 'loss/grad',
         'loss/generator', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(out_generated_image(generator, 28, 28, 5, 5, args.out),
                   trigger=(1, 'epoch'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()

if __name__ == '__main__':
    main()
