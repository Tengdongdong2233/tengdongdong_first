import chainer
from chainer.training import StandardUpdater
import cupy as cp
from loss import l2_loss, gradient_loss, loss_target1, loss_target0, huber_loss, l1_loss
from chainer.functions import resize_images
from chainer import report
from chainer import Variable
from chainer.functions import split_axis
from MultiScaleNetwork import MultiScaleGenerator, MultiScaleDiscriminator
import numpy as np


class Updater(StandardUpdater):
	def __init__(self, iterators, optimizers, GeneratorNetwork, DiscriminatorNetwork,
	             device=0,**kwargs):
		"""

		:param iterators: iterator to be passed to the Updater
		:param optimizers: Dict of optimizers for the Gen and Dis network
						{'GeneratorNetwork':, 'DiscriminatorNetwork':}
		:param GeneratorNetwork: the multiscaleGen network that is to be trained
		:param DiscriminatorNetwork: the multiscale Dis network that is to to be trained
		:param device: whther to use gpu or not
		:param kwargs:  dict params containing the loss factors
					param = {LAM_ADV:, LAM_LP:, LAM_GDL}
		"""

		self.GenNetwork = GeneratorNetwork
		self.DisNetwork = DiscriminatorNetwork

		super(Updater, self).__init__(iterators, optimizers)
		params = kwargs.pop('params')
		self.device = device
		self.LAM_ADV = params['LAM_ADV']
		self.LAM_LP = params['LAM_LP']
		self.LAM_GDL = params['LAM_GDL']
		self.LAM_HB = params['LAM_HB']

		self._optimizers['GeneratorNetwork'].setup(self.GenNetwork)
		self._optimizers['DiscriminatorNetwork'].setup(self.DisNetwork)


	def update_core(self):
		#convert incoming array into variables with either cpu/gpu compatibility
		data = Variable(self.converter(self.get_iterator('main').next(), self.device))
		# data.backward(retain_grad=True)

		# LAMBDA = .1
		n, c, h, w = data.shape
		# Get the ground truth and the sequential inout that is to be fed to the network
		seq, gt = split_axis(data, [c-3], 1)
		# get rid of memory
		del data

		output = None
		total_loss_dis_adv = 0
		total_loss_gen_adv = 0
		for i in range(1, 5):
			# Downscaling of ground truth images for loss calculations
			if i != 4:
				downscaled_gt = resize_images(gt, (int(h / 2 ** (4 - i)), int(w / 2 ** (4 - i))))
				downscaled_seq = resize_images(seq, (int(h / 2 ** (4 - i)), int(w / 2 ** (4 - i))))
			else:
				downscaled_gt = gt
				downscaled_seq = seq

			output = self.GenNetwork.singleforward(i, downscaled_seq, output)
			dis_output_fake = self.DisNetwork.singleforward(i, output)
			dis_output_real = self.DisNetwork.singleforward(i, downscaled_gt)
			# gradient-penalty
			# alpha = cp.random.randn(output.shape[0], output.shape[1] , output.shape[2] , output.shape[3],  dtype='float32')
			# interpolates = alpha * output + ((1 - alpha) * downscaled_gt)
			# interpolates = interpolates.reshape(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
			# disc_interpolates = self.DisNetwork.singleforward(i, interpolates)
			# disc_interpolates = Variable(disc_interpolates)
			# gradients = disc_interpolates.grad
			# gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

			loss_dis = (loss_target1(dis_output_real) + loss_target0(dis_output_fake)) / 2
			# loss_dis += gradient_penalty

			loss_gen = loss_target1(dis_output_fake)

			total_loss_dis_adv += loss_dis
			total_loss_gen_adv += loss_gen

		loss_l2 = l2_loss(output, gt)
		loss_gdl = gradient_loss(output, gt)
		loss_hb = huber_loss(output, gt)
		loss_l1 = l1_loss(output, gt)

		composite_gen_loss = self.LAM_LP*loss_l1 + self.LAM_GDL*loss_gdl + self.LAM_ADV*total_loss_gen_adv + self.LAM_HB*loss_hb

		report({'L1Loss': loss_l1}, self.GenNetwork)
		report({'HuberLoss':loss_hb}, self.GenNetwork)
		report({'L2Loss':loss_l2},self.GenNetwork)
		report({'GDL':loss_gdl},self.GenNetwork)
		report({'AdvLoss':total_loss_gen_adv},self.GenNetwork)
		report({'DisLoss':total_loss_dis_adv},self.DisNetwork)
		report({'CompositeGenLoss':composite_gen_loss},self.GenNetwork)

		# TODO: Come up with a more elegant way
		self.DisNetwork.cleargrads()
		self.GenNetwork.cleargrads()
		composite_gen_loss.backward()
		self._optimizers["GeneratorNetwork"].update()

		self.DisNetwork.cleargrads()
		self.GenNetwork.cleargrads()
		total_loss_dis_adv.backward()
		self._optimizers["DiscriminatorNetwork"].update()







