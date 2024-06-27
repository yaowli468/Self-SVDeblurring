import pdb
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from multiprocessing import freeze_support

def train(opt, data_loader, visualizer):
	dataset = data_loader.load_data()
	dataset_size = len(data_loader) * opt.batchSize
	print('#training images = %d' % dataset_size)

	for i, data in enumerate(dataset):
		total_steps = 0
		model = create_model(opt)

		for step in range(1, 4000+1):
			total_steps =step
			model.set_input(data)

			model.optimize_parameters(step)

			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				visualizer.print_current_errors_all_stage(total_steps, errors)

				#model.save_kernel(step)

			if total_steps % opt.display_freq == 0:
				model.get_current_visuals()
				model.save_image(total_steps)

			model.update_learning_rate(step)

		print('saving the model at the end of epoch %d' % (step))
		model.save('latest')
		model.save(step)


if __name__ == '__main__':
	freeze_support()

	# python run.py --pretrainStyleGanModel './pretrainModel/stylegan2-ffhq-config-f.pt'

	opt = TrainOptions().parse()
	opt.dataroot = './images/NonUniform'
	opt.saveroot='./checkpoints/finetune'
	#opt.pretrainModel='./pretrainModel/pggan_celebahq1024_generator.pth'
	#opt.pretrainStyleGanModel='./pretrainModel/stylegan2-ffhq-config-f.pt'
	#opt.pretrainStyleGanModel='./pretrainModel/afhqwild.pt'
	#opt.pretrainStyleGanModel='./pretrainModel/stylegan2-car-config-f.pt'
	#opt.pretrainStyleGanModel='./pretrainModel/stylegan2-church-config-f.pt'
	#opt.pretrainedEncoder='./pretrainModel/best_encoder.pt'
	#pdb.set_trace()
	opt.learn_residual = True
	opt.resize_or_crop = "crop"
	opt.fineSize = 256
	opt.gan_type = "gan"
	opt.dataset_mode = 'single'
	opt.stage='train'
	opt.kernel_size=21

	opt.save_latest_freq = 50

	opt.print_freq = 20
	opt.display_freq=20
	data_loader= CreateDataLoader(opt)

	visualizer = Visualizer(opt)
	train(opt, data_loader, visualizer)
