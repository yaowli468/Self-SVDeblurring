from .run_model import Self_Spatial_Variant_Deblurring

def create_model(opt):

	model = Self_Spatial_Variant_Deblurring(opt)
	print("model [%s] was created" % (model.name()))
	return model
