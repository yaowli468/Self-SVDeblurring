
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    if opt.phase=='pre_train':
        customDataLoader= CustomDatasetDataLoader(opt)
        data_loader, val_data_loader=customDataLoader.load_data()
        return data_loader, val_data_loader
    else:
        data_loader = CustomDatasetDataLoader(opt)
        print(data_loader.name())
        # data_loader.initialize(opt)
        return data_loader

