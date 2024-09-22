import torch
def build_model_pu(args,ema=False,try_assert=True):
    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        from . import resnet_cifar as models    
    model = models.resnet18_pu(no_class=args.no_class,try_assert=try_assert)

    # use dataparallel if there's multiple gpus
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model
