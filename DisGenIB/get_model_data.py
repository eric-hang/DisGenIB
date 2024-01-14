from models.model import Encoder,Generator,Classifier
def get_model(options):
    if options.dataset == 'miniImageNet' or options.dataset == 'CIFAR-FS':
        n_classes=64

    fea_dim = feat_dim
    network = None

    netA = Encoder(opt = options,encoder_layer_sizes=[fea_dim,4096],latent_size=128).cuda()
    netZ = Encoder(opt = options,encoder_layer_sizes=[fea_dim,4096],latent_size=128).cuda()
    if options.general:
        netG_A = Generator(opt = options,decoder_layer_sizes=[4096,fea_dim],latent_size=128,attsize=128).cuda()
    else:
        netG_A = Generator(opt = options,decoder_layer_sizes=[4096,fea_dim],latent_size=128,attsize=options.attSize).cuda()
    classifier = Classifier(input_size=128,hidden=256,n_class=n_classes).cuda()

    if options.dataset == "miniImageNet":
        netG_Z = Generator(opt = options,decoder_layer_sizes=[4096,fea_dim],latent_size=128,attsize=64).cuda()

    return (network,netA,netZ,netG_A,netG_Z,classifier) #

def get_dataset(options):
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import FewShotDataloader, MiniImageNetPC_feature, MiniImageNet_feature
        if options.phase == 'pretrain_encoder':
            if options.use_feature:
                dataset_train = MiniImageNetPC_feature(phase='train', shot=options.train_shot)

        if options.phase == 'pretrain_encoder':
            if options.use_feature:
                dataset_val = MiniImageNet_feature(phase='val')

        if options.phase == 'generative_test':
            if options.use_feature:
                dataset_test = MiniImageNet_feature(phase='test')
                
        data_loader = FewShotDataloader
    
        
    return (dataset_train, dataset_val, dataset_test, data_loader)