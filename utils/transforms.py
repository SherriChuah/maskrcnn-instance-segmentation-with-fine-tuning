import torchvision.transforms as T


def get_transform(train=True):
    transforms = []
    # convert image to pytorch tensor
    transforms.append(T.ToTensor())

    if train:
        # apply horizontal flip randomly
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    # combined all transforms into a pipeline
    return T.Compose(transforms)
