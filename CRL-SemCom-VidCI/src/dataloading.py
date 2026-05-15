import torch
from torch.utils.data import DataLoader
from src.dataio import NFS_Video

REMOTE_CUSTOM_PATH = '/media/data6/cindy/custom_data'
LOCAL_CUSTOM_PATH = '/home/cindy/PycharmProjects/custom_data'

block_sz = 256


def _dataset_root(args):
    dataset_name = getattr(args, 'dataset_name', f'nfs_block_rgb_{block_sz}_8f')
    return f'{args.data_root}/{dataset_name}'


def _loader_kwargs(num_workers):
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs['prefetch_factor'] = 4
        kwargs['persistent_workers'] = True
    return kwargs


def loadTrainingDataset(args, color=False, test=False):
    # global REMOTE_CUSTOM_PATH
    # if args.ares:
    #     REMOTE_CUSTOM_PATH = '/media/data4b/cindy/custom_data'
    #
    # if not args.local:
    #     args.data_root = f'{REMOTE_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    # else:
    #     args.data_root = f'{LOCAL_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    data_root = _dataset_root(args)
    if args.test:  # use a smaller dataset when you're testing
        split = 'test'
    else:
        split = 'train'
    train_dataset = NFS_Video(log_root=data_root,
                                  block_size=args.block_size,
                                  gt_index=args.gt,
                                  color=color,
                                  split=split,
                                  test=test)
    if test:
        return DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=False)
    loader_kwargs = _loader_kwargs(args.num_workers)
    return DataLoader(train_dataset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      **loader_kwargs)


def loadValDataset(args, color=False):
    # global REMOTE_CUSTOM_PATH
    # if args.ares:
    #     REMOTE_CUSTOM_PATH = '/media/data4b/cindy/custom_data'
    # if not args.local:
    #     args.data_root = f'{REMOTE_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    # else:
    #     args.data_root = f'{LOCAL_CUSTOM_PATH}/nfs_block_rgb_{block_sz}_8f'
    data_root = _dataset_root(args)

    if args.test:
        return None
    else:
        split = 'test'
    val_dataset = NFS_Video(log_root=data_root,
                                block_size=args.block_size,
                                gt_index=args.gt,
                                color=color,
                                split=split)

    loader_kwargs = _loader_kwargs(args.num_workers)
    return DataLoader(val_dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      **loader_kwargs), val_dataset
