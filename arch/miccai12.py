# pylint: disable=C,R,E1101
'''
Architecture for MRI image segmentation.

'''

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import time
from functools import partial

from se3_cnn.blocks import GatedBlock
from se3_cnn.datasets import MRISegmentation
from se3_cnn import basis_kernels

tensorflow_available = True
try:
    import tensorflow as tf

    class Logger(object):
        '''From https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard'''

        def __init__(self, log_dir):
            """Create a summary writer logging to log_dir."""
            self.writer = tf.summary.FileWriter(log_dir)

        def scalar_summary(self, tag, value, step):
            """Log a scalar variable."""
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.writer.add_summary(summary, step)

        def histo_summary(self, tag, values, step, bins=1000):
            """Log a histogram of the tensor of values."""

            # Create a histogram using numpy
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill the fields of the histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values ** 2))

            # Drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            # Create and write Summary
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            self.writer.add_summary(summary, step)
            self.writer.flush()

except:
    tensorflow_available = False


class FlattenSpacial(nn.Module):
    def __init__(self):
        super(FlattenSpacial, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1, x.size(-3)*x.size(-2)*x.size(-1))


class BasicModel(nn.Module):
    def __init__(self, output_size, filter_size=5):
        super(BasicModel, self).__init__()
        size = filter_size
        bias = True
        self.layers = nn.Sequential(
                        nn.Conv3d(1, 8, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(8),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(8, 16, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(16),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(16, 32, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(32, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(64, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
                        nn.BatchNorm3d(128),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(128, output_size, kernel_size=1, padding=0, stride=1, bias=bias))

    def forward(self, x):
        out = self.layers(x)
        return out


class SE3BasicModel(nn.Module):

    def __init__(self, output_size, filter_size=5):
        super(SE3BasicModel, self).__init__()

        features = [(1,),
                    (4, 4, 4, 4),
                    (4, 4, 4, 4),
                    # (4, 4, 4, 4),
                    (output_size,)]

        common_block_params = {
            'size': filter_size,
            'padding': filter_size//2,
            'stride': 1,
            'normalization': 'instance',
            'radial_window': partial(
                basis_kernels.gaussian_window_fct_convenience_wrapper,
                mode='compromise', border_dist=0, sigma=0.6),
        }

        block_params = [
            {'activation': (F.relu, F.sigmoid)},
            {'activation': (F.relu, F.sigmoid)},
            # {'activation': (F.relu, F.sigmoid)},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1],
                             **common_block_params, **block_params[i])
                  for i in range(len(block_params))]

        self.layers = torch.nn.Sequential(
            *blocks,
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class Merge(nn.Module):
    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class UnetModel(nn.Module):
    def __init__(self, output_size, filter_size=5):
        super(UnetModel, self).__init__()
        size = filter_size
        bias = True

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64))

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=size, padding=size//2, stride=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128))

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=size, padding=size//2, stride=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(256, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True))

        self.merge1 = Merge()

        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(128))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(128, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True))

        self.merge2 = Merge()

        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=size, padding=size // 2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=size, padding=size//2, stride=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64))

        self.conv_final = nn.Conv3d(64, output_size, kernel_size=1, padding=0, stride=1, bias=bias)


    def forward(self, x):

        conv1_out  = self.conv1(x)
        conv2_out  = self.conv2(conv1_out)
        conv3_out  = self.conv3(conv2_out)
        up1_out    = self.up1(conv3_out)
        merge1_out = self.merge1(conv2_out, up1_out)
        conv4_out  = self.conv4(merge1_out)
        up2_out    = self.up2(conv4_out)
        merge2_out = self.merge2(conv1_out, up2_out)
        conv5_out  = self.conv5(merge2_out)
        out        = self.conv_final(conv5_out)

        return out


class SE3UnetModel(nn.Module):
    def __init__(self, output_size, filter_size=5):
        super(SE3UnetModel, self).__init__()
        size = filter_size

        common_params = {
            'radial_window': partial(basis_kernels.gaussian_window_fct_convenience_wrapper,
                                     mode='compromise', border_dist=0, sigma=0.6),
            'batch_norm_momentum': 0.01,
        }

        features = [(1,),
                    (12, 12, 12),
                    (24, 24, 24),
                    (48, 48, 48),
                    (24, 24, 24),
                    (12, 12, 12),
                    (output_size,)]

        # TODO: do padding using ReplicationPad3d?
        # TODO: on validation - use overlapping patches and only use center of patch

        self.conv1 = nn.Sequential(
            GatedBlock(features[0], features[1], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[1], features[1], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.conv2 = nn.Sequential(
            GatedBlock(features[1], features[2], size=size, padding=size//2, stride=2, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[2], features[2], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.conv3 = nn.Sequential(
            GatedBlock(features[2], features[3], size=size, padding=size//2, stride=2, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[3], features[3], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedBlock(features[3], features[4], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.merge1 = Merge()

        self.conv4 = nn.Sequential(
            GatedBlock(features[3], features[4], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[4], features[4], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            GatedBlock(features[4], features[5], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.merge2 = Merge()

        self.conv5 = nn.Sequential(
            GatedBlock(features[4], features[5], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params),
            GatedBlock(features[5], features[5], size=size, padding=size//2, stride=1, activation=(F.relu, F.sigmoid), normalization="instance", **common_params))

        self.conv_final = GatedBlock(features[5], features[6], size=1, padding=0, stride=1, activation=None, normalization=None, **common_params)

    def forward(self, x):

        conv1_out  = self.conv1(x)
        conv2_out  = self.conv2(conv1_out)
        conv3_out  = self.conv3(conv2_out)
        up1_out    = self.up1(conv3_out)
        merge1_out = self.merge1(conv2_out, up1_out)
        conv4_out  = self.conv4(merge1_out)
        up2_out    = self.up2(conv4_out)
        merge2_out = self.merge2(conv1_out, up2_out)
        conv5_out  = self.conv5(merge2_out)
        out        = self.conv_final(conv5_out)

        return out


def dice_coefficient_orig_binary(y_pred, y_true, y_pred_is_dist=False,
                                 classes=None, epsilon=1e-5, reduce=True):
    """As originally specified: using binary vectors and an explicit average
       over classes. If y_pred_is_dist is false, the classes variable must specify the
       number of classes"""

    if y_pred_is_dist:
        y_pred = torch.max(nn.Softmax(dim=1)(y_pred), dim=1)[1]

    if classes is None:
        classes = y_pred.size(1)

    dice_coeff = 0
    if torch.cuda.is_available():
        intersection = torch.cuda.FloatTensor(y_pred.size(0), classes).fill_(0)
        union = torch.cuda.FloatTensor(y_pred.size(0), classes).fill_(0)
    else:
        intersection = torch.zeros(y_pred.size(0), classes)
        union = torch.zeros(y_pred.size(0), classes)
    if isinstance(y_pred, torch.autograd.Variable):
        intersection = torch.autograd.Variable(intersection)
        union = torch.autograd.Variable(union)
    for i in range(classes):

        # Convert to binary values
        y_pred_b = (y_pred == i)
        y_true_b = (y_true == i)

        y_pred_f = y_pred_b.contiguous().view(y_pred.size(0), -1).float()
        y_true_f = y_true_b.contiguous().view(y_true.size(0), -1).float()

        s1 = y_true_f
        s2 = y_pred_f

        # Calculate dice score
        intersection[:,i] = 2. * torch.sum(s1 * s2, dim=1)
        union[:,i] = torch.sum(s1, dim=1) + torch.sum(s2, dim=1)
        dice_coeff += (2. * torch.sum(s1 * s2, dim=1)) / \
                      (epsilon + torch.sum(s1, dim=1) + torch.sum(s2, dim=1))

    if reduce:
        return torch.mean(torch.sum(intersection, dim=0) /
                          (epsilon+torch.sum(union, dim=0)))
    else:
        return intersection, union


# def dice_coefficient_orig(y_pred, y_true, epsilon=1e-5):
#     """Original version but multiplying probs instead of 0-1 variables"""
#
#     y_pred = nn.Softmax(dim=1)(y_pred)
#
#     dice_coeff = 0
#     for i in range(y_pred.size(1)):
#
#         y_pred_b = y_pred[:,i,:,:,:]
#         y_true_b = y_true == i
#
#         y_pred_f = y_pred_b.contiguous().view(y_pred.size(0), -1)
#         y_true_f = y_true_b.contiguous().view(y_true.size(0), -1).float()
#
#         s1 = y_true_f
#         s2 = y_pred_f
#
#         dice_coeff += (2. * torch.sum(s1 * s2, dim=1)) / \
#                       (epsilon + torch.sum(s1, dim=1) + torch.sum(s2, dim=1))
#
#     dice_coeff /= float(y_pred.size(1))
#
#     return dice_coeff.mean()


# def dice_coefficient_onehot(y_pred, y_true, epsilon=1e-5, reduce=True):
#     """Reimplementation with matrix operations - with onehot encoding
#        of y_true"""
#
#     y_pred = nn.Softmax(dim=1)(y_pred)
#
#     y_pred_f = y_pred.view(y_pred.size(0), y_pred.size(1), -1)
#     y_true_f = y_true.view(y_true.size(0), y_true.size(1), -1)
#
#     intersection = torch.sum(y_true_f * y_pred_f, dim=2)
#     coeff = 2./y_pred.shape[1] * torch.sum(
#         intersection / (epsilon +
#                         torch.sum(y_true_f, dim=2) +
#                         torch.sum(y_pred_f, dim=2)),
#         dim=1)
#     return coeff.mean()


def dice_coefficient(y_pred, y_true, valid=None, reduce=True, epsilon=1e-5):
    """Reimplementation with matrix operations - directly on y_true class
       labels"""

    y_pred = nn.Softmax(dim=1)(y_pred)

    mask = None
    if valid is not None:
        mask = get_mask((y_true.size(0), y_true.size(-3), y_true.size(-2), y_true.size(-1)), valid)

    all_classes = torch.autograd.Variable(
        torch.LongTensor(np.arange(y_pred.size(1))))
    if torch.cuda.is_available():
        all_classes = all_classes.cuda()
    intersections = []
    unions = []
    for i in range(y_pred.shape[0]):

        if mask is not None:
            y_pred_f = y_pred[i][mask[i]].view(y_pred.size(1), -1)
            y_true_f = y_true[i][mask[i]].view(-1)
        else:
            y_pred_f = y_pred[i].view(y_pred.size(1), -1)
            y_true_f = y_true[i].view(-1)

        if len(y_true_f.shape) > 0:
            # Dynamically create one-hot encoding
            class_at_voxel = (all_classes.view(-1, 1) == y_true_f).float()
            intersection = torch.sum(class_at_voxel * y_pred_f,
                                     dim=1)
            intersections.append(2*intersection)
            unions.append(torch.sum(class_at_voxel, dim=1) +
                          torch.sum(y_pred_f, dim=1))

    if len(intersections) > 0:
        intersections = torch.stack(intersections)
        unions = torch.stack(unions)

    if reduce:
        return (torch.mean(torch.sum(intersections, dim=0) /
                           torch.sum(unions, dim=0)))
    else:
        return intersections, unions


def dice_coefficient_loss(y_pred, y_true, valid=None, reduce=True, epsilon=1e-5):
    if reduce:
        return -dice_coefficient(y_pred, y_true, valid, reduce, epsilon)
    else:
        numerator, denominator = dice_coefficient(y_pred, y_true, valid, reduce, epsilon)
        return -numerator, denominator


def cross_entropy_loss(y_pred, y_true, valid=None, reduce=True, class_weight=None):

    # Reshape into 2D image, which pytorch can handle
    y_true_f = y_true.view(y_true.size(0), y_true.size(2), -1)
    y_pred_f = y_pred.view(y_pred.size(0), y_pred.size(1), y_pred.size(2), -1)


    loss_per_voxel = torch.nn.functional.cross_entropy(
        y_pred_f, y_true_f, reduce=False, weight=class_weight).view(y_true.shape).squeeze()

    if valid is not None:
        mask = get_mask(loss_per_voxel.shape, valid)

        if reduce:
            return loss_per_voxel[mask].mean()
        else:
            loss_per_voxel_sums = []
            loss_per_voxel_norm_consts = []
            for i in range(y_pred.shape[0]):
                loss_per_voxel_masked = loss_per_voxel[i][mask[i]]
                if len(loss_per_voxel_masked.shape) > 0:
                    loss_per_voxel_sums.append(torch.sum(loss_per_voxel_masked))
                    loss_per_voxel_norm_consts.append(torch.LongTensor([loss_per_voxel_masked.shape[0]]))
            return (torch.cat(loss_per_voxel_sums),
                    torch.cat(loss_per_voxel_norm_consts))
    else:
        if reduce:
            return loss_per_voxel.view(-1).mean()
        else:
            return (torch.sum(loss_per_voxel, dim=1),
                    torch.LongTensor([loss_per_voxel.size(0)]).repeat(loss_per_voxel.shape[0]))


def get_mask(image_shape, index):
    if torch.cuda.is_available():
        mask = torch.cuda.ByteTensor(*image_shape).fill_(0)
    else:
        mask = torch.zeros(image_shape).byte()

    for i in range(index.shape[0]):
        if ((index[i, 1, :] - index[i, 0, :]) > 0).all():
            mask[i,
                 index[i, 0, 0]:index[i, 1, 0],
                 index[i, 0, 1]:index[i, 1, 1],
                 index[i, 0, 2]:index[i, 1, 2]] = 1
    return mask


def infer(model, loader, loss_function):
    model.eval()
    losses_numerator = []
    losses_denominator = []
    out_images = []
    for i in range(len(loader.dataset.unpadded_data_shape)):
        out_images.append(np.full(loader.dataset.unpadded_data_shape[i], -1))
    for i, (data, target, img_index, patch_index, valid) in enumerate(loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data)
        y = torch.autograd.Variable(target)
        out = model(x)

        _, out_predict = torch.max(out, 1)
        mask = get_mask(out_predict.shape, valid)
        patch_index = patch_index.cpu().numpy()
        for j in range(out.size(0)):
            out_predict_masked = out_predict[j][mask[j]]
            patch_start = patch_index[j,0] + valid[j,0]
            patch_end = patch_start + (valid[j,1]-valid[j,0])
            if (patch_end-patch_start > 0).all():
                out_images[img_index[j]][patch_start[0]:patch_end[0],
                                         patch_start[1]:patch_end[1],
                                         patch_start[2]:patch_end[2]] = out_predict_masked.view((valid[j,1] - valid[j,0]).tolist()).data.cpu().numpy()

        numerator, denominator = loss_function(out, y, valid=valid, reduce=False)
        try:
            numerator = numerator.data
            denominator = denominator.data
        except:
            pass
        losses_numerator.append(numerator.cpu().numpy())
        losses_denominator.append(denominator.cpu().numpy())

        # print(np.mean(np.sum(losses_numerator[-1], axis=0)/np.sum(losses_denominator[-1], axis=0)), loss_function(out, y, valid).data.cpu().numpy())

    # Check that entire image was filled in
    for out_image in out_images:
        assert not (out_image == -1).any()

    losses_numerator = np.concatenate(losses_numerator)
    losses_denominator = np.concatenate(losses_denominator)
    loss = np.mean(np.sum(losses_numerator, axis=0) / np.sum(losses_denominator, axis=0))
    return out_images, loss


model_classes = {"basic_k5":
                 partial(BasicModel, filter_size=5),
                 "se3_basic_k5":
                 partial(SE3BasicModel, filter_size=5),
                 "unet_k5":
                 partial(UnetModel, filter_size=5),
                 "se3_unet_k5":
                 partial(SE3UnetModel, filter_size=5)
                 }

def main(args):

    torch.backends.cudnn.benchmark = True

    train_filter = ["1000_3",
                    "1001_3",
                    "1002_3",
                    "1006_3",
                    "1007_3",
                    "1008_3",
                    "1009_3",
                    "1010_3",
                    "1011_3",
                    "1012_3",
                    "1013_3",
                    "1014_3"
                    ]
    validation_filter = ["1015_3",
                         "1017_3",
                         "1036_3"
                         ]
    test_filter = ["1003_3",
                   "1004_3",
                   "1005_3",
                   "1018_3",
                   "1019_3",
                   "1023_3",
                   "1024_3",
                   "1025_3",
                   "1038_3",
                   "1039_3",
                   "1101_3",
                   "1104_3",
                   "1107_3",
                   "1110_3",
                   "1113_3",
                   "1116_3",
                   "1119_3",
                   "1122_3",
                   "1125_3",
                   "1128_3"]

    # Check that sets are non-overlapping
    assert len(set(validation_filter).intersection(train_filter)) == 0
    assert len(set(test_filter).intersection(train_filter)) == 0

    if args.mode == 'train':
        train_set = MRISegmentation(args.data_filename,
                                    patch_shape=args.patch_size,
                                    filter=train_filter,
                                    log10_signal=args.log10_signal)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=False,
                                                   drop_last=True)
        np.set_printoptions(threshold=np.nan)
        print(np.unique(train_set.labels[0]))

    if args.mode in ['train', 'validate']:
        validation_set = MRISegmentation(args.data_filename,
                                         patch_shape=args.patch_size,
                                         filter=validation_filter,
                                         randomize_patch_offsets=False,
                                         log10_signal=args.log10_signal)
        validation_loader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        pin_memory=False,
                                                        drop_last=False)

    if args.mode == 'test':
        test_set = MRISegmentation(args.data_filename,
                                   patch_shape=args.patch_size,
                                   filter=test_filter,
                                   randomize_patch_offsets=False,
                                   log10_signal=args.log10_signal)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=False,
                                                  drop_last=False)


    output_size = 135
    model = model_classes[args.model](output_size=output_size)
    if torch.cuda.is_available():
        model.cuda()

    print("The model contains {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer.zero_grad()

    loss_function = None
    if args.loss == "dice":
        loss_function = dice_coefficient_loss
    elif args.loss == "cross_entropy":
        if args.class_weighting:
            class_weight = torch.Tensor(1/train_set.class_count)
            class_weight *= np.sum(train_set.class_count)/len(train_set.class_count)
            if torch.cuda.is_available():
                class_weight = class_weight.cuda()
        else:
            class_weight = None
        # loss_function = lambda *x: cross_entropy_loss(*x, class_weight=class_weight)
        from functools import partial
        loss_function = partial(cross_entropy_loss, class_weight=class_weight)

    # Set the logger
    if args.log_to_tensorboard:
        from datetime import datetime
        now = datetime.now()
        logger = Logger('./logs/%s/' % now.strftime("%Y%m%d_%H%M%S"))

    epoch_start_index = 0
    if args.mode == 'train':

        for epoch in range(epoch_start_index, args.training_epochs):

            training_losses = []
            training_accs = []
            for batch_idx, (data, target, img_index, patch_index, valid) in enumerate(train_loader):

                model.train()
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                x, y = torch.autograd.Variable(data), torch.autograd.Variable(target)

                out = model(x)

                # # Compare dice implementations
                # print("dice original impl: ",
                #       dice_coefficient_orig(out, y).data.cpu().numpy())
                # print("dice original impl - binary: ",
                #       dice_coefficient_orig_binary(out, y).data.cpu().numpy())
                # print("dice new impl - on class label: ",
                #       dice_coefficient(out, y).data.cpu().numpy())

                time_start = time.perf_counter()
                loss = loss_function(out, y)
                training_losses.append(loss)
                loss.backward()
                if batch_idx % args.batchsize_multiplier == args.batchsize_multiplier-1:
                    optimizer.step()
                    optimizer.zero_grad()

                binary_dice_acc = dice_coefficient_orig_binary(out, y, y_pred_is_dist=True).data[0]
                training_accs.append(binary_dice_acc)

                print("[{}:{}/{}] loss={:.4} acc={:.4} time={:.2}".format(
                    epoch, batch_idx, len(train_loader),
                    float(loss.data[0]), binary_dice_acc,
                    time.perf_counter() - time_start))

            acc_avg = np.mean(training_accs)
            loss_avg = np.mean(training_losses)

            print('TRAINING SET [{}:{}/{}] loss={:.4} acc={:.2}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                loss_avg.data[0], acc_avg))


            validation_ys, validation_loss = infer(model,
                                                   validation_loader,
                                                   loss_function)

            # Calculate binary dice score on predicted images
            numerators = []
            denominators = []
            for i in range(len(validation_set.data)):
                y_true = torch.LongTensor(validation_set.get_original(i)[1])
                y_pred = torch.LongTensor(validation_ys[i])
                if torch.cuda.is_available():
                    y_true = y_true.cuda()
                    y_pred = y_pred.cuda()
                numerator, denominator = dice_coefficient_orig_binary(
                    y_pred.unsqueeze(0),
                    y_true.unsqueeze(0),
                    classes=output_size,
                    reduce=False)
                numerators.append(numerator)
                denominators.append(denominator)
            numerators = torch.cat(numerators)
            denominators = torch.cat(denominators)
            validation_binary_dice_acc = torch.mean(
                torch.sum(numerators, dim=0) /
                (torch.sum(denominators, dim=0)))

            print('VALIDATION SET [{}:{}/{}] loss={:.4} acc={:.2}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                validation_loss, validation_binary_dice_acc))

            if args.log_to_tensorboard:

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'training set avg loss': loss_avg.data[0],
                    'training set accuracy': acc_avg,
                    'validation set avg loss': validation_loss,
                    'validation set accuracy': validation_binary_dice_acc,
                }

                step = epoch
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step + 1)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(),
                                         step + 1)


            # Adjust patch indices at end of each epoch
            train_set.initialize_patch_indices()



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-filename", required=True,
                        help="The name of the data file.")
    parser.add_argument("--model", choices=model_classes.keys(), required=True,
                        help="Which model definition to use")
    parser.add_argument("--patch-size", default=64, type=int,
                        help="Size of patches (default: %(default)s)")
    parser.add_argument("--loss", choices=['dice', 'dice_onehot', 'cross_entropy'],
                        default="cross_entropy",
                        help="Which loss function to use(default: %(default)s)")
    parser.add_argument("--mode", choices=['train', 'test', 'validate'],
                        default="train",
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--randomize-orientation", action="store_true", default=False,
                        help="Whether to randomize the orientation of the structural input during training (default: %(default)s)")
    parser.add_argument("--batch-size", default=2, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--log10-signal", action="store_true", default=False,
                        help="Whether to logarithmize the MIR scan signal (default: %(default)s)")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--class-weighting", action='store_true', default=False,
                        help="switches on class weighting, only used in cross entropy loss (default: %(default)s)")
    parser.add_argument("--log-to-tensorboard", action="store_true", default=False,
                        help="Whether to output log information in tensorboard format (default: %(default)s)")

    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    main(args=args)

