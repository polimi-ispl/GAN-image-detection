import os
from collections import OrderedDict

import numpy as np
import torch

torch.manual_seed(21)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import albumentations as A
import albumentations.pytorch as Ap
from utils import architectures
from PIL import Image


class Detector:
    def __init__(self):

        # model directory and path for detector A
        # model_A_dir = 'weights/method_A/net-EfficientNetB4_lr-0.001_img_aug-[\'flip\', \'rotate\', \'clahe\', \'blur\', ' \
        #               '\'brightness&contrast\', \'jitter\', \'downscale\', \'hsv\', \'resize\', \'jpeg\']' \
        #               '_img_aug_p-0.5_patch_size-128_patch_number-1_batch_size-250_num_classes-2'
        #
        # # model directory and path for detector B
        # model_B_dir = 'weights/method_B/net-EfficientNetB4_lr-0.001_aug-[\'flip\', \'rotate\', \'clahe\', \'blur\', ' \
        #               '\'crop&resize\', \'brightness&contrast\', \'jitter\', \'downscale\', \'hsv\']' \
        #               '_aug_p-0.5_jpeg_aug_p-0.7_patch_size-128_patch_number-1_batch_size-250_num_classes-2'
        #
        # # model directory and path for detector C
        # model_C_dir = 'weights/method_C/net-EfficientNetB4_lr-0.001_aug-[\'flip\', \'rotate\', \'clahe\', \'blur\',' \
        #               ' \'crop&resize\', \'brightness&contrast\', \'jitter\', \'downscale\', \'hsv\']' \
        #               '_aug_p-0.5_jpeg_aug_p-0_patch_size-128_patch_number-5_batch_size-50_num_classes-2'
        #
        # # model directory and path for detector D
        # model_D_dir = 'weights/method_D/net-EfficientNetB4_lr-0.001_aug-[\'flip\', \'rotate\', \'clahe\', \'blur\',' \
        #               '\'crop&resize\', \'brightness&contrast\', \'jitter\', \'downscale\', \'hsv\']' \
        #               '_aug_p-0.5_jpeg_aug_p-0_patch_size-128_patch_number-10_batch_size-25_num_classes-2'
        #
        # # model directory for detector E
        # model_E_dir = 'weights/method_E/net-EfficientNetB4_lr-0.001_aug-[\'flip\', \'rotate\', \'clahe\', \'blur\',' \
        #               ' \'crop&resize\', \'brightness&contrast\', \'jitter\', \'downscale\', \'hsv\']' \
        #               '_aug_p-0.5_jpeg_aug_p-0.7_patch_size-128_patch_number-1_batch_size-250_num_classes-2'

        self.weights_path_list = [os.path.join('weights', f'method_{x}.pth') for x in 'ABCDE']
        # self.model_path = os.path.join(model_dir, 'bestval.pth')

        # GPU configuration if available
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []
        for i in range(5):
            # Instantiate and load network
            network_class = getattr(architectures, 'EfficientNetB4')
            net = network_class(n_classes=2, pretrained=False).eval().to(self.device)
            print('Loading model...')
            state_tmp = torch.load(self.weights_path_list[i], map_location='cpu')
            if 'net' not in state_tmp.keys():
                state = OrderedDict({'net': OrderedDict()})
                [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
            else:
                state = state_tmp
            incomp_keys = net.load_state_dict(state['net'], strict=True)
            print(incomp_keys)
            print('Model loaded!')

            self.nets += [net]

        net_normalizer = net.get_normalizer()  # pick normalizer from last network
        transform = [
            A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),
            Ap.transforms.ToTensorV2()
        ]
        self.trans = A.Compose(transform)

        self.cropper = A.RandomCrop(width=128, height=128, always_apply=True, p=1.)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def synth_real_detector(self, img_path: str, n_patch: int = 50):

        # Load image:
        img = np.asarray(Image.open(img_path))

        # Optout if image is non conforming
        if img.shape == ():
            print('{} None dimension'.format(img_path))
            return None
        if img.shape[0] < 128 or img.shape[1] < 128:
            print('Too small image')
            return None
        if img.ndim != 3:
            print('RGB images only')
            return None
        if img.shape[2] > 3:
            print('Omitting alpha channel')
            img = img[:, :, :3]

        # Extract test_N random patches from image:
        patch_list = [self.cropper(image=img)['image'] for _ in range(n_patch)]

        # Normalization
        transf_patch_list = [self.trans(image=patch)['image'] for patch in patch_list]

        # Compute scores
        transf_patch_tensor = torch.stack(transf_patch_list, dim=0).to(self.device)
        with torch.no_grad():
            patch_scores = self.net(transf_patch_tensor)
        softmax_scores = torch.softmax(patch_scores, dim=1)
        predictions = torch.argmax(softmax_scores, dim=1)

        # Majority voting on patches
        if sum(predictions) > len(predictions) // 2:
            majority_voting = 1
        else:
            majority_voting = 0

        # get an output score from softmax scores:
        # LLR < 0: real
        # LLR > 0: synthetic

        sign_predictions = majority_voting * 2 - 1
        # select only the scores associated with the estimated class (by majority voting)
        softmax_scores = softmax_scores[:, majority_voting]
        normalized_prediction = torch.max(softmax_scores).item() * sign_predictions

        return normalized_prediction


def main():
    # img_path
    img_path = "/nas/public/exchange/semafor/eval1/stylegan2/100k-generated-images/car-512x384_cropped/stylegan2-" \
               "config-f-psi-0.5/097000/097001.png"

    # number of random patches to extract from images
    test_N = 50

    detector = Detector()
    detector.synth_real_detector(img_path, test_N)

    return 0


if __name__ == '__main__':
    main()
