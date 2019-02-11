from matplotlib import pyplot as plt
import numpy as np
import time

import torchvision.utils as vutils

results_folder = '../../results/'


def imsave(batch, device, title):
    # Plot some training images
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu().numpy(), (1, 2, 0))
    )
    plt.savefig(+ title)


def timeit(f):
    def timed(*args, **kwargs):
        s = time.time()
        result = f(*args, **kwargs)
        e = time.time()

        print(f.__name__, 'took (seconds):', e - s)

        return result

    return timed
