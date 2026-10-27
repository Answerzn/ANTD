"""Input checks shared by ANTD training and evaluation."""

import os
import re

from utils import utils_image as util


def paired_image_paths(noisy_root, clean_root):
    """Match relative names, including official SIDD Noisy/GT naming variants."""
    def indexed_paths(root):
        paths = util.get_image_paths(root)
        if not paths:
            raise ValueError('No images found in: ' + str(root))
        indexed = {}
        for path in paths:
            key = os.path.splitext(os.path.relpath(path, root))[0].replace('\\', '/').lower()
            key = key.replace('validationnoisyblocks', 'validationblocks')
            key = key.replace('validationgtblocks', 'validationblocks')
            key = re.sub(r'(^|[/_])(noisy|noise|gt)(?=[/_]|$)', r'\1paired', key)
            if key in indexed:
                raise ValueError('Duplicate image-pair identifier: ' + path)
            indexed[key] = path
        return indexed

    noisy, clean = indexed_paths(noisy_root), indexed_paths(clean_root)
    if noisy.keys() != clean.keys():
        missing = sorted(noisy.keys() ^ clean.keys())[:5]
        raise ValueError('Noisy/GT names do not match. Use identical relative names or SIDD NOISY/GT names. '
                         'Unmatched identifiers: ' + ', '.join(missing))
    keys = sorted(noisy)
    return [noisy[key] for key in keys], [clean[key] for key in keys]


def validate_image_size(height, width):
    # Current attention/padding path expects images at least 64 x 64.
    if min(height, width) < 64:
        raise ValueError('ANTD expects images at least 64 x 64 pixels.')
