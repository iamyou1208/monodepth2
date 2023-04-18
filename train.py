# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer, Trainer_Carla
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    print(opts)
    if opts.dataset == "carla":
        trainer = Trainer_Carla(opts)
    else:
        trainer = Trainer(opts)
    trainer.train()
