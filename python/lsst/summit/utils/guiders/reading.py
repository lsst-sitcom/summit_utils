# This file is part of summit_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "read_guider_data"
]


import numpy as np
from astropy.io import fits
from lsst.resources import ResourcePath

def read_guider_data(filename):
    rp = ResourcePath(filename)
    
    with rp.open(mode="rb") as f:
        
        hdu_list = fits.open(f)
        
        N_frames = hdu_list[0].header['N_STAMPS']
        x, y = hdu_list[0].header['ROIROWS'], hdu_list[0].header['ROICOLS']

        imgs = np.zeros((N_frames, x, y))
        timestamps = np.zeros(N_frames)

        if (len(hdu_list) - 1)/2 == N_frames:  ## if has rawstamps
            for i, j in zip(range(N_frames), range(2, len(hdu_list), 2)):
                imgs[i, :, :] = hdu_list[j].data
                timestamps[i] = hdu_list[j].header['STMPTMJD']

        elif len(hdu_list)-1 == N_frames:     ## if no rawstamps
            for i, j in zip(range(N_frames), range(1, len(hdu_list))):
                imgs[i, :, :] = hdu_list[j].data
                timestamps[i] = hdu_list[j].header['STMPTMJD']
    
    grand_hdr = hdu_list[0].header

    return imgs, timestamps, grand_hdr
