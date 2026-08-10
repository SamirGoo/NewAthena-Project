from astropy.io import fits
import numpy as np


class AthenaWFI:

    def __init__(self, rsp_file, pha_file):

        self.rsp_file = rsp_file
        self.pha_file = pha_file

        self._load_response()
        self._load_background()

    def _load_response(self):

        hdul = fits.open(self.rsp_file)

        rmf = hdul["MATRIX"].data

        self.energy_lo = rmf["ENERG_LO"]
        self.energy_hi = rmf["ENERG_HI"]

        self.energy = (
            self.energy_lo +
            self.energy_hi
        ) / 2

        self.delta_e = (
            self.energy_hi -
            self.energy_lo
        )

        self.matrix = np.vstack(rmf["MATRIX"])

        hdul.close()

    def _load_background(self):

        hdul = fits.open(self.pha_file)

        spec = hdul["SPECTRUM"].data

        self.background_rate = np.array(
            spec["RATE"]
        )

        hdul.close()

    def fold(
        self,
        photon_flux,
        exposure,
    ):
        """
        photon_flux:
            photons/cm^2/s/keV
        """

        photons_per_bin = (
            photon_flux *
            self.delta_e *
            exposure
        )

        counts = (
            self.matrix.T
            @ photons_per_bin
        )

        return counts

    def background_counts(
        self,
        exposure,
    ):

        return (
            self.background_rate *
            exposure
        )
