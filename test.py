from astropy.io import fits

with fits.open('average_psfmap.fits', mode='readonly') as fits:
    print(fits[0].header)
