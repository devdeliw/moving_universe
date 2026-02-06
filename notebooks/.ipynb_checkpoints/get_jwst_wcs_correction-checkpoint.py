from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u

def get_jwst_wcs_correction(epoch_dir):
  """
  Helper function to get JWST WCS correction (bulk offset in X and Y)
  to have it line up with HST/Gaia properly. These are keyed to the
  epoch direcotry names

  Works by getting the offset between a given star's actual WCS coordinate
  (based on HST/Gaia) and its JWST WCS coordinate (from the JWST image).
  We assume that this offset is the same for all filters within
  an epoch.

  Application:
  This offset must be ADDED to the JWST WCS coordinates. Units are arcsecs
  """
  epoch_dir = epoch_dir.split('/')[0]
  
  # True WCS coordinates, Skycoord Obj
  coord_true_dict = {'2022_08_10_dl20240617': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2022_09_19_dl20240617': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2023_03_17_dl20240617': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2023_07_24_dl20240617': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2023_09_23_dl20240617': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2024_09_19_dl20241106': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2022_08_10_dl20250618': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2022_09_19_dl20250618': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2023_03_17_dl20250618': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2023_07_24_dl20250618': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2023_09_23_dl20250618': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2024_09_19_dl20250618': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs'),
              '2025_09_15_dl20250918': SkyCoord('17h45m40.1045s', '-29d00m27.582s', frame='icrs')
              }
  
  # Orig JWST WCS Coordinates, Skycoord Obj
  coord_orig_dict = {'2022_08_10_dl20240617': SkyCoord('17h45m40.6803', '-29d00m25.745s', frame='icrs'),
              '2022_09_19_dl20240617': SkyCoord('17h45m39.421s', '-29d00m39.455s', frame='icrs'),
              '2023_03_17_dl20240617': SkyCoord('17h45m40.1233s', '-29d00m27.347s', frame='icrs'),
              '2023_07_24_dl20240617': SkyCoord('17h45m40.5827s', '-29d00m32.385s', frame='icrs'),
              '2023_09_23_dl20240617': SkyCoord('17h45m40.1676s', '-29d00m27.736s', frame='icrs'),
              '2024_09_19_dl20241106': SkyCoord('17h45m40.1101s', '-29d00m27.416s', frame='icrs'),
              '2022_08_10_dl20250618': SkyCoord('17h45m40.6803', '-29d00m25.745s', frame='icrs'),
              '2022_09_19_dl20250618': SkyCoord('17h45m39.421s', '-29d00m39.455s', frame='icrs'),
              '2023_03_17_dl20250618': SkyCoord('17h45m40.1233s', '-29d00m27.347s', frame='icrs'),
           '2023_07_24_dl20250618': SkyCoord('17h45m40.5827s', '-29d00m32.385s', frame='icrs'),
              '2023_09_23_dl20250618': SkyCoord('17h45m40.1676s', '-29d00m27.736s', frame='icrs'),
              '2024_09_19_dl20250618': SkyCoord('17h45m40.1101s', '-29d00m27.416s', frame='icrs'),
              '2025_09_15_dl20250918': SkyCoord('17h45m40.1105s', '-29d00m27.442s', frame='icrs'),
              }

  # Calculate offsets
  coord_we_want = coord_true_dict[epoch_dir]
  coord_orig_wcs = coord_orig_dict[epoch_dir]

  xoff, yoff = coord_orig_wcs.spherical_offsets_to(coord_we_want) # coord_orig + xoff = coord_we_want
  xoff = xoff.to(u.arcsec).value
  yoff = yoff.to(u.arcsec).value
  
  return xoff, yoff