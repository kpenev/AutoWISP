# import argparse
import sys

# import configargparse

sys.path.append("..") #from parent directory import...
# from superphot_pipeline.processing_steps import fit_star_shape

from superphot_pipeline.processing_steps import __all__ as all_steps

if __name__ == '__main__':
    # print(type(all_steps))
    # fit_star_shape.parse_command_line("-c PANOPTES_R.cfg --photometry-catalogue dummy")

    for x in all_steps:
        if(hasattr(x, 'parse_command_line')):
            print(x.__name__)
            try:
                x.parse_command_line(['-c', 'PANOPTES_R.cfg'])
            except BaseException:   #ok to use BaseException? too broad, Configargparse.ArgumentException was not it
                x.parse_command_line("-c PANOPTES_R.cfg --photometry-catalogue dummy")

