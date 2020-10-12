#!/usr/bin/env python3

"""Find the sources in a collection FITS images."""

import matplotlib
matplotlib.use('TkAgg')

#Switching backend must be at top
#pylint: disable=wrong-import-position
#pylint: disable=wrong-import-order
import tkinter
import tkinter.messagebox
import tkinter.ttk
import logging
#pylint: enable=wrong-import-order

import matplotlib.pyplot
import PIL.Image
import PIL.ImageDraw
import PIL.ImageTk
from astropy.io import fits

from command_line_util import get_default_frame_processing_cmdline

from superphot_pipeline.image_utilities import\
    fits_image_generator,\
    zscale_image
from superphot_pipeline import SourceFinder
#pylint: enable=wrong-import-position

def parse_configuration(default_config_files=('find_sources.cfg',),
                        default_fname_pattern='%(FITS_ROOT)s.srcextract'):
    """Return the configuration to use for splitting by channel."""

    parser = get_default_frame_processing_cmdline(__doc__,
                                                  default_config_files,
                                                  default_fname_pattern)

    parser.add_argument(
        '--tool',
        default='fistar',
        choices=['fistar', 'hatphot', 'mock'],
        help='What tool to use for fintding sources in the images.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1000.0,
        help='The threshold to use as the faind limit of the sources. If '
        '--tool is `fistar` this is in units of ADU, if --tool is `hatphot` '
        'this is in units of standard deviations.'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='If this argument is passed, only the first image is processed '
        'but the user is presented with a dailog to change the threshold and '
        'the detected sources are shown on top of the image in zscale.'
    )

    return parser.parse_args()

def mark_extracted_sources(image,
                           sources,
                           shape='ellipse',
                           size=10,
                           **shape_format):
    """
    Annotate the given image to show the positions of the given sources.

    Args:
        image(PIL.ImageDraw):    The image to draw shapes on to indicate the
            positions of sources.

        sources(numpy array):    The sources to annotate on the
            image. Must be iterable and each entry must support indexing by name
            with at least `'x'` and `'y'' keys defined.

        shape(str):    The shape to draw. Either `'ellipse'` or `'rectangle'`.

        shape_format:    Any keyword arguments to pass directly to the
            corresponding method of the image (e.g. `outline`, or `width`).

    Returns
        None
    """

    mark_source = getattr(image, shape)
    for source in sources:
        mark_source([source['x'] - size / 2, source['y'] - size/2,
                     source['x'] + size / 2, source['y'] + size/2],
                    **shape_format)

#Out of my control
#pylint: disable=too-many-ancestors
class SourceExtractionTuner(tkinter.Frame):
    """Application for manually tuning source extraction."""

    def _display_image(self, sources):
        """Display the image, annotated to show the given sources."""

        annotated_image = self._image['zscaled'].copy()
        mark_extracted_sources(PIL.ImageDraw.Draw(annotated_image),
                               sources,
                               outline='green',
                               width=3)
        self._image['photo'] = PIL.ImageTk.PhotoImage(
            annotated_image,
            master=self._widgets['canvas']
        )
        self._widgets['canvas'].create_image(0,
                                             0,
                                             image=self._image['photo'],
                                             anchor='nw')

    def _update(self):
        """Re-extract sources and mark them on the image."""

        threshold = self._widgets['threshold_entry'].get()
        try:
            threshold = float(threshold)
        except ValueError:
            tkinter.messagebox.showinfo(
                message='Threshold specified (%s) not a valid number.'
                %
                repr(threshold)
            )
            return
        if threshold <= 0:
            tkinter.messagebox.showinfo(
                message='Invalid threshold specified: %s. Must be positive.'
                %
                repr(threshold)
            )
        self._display_image(self.find_sources(self._fits_images[0]))

    def _create_widgets(self):
        """Return a dictionary of all the widgets needed."""

        result = dict(
            xscroll=tkinter.ttk.Scrollbar(self, orient=tkinter.HORIZONTAL),
            yscroll=tkinter.ttk.Scrollbar(self, orient=tkinter.VERTICAL),
            controls_frame=tkinter.Frame(self)
        )
        result['canvas'] = tkinter.Canvas(
            self,
            scrollregion=(0,
                          0,
                          self._image['data'].shape[1],
                          self._image['data'].shape[0]),
            xscrollcommand=result['xscroll'],
            yscrollcommand=result['yscroll']
        )
        result['xscroll']['command'] = result['canvas'].xview
        result['yscroll']['command'] = result['canvas'].yview

        result['threshold_entry'] = tkinter.Entry(
            result['controls_frame'],
            width=100
        )
        result['threshold_entry'].insert(0, repr(self.configuration.threshold))
        result['update_button'] = tkinter.Button(
            result['controls_frame'],
            text='Update',
            command=self._update
        )
        return result

    def _arrange_widgets(self):
        """Arranges the widgets using the tkinter grid geometry manager."""

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._widgets['canvas'].grid(
            column=0,
            row=0,
            sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S)
        )
        self._widgets['xscroll'].grid(
            column=0,
            row=1,
            sticky=(tkinter.E, tkinter.W),
        )
        self._widgets['yscroll'].grid(
            column=1,
            row=0,
            sticky=(tkinter.N, tkinter.S),
        )
        self._widgets['controls_frame'].grid(column=0, row=2)

        self._widgets['threshold_entry'].grid(column=0, row=0)
        self._widgets['update_button'].grid(column=1, row=0)

    def quit(self):
        """Exit the application."""

        self.master.quit()     # stops mainloop
        self.master.destroy()  # this is necessary on Windows to prevent
                               # Fatal Python Error: PyEval_RestoreThread:
                               # NULL tstate

    def __init__(self, master, configuration):
        """Set-up the user controls and display the image."""

        super().__init__(master)
        self.master = master

        self.configuration = configuration

        self.find_sources = SourceFinder(tool=configuration.tool,
                                         threshold=configuration.threshold,
                                         allow_overwrite=True,
                                         allow_dir_creation=True)

        self._fits_images = list(fits_image_generator(configuration.images))
        with fits.open(self._fits_images[0], 'readonly') as fits_image:
            self._image = dict(
                #False positive
                #pylint: disable=no-member
                data=fits_image[0 if fits_image[0].header['NAXIS'] else 1].data,
                #pylint: enable=no-member
            )

        self._widgets = self._create_widgets()

        self._image['zscaled'] = PIL.Image.fromarray(
            zscale_image(self._image['data']),
            'L'
        ).convert('RGB')

        self._arrange_widgets()

        self._update()

#pylint: enable=too-many-ancestors

def tune(configuration):
    """Allow the user to tune the source extraction threshold visually."""

    main_window = tkinter.Tk()
    ttk_style = tkinter.ttk.Style()
    ttk_style.theme_use('classic')
    main_window.columnconfigure(0, weight=1)
    main_window.rowconfigure(0, weight=1)
    main_window.wm_title("Source Extraction Tuning")
    SourceExtractionTuner(main_window, configuration).grid(
        row=0,
        column=0,
        sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S)
    )
    main_window.mainloop()

def main(configuration):
    """Do not pollute global namespace."""

    logging.basicConfig(level=getattr(logging, configuration.log_level))
    if configuration.tune:
        tune(configuration)

if __name__ == '__main__':
    main(parse_configuration())
