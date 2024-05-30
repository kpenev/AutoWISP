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
import numpy

from astrowisp.utils.file_utilities import\
    get_fits_fname_root,\
    prepare_file_output


from command_line_util import get_default_frame_processing_cmdline

from autowisp.image_utilities import zscale_image
from autowisp.file_utilities import find_fits_fnames
from autowisp.fits_utilities import read_image_components
from autowisp import SourceFinder, Evaluator
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
        '--filter',
        default=None,
        help='Apply a filter to the extracted sources to discard artefacts. '
        'Should be an expression involving extracted source columns that '
        'evaluates to True (keep) or False (discard).'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='If this argument is passed, only the first image is processed '
        'but the user is presented with a dailog to change the threshold and '
        'the detected sources are shown on top of the image in zscale.'
    )
    parser.add_argument(
        '--save-srcfind-snapshots',
        default=None,
        help='A %%-substitution pattern to save jpegs showing the extracted '
        'sources.'
    )

    return parser.parse_args()

def mark_extracted_sources(image,
                           sources,
                           filter_expression=None,
                           shape='ellipse',
                           size=15,
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
            corresponding method of the image (e.g. `width`).

    Returns
        None
    """

    draw_on_image = PIL.ImageDraw.Draw(image)
    mark_source = getattr(draw_on_image, shape)
    if filter_expression is None:
        selected = numpy.full(sources.shape, True)
    else:
        selected = Evaluator(sources)(filter_expression)

    for flagged, color in [(selected, 'lightgreen'),
                           (numpy.logical_not(selected), 'salmon')]:
        for source in sources[flagged]:
            mark_source([source['x'] - size / 2, source['y'] - size/2,
                         source['x'] + size / 2, source['y'] + size/2],
                        outline=color,
                        **shape_format)

#Out of my control
#pylint: disable=too-many-ancestors
class SourceExtractionTuner(tkinter.Frame):
    """Application for manually tuning source extraction."""

    def _display_image(self, sources, filter_expression):
        """Display the image, annotated to show the given sources."""

        annotated_image = self._image['zscaled'].copy()
        mark_extracted_sources(annotated_image,
                               sources,
                               filter_expression=filter_expression)
        self._image['photo'] = PIL.ImageTk.PhotoImage(
            annotated_image,
            master=self._widgets['canvas']
        )
        self._widgets['canvas'].create_image(0,
                                             0,
                                             image=self._image['photo'],
                                             anchor='nw')
        self._widgets['nsources_label']['text'] = str(sources.size)

    def _update(self):
        """Re-extract sources and mark them on the image."""

        threshold = self._widgets['threshold_entry'].get()
        try:
            threshold = float(threshold)
        except ValueError:
            tkinter.messagebox.showinfo(
                message=f'Threshold specified ({threshold!r}) '
                'not a valid number.'
            )
            return
        if threshold <= 0:
            tkinter.messagebox.showinfo(
                message=f'Invalid threshold specified: {threshold!r}. '
                'Must be positive.'
            )

        filter_expression = self._widgets['filter_entry'].get()
        if not filter_expression.strip():
            filter_expression = None

        self._display_image(
            self.find_sources(self._fits_images[0],
                              brightness_threshold=threshold),
            filter_expression=filter_expression
        )

    def _create_active_widgets(self):
        """Return dictionary of all the widgets that will be udptade by app."""

        result = {
            'xscroll': tkinter.ttk.Scrollbar(self, orient=tkinter.HORIZONTAL),
            'yscroll': tkinter.ttk.Scrollbar(self, orient=tkinter.VERTICAL),
            'controls_frame': tkinter.Frame(self)
        }
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

        result['filter_entry'] = tkinter.Entry(
            result['controls_frame']
        )
        if self.configuration.filter:
            result['filter_entry'].insert(0, repr(self.configuration.filter))

        result['update_button'] = tkinter.Button(
            result['controls_frame'],
            text='Update',
            command=self._update
        )
        result['nsources_label'] = tkinter.ttk.Label(result['controls_frame'],
                                                     text='')

        return result

    def _arrange_widgets(self):
        """Arranges the widgets using grid geometry manager and add passive."""

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

        tkinter.ttk.Label(
            self._widgets['controls_frame'],
            text='Extracted source count:'
        ).grid(
            column=0, row=0, sticky=tkinter.E
        )
        self._widgets['nsources_label'].grid(column=1, row=0)

        tkinter.ttk.Label(
            self._widgets['controls_frame'],
            text='Threshold:'
        ).grid(
            column=0, row=1, sticky=tkinter.E
        )
        self._widgets['threshold_entry'].grid(column=1, row=1)

        tkinter.ttk.Label(
            self._widgets['controls_frame'],
            text='Filter expression:'
        ).grid(
            column=0, row=2, sticky=(tkinter.E, tkinter.W)
        )
        self._widgets['filter_entry'].grid(column=1, row=2)

        self._widgets['update_button'].grid(column=2, row=0, rowspan=3)

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

        self.find_sources = SourceFinder(
            tool=configuration.tool,
            brightness_threshold=configuration.threshold,
            allow_overwrite=True,
            allow_dir_creation=True
        )

        self._fits_images = list(find_fits_fnames(configuration.images))
        self._image = {
            #False positive
            #pylint: disable=no-member
            'data': read_image_components(self._fits_images[0],
                                          read_error=False,
                                          read_mask=False,
                                          read_header=False)[0]
            #pylint: enable=no-member
        }

        self._widgets = self._create_active_widgets()

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
        return

    find_sources = SourceFinder(
        tool=configuration.tool,
        brightness_threshold=configuration.threshold,
        allow_overwrite=configuration.allow_overwrite,
        allow_dir_creation=configuration.allow_dir_creation,
        always_return_sources=bool(configuration.save_srcfind_snapshots)
    )

    for image_fname in find_fits_fnames(configuration.images):
        #False positive
        #pylint: disable=unbalanced-tuple-unpacking
        image, header = read_image_components(image_fname,
                                              read_error=False,
                                              read_mask=False)
        #pylint: enable=unbalanced-tuple-unpacking

        fname_substitutions = dict(header,
                                   FITS_ROOT=get_fits_fname_root(image_fname))
        sources = find_sources(
            image_fname,
            configuration.outfname_pattern % fname_substitutions
        )
        if configuration.save_srcfind_snapshots:
            #False positive
            #pylint: disable=unbalanced-tuple-unpacking
            #pylint: enable=unbalanced-tuple-unpacking
            image = PIL.Image.fromarray(zscale_image(image), 'L').convert('RGB')
            mark_extracted_sources(image, sources, configuration.filter)
            snapshot_fname = (
                configuration.save_srcfind_snapshots
                %
                fname_substitutions
            )

            prepare_file_output(
                snapshot_fname,
                allow_existing=configuration.allow_overwrite,
                allow_dir_creation=configuration.allow_dir_creation,
                delete_existing=True
            )
            image.save(snapshot_fname)

if __name__ == '__main__':
    main(parse_configuration())
