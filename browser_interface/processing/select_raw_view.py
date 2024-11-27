"""Define the view allowing users to add new raw images for processing."""

from os import path, scandir
import fnmatch
import re
from traceback import print_exc

from django.views import View
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from autowisp.file_utilities import find_fits_fnames
from autowisp.database.image_processing import ImageProcessingManager


class SelectRawImages(View):
    """A view for selecting raw images to add for processing."""

    @staticmethod
    def _get_context(config, search_dir):
        """Return te context required by the file selection template."""

        print(f'Config: {config!r}')
        result = {}
        filename_check = config.get('filename_filter', r'.*\.fits(.fz)?\Z')
        result['filename_filter'] = filename_check
        result['filename_filter_type'] = config.get('filefilter_type',
                                                    'Regular Expression')
        if result['filename_filter_type'] != 'Regular Expression':
            filename_check = fnmatch.translate(filename_check)
        try:
            filename_check = re.compile(filename_check)
        except re.error:
            filename_check = re.compile('')

        dirname_check = config.get('dirname_filter', r'[^.]')
        result['dirname_filter'] = dirname_check
        result['dirname_filter_type'] = config.get('dirfilter_type',
                                                   'Regular Expression')
        if result['dirname_filter_type'] != 'Regular Expression':
            dirname_check = fnmatch.translate(dirname_check)
        try:
            dirname_check = re.compile(dirname_check)
        except re.error:
            print(f'Invalid REX: {dirname_check!r}')
            dirname_check = re.compile('')

        if search_dir is None:
            search_dir = config.get("currentdir", path.expanduser('~'))
            if 'enter_dir' in config:
                search_dir = path.join(search_dir, config['enter_dir'])
        result['file_list'] = []
        result['dir_list'] = []
        with scandir(search_dir) as dir_entries:
            for entry in dir_entries:
                if entry.is_dir():
                    if dirname_check.match(entry.name):
                        result['dir_list'].append(entry.name)
                elif filename_check.match(entry.name):
                    result['file_list'].append(entry.name)

        result['file_list'].sort()
        result['dir_list'].sort()

        head = path.abspath(search_dir)
        parent_dir_list = [('/', 'Computer')]
        while head and head != '/':
            parent_dir_list.insert(1, (head, path.basename(head)))
            head = path.dirname(head)

        result['parent_dir_list'] = parent_dir_list

        print(f'Context: {result!r}')
        return result


    def get(self, request, dirname=None):
        """Display the interface for selecting files."""

        return render(
            request,
            'processing/select_raw_images.html',
            self._get_context(request.GET, dirname)
        )


    def post(self, request, *_args, **_kwargs):
        """Respond to user changing file selection configuration."""

        print(f'POST: {request.POST!r}')
        dir_name = request.POST['currentdir']
        image_list = []
        selected = request.POST['selected']
        if isinstance(selected, str):
            selected = [selected]
        for item_name in selected:
            full_path = path.join(dir_name, item_name)
            if path.isdir(full_path):
                print(f'Adding images under: {full_path!r}')
                image_list.extend(find_fits_fnames(full_path))
            else:
                print(f'Adding single image: {full_path!r}')
                assert path.isfile(full_path)
                image_list.append(full_path)

            try:
                ImageProcessingManager().add_raw_images(image_list)
            except OSError:
                print_exc()
                return HttpResponseRedirect(
                    reverse('processing:select_raw_images')
                )


        return HttpResponseRedirect(
            reverse('processing:progress')
        )
