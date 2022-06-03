"""Functions for detrending light curves (EPD or TFA)."""

def extract_target_lc(lc_fnames, target_id):
    """Return target LC fname, & LC fname list with the target LC removed."""

    for index, fname in enumerate(lc_fnames):
        if basename(fname).startswith(target_id):
            return lc_fnames.pop(index), lc_fnames
    raise ValueError('None of the lightcurves seems to be for the target.')

def add_catalogue_info(lc_fnames, catalogue_fname, magnitude_column, result):
    """Fill the catalogue information fields in result."""

    mem_dr = DataReductionFile()
    catalogue = read_master_catalogue(catalogue_fname,
                                      mem_dr.parse_hat_source_id)

    for lc_ind, fname in enumerate(lc_fnames):
        source_id = parse_lc_fname(fname)
        result[lc_ind]['ID'] = source_id
        cat_info = catalogue[source_id]
        result[lc_ind]['mag'] = cat_info[magnitude_column]
        result[lc_ind]['xi'] = cat_info['xi']
        result[lc_ind]['eta'] = cat_info['eta']

def correct_target_lc(target_lc_fname, cmdline_args, correct):
    """Perform reconstructive detrending on the target LC."""

    num_limbdark_coef = len(cmdline_args.limb_darkening)
    assert num_limbdark_coef == 2

    transit_parameters = (
        [cmdline_args.radius_ratio]
        +
        list(cmdline_args.limb_darkening)
        +
        [
            cmdline_args.mid_transit,
            cmdline_args.period,
            cmdline_args.scaled_semimajor,
            cmdline_args.inclination * numpy.pi / 180.0
        ]
    )
    if hasattr(cmdline_args, 'eccentricity'):
        transit_parameters.append(cmdline_args.eccentricity)
    if hasattr(cmdline_args, 'periastron'):
        transit_parameters.append(cmdline_args.periastron)

    fit_parameter_flags = numpy.zeros(len(transit_parameters), dtype=bool)

    param_indices = dict(depth=0,
                         limbdark=list(
                             range(1, num_limbdark_coef + 1)
                         ),
                         mid_transit=num_limbdark_coef + 1,
                         period=num_limbdark_coef + 2,
                         semimajor=num_limbdark_coef + 3,
                         inclination=num_limbdark_coef + 4,
                         eccentricity=num_limbdark_coef + 5,
                         periastron=num_limbdark_coef + 6)
    for to_fit in cmdline_args.mutable_transit_params:
        fit_parameter_flags[param_indices[to_fit]] = True

    return apply_reconstructive_correction_transit(
        target_lc_fname,
        correct,
        transit_model=QuadraticModel(),
        transit_parameters=numpy.array(transit_parameters),
        fit_parameter_flags=fit_parameter_flags,
        num_limbdark_coef=num_limbdark_coef
    )

def detrend_light_curves(cmdline_args, correct, output_statistics_fname):
    """Detrend all lightcurves and create statistics file."""

    if cmdline_args.target_id is not None:
        target_lc_fname, lc_fnames = extract_target_lc(cmdline_args.lc_fnames,
                                                       cmdline_args.target_id)

        _, target_result = correct_target_lc(
            target_lc_fname,
            cmdline_args,
            correct
        )
    else:
        lc_fnames = cmdline_args.lc_fnames

    if lc_fnames:
        result = apply_parallel_correction(
            lc_fnames,
            correct,
            cmdline_args.num_parallel_processes
        )
        if cmdline_args.target_id is not None:
            result = numpy.concatenate((result, target_result))
    else:
        result = target_result

    if cmdline_args.target_id is not None:
        lc_fnames.append(target_lc_fname)

    add_catalogue_info(lc_fnames,
                       cmdline_args.catalogue_fname,
                       cmdline_args.magnitude_column,
                       result)

    if not exists(dirname(output_statistics_fname)):
        makedirs(dirname(output_statistics_fname))
    save_correction_statistics(result, output_statistics_fname)

    logging.info('Generated statistics file: %s.',
                 repr(output_statistics_fname))

def recalculate_detrending_performance(lc_fnames,
                                       catalogue_fname,
                                       magnitude_column,
                                       output_statistics_fname,
                                       **recalc_arguments):
    """
    Re-create a statistics file after de-trending directly from LCs.

    Args:
        lc_fnames:    The filenames of the de-trended lightcurves to rederive
            the statistics for.

        catalogue_fname:     The filename of the catalogue to add information to
            the statistics.

        magnitude_column:     The column from the catalogue to use as brightness
            indicator in the statistics file.

        output_statistics_fname:    The filename to save the statistics under.

        recalc_arguments:    Passed directly to
            recalculate_correction_statistics()
    """

    statistics = recalculate_correction_statistics(lc_fnames,
                                                   **recalc_arguments)
    add_catalogue_info(lc_fnames, catalogue_fname, magnitude_column, statistics)

    if not exists(dirname(output_statistics_fname)):
        makedirs(dirname(output_statistics_fname))
    save_correction_statistics(statistics, output_statistics_fname)
