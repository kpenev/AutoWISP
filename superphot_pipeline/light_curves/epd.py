#!/usr/bin/env python3

"""Define object for performing EPD correction on lightcurves."""

from multiprocessing import Pool
from os.path import expanduser
from os.path import join as join_paths
from os import symlink
from subprocess import Popen, PIPE
from traceback import print_exc
import re

from scipy import array, nan, isnan, median, histogram, mean, sin, cos, pi,\
                  sqrt, delete, zeros, dot
from scipy.optimize import leastsq


from HATpipepy.Common.LCUtil import light_curves_to_db, pending_stars, \
                                    rawlc_config
from HATpipepy.Common.FitUtil import variable_names, coefficient_names,\
                                     linear_predictors
from HATpipepy.Common.HATUtil import hat_id_fmt, parse_hat_id, Structure
from HATpipepy.Common.DBUtil import get_db, raw_db_table, light_curve_table
from HATpipepy.Common.BinLC import read_lc, modify_lc
from HATpipepy.Common import Error

git_id = '$Id$'

def mode(values, bins):
    """
    Estimate mode(values) as the middle of the most poulated bin in a histogram.

    Args:
        values:    The array of values to estimate the mode of.

        bins:    The number of bins into which to split the array values.

    Returns:
        int:
            The number of values in the most populated bin.

        float:
            The middle of the most populated bin in a histogram of the input
            values. See scipy.histogram for details no how that is built.
    """

    hist = histogram(values, bins=bins)
    return max(
        (hist[0][i], 0.5 * (hist[1][i] + hist[1][i + 1]))
        for i in range(len(hist[0]))
    )

def add_epd_to_lc(lcfname, lc, rawlccfg) :
    """ Writes the given light curve to a file with the given name.

    Input
    -----
    lcfname : str
        filename to read the light curve from
    lc : dict
        dictionary of the stored columns
    rawlccfg: config
        config object which contains configuration options read from the
        database
    """

    epd_col_ind=filter(
        lambda i: lc.hdr['COLUMNS'][i][0].startswith('MEPD'),
        range(len(lc.hdr['COLUMNS'])))
    modify_lc(lcfname, rawlccfg.hdrreclen,
              header=dict(EPD=('True', 'Whether EPD was performed on this '
                               'light curve.')),
              values=[(i, lc.data[i]) for i in epd_col_ind])

def exclude_frames(stars, image_type, field, cmpos, column_name, db,
                   logger) :
    """ Marks in the database frames which are outliers in the given column
    for an abnormal number of stars. """

    known_keys=set('IMAGETYP', 'OBJECT', 'CMPOS')

    def outliers(lc, outlier_limit) :
        """ Returns a list of the frames for which the exclusion column is an
        outlier for the given light curve file. """

        def outlier_ind(exclusion_column) :
            """ Returns a list of indices in the given column which
            correspond to outliers. """

            med=median(exclusion_column)
            res2=pow(exclusion_column-med, 2)
            max_res2=outlier_limit*mean(res2)
            return filter(lambda i: res2>max_res2,
                          range(len(exclusion_column)))

        exclusion_column=array(lc[column_name])
        exclusion_orig=exclusion_column.copy()
        dropped=True
        while dropped :
            dropped=outlier_ind(exclusion_column)
            exclusion_column=delete(exclusion_column, dropped)
        return (map(lambda i : lc['KEY'][i], outlier_ind(exclusion_orig)),
                len(lc['KEY']))

    def get_config() :
        """ Reads the relevant configuration from the database. """

        config=Structure()
        lcdir, lcdir_keys=db.get_name_template(db.station, 'lightcurves',
                                               'dirname')
        assert(set(lcdir_keys)<=known_keys)
        config.lcdir=lcdir.substitute(IMAGETYP=image_type, OBJECT=field,
                                      CMPOS=cmpos)
        (config.excl_limit, config.min_lc_length, config.lclength_base,
         config.lclength_limit, config.lclength_bins, config.outlier_limit,
         config.version)=db(
             'SELECT `excl_limit`, `ignore_lc_length`, `lclength_base`, '
             '`lclength_limit`, `lclength_bins`, `outlier_limit`, `version` '
             'FROM `OutlierFramesCfg` ORDER BY `version` DESC LIMIT 1')
        if config.outlier_base!='mode' :
            config.outlier_base=eval(config.outlier_base)

    config=get_config()
    lccolumns=rawlc_config(db, project_id, sphotref_id).columns
    stat=dict()
    lightcurve_lengths=[]
    for s in stars :
        lc=read_lc(join(config.lcdir, s+db.extensions['lightcurve']['raw']),
                   lccolumns)
        if len(lc)<config.min_lc_length : continue
        lightcurve_lengths.append(len(lc))
        for o in outliers(lc, config.excl_limit) :
            if o in stat : stat[o]+=1
            else : stat[o]=1
    if config.lclength_base=='mode' :
        min_lclength=mode(lightcurve_lengths, config.lclength_bins)
    else : min_lclength=config.lclength_base(lightcurve_lengths)
    min_lclength*=config.lclength_limit
    crit_outliers=reduce(lambda c,v: c+1 if v>min_lclength else c,
                         lightcurve_lengths)*config.outlier_limit
    outliers=map(lambda fo: fo[0],
                 filter(lambda fo: fo[1]>crit_outliers, stat.iteritems()))
    table=raw_db_table(image_type)
    db("UPDATE `"+table+"` SET `outlier`=CONCAT_WS(',', `outlier`, %s) WHERE"
       " `station_id`=%s AND `fnum`=%s AND `cmpos`=%s", outliers)

def get_work(db) :
    """ Returns a list of (project_id, sphotref_id, imagetype, object, cmpos,
    check_outliers) identifying sets of light curves for which EPD has not yet
    been done and if they should be checked for outliers or not. """

    return db('SELECT `LCStatus`.`project_id`, `LCStatus`.`sphotref_id`, '
              '`imagetype`, `object`, `cmpos`, `outlier_cfg`  FROM '
              '`LCStatus`, `SinglePhotRef` WHERE `rawlc_cfg` IS NOT NULL '
              'AND `epd_cfg` IS NULL '
              'AND `LCStatus`.`project_id`=`SinglePhotRef`.`project_id` '
              'AND `LCStatus`.`sphotref_id`=`SinglePhotRef`.`sphotref_id` '
              'ORDER BY `project_id`, `sphotref_id`, `imagetype`, `object`, '
              '`cmpos`',
              no_simplify=True)

class EPDFit :
    """ A class that performs the EPD fit, stores the coefficients in the
    database and inserts the fitted magnitudes in the light curve files. """

    def __get_config(self) :
        """ Creates self.config containing the configuration parameters of
        how to do the EPD. """

        self.config=Structure()
        (self.config.func_type, self.__func_param, self.config.min_pts_mult,
         self.config.outlier_limit, fit_res_avg,
         self.config.fit_res_limit, self.config.max_rej_iter,
         self.max_chunk_size, self.config.version)=self.__db(
             'SELECT `func_type`, `func_param`, `min_pts_mult`, '
             '`outlier_limit`, `fit_res_avg`, `fit_res_limit`, '
             '`max_rej_iter`, `max_chunk_size`, `version` FROM `EPDCfg` '
             'ORDER BY `version` DESC LIMIT 1')
        self.config.fit_res_avg=eval(fit_res_avg)
        self.__varnames=variable_names(self.__func_param)
        self.__coefficients=coefficient_names(self.__func_param,
                                              self.__varnames)
        self.config.lcdir, lcdir_keys=db.get_name_template(db.station,
                                                             'lightcurves',
                                                             'dirname')
        known_keys=set(['IMAGETYP', 'OBJECT', 'CMPOS'])
        assert(set(lcdir_keys)<=known_keys)

    def __get_excluded(self) :
        """ Sets self.__excluded frames to a list (by aperture) of
        dictionaries (by station id) of lists of excluded frame numbers for
        the current observed field and camera. """

        self.__excluded_frames=[]
        tbl=raw_db_table(self.__image_type)
        for ap in range(self.__num_apertures) :
            records=db('SELECT `station_id`, `fnum` FROM `'+tbl+'` WHERE '
                       '`cmpos`=%s AND FIND_IN_SET(%s, `outlier`) ORDER BY '
                       '`station_id`, `fnum`', (self.__cmpos, 'mfit%d'%ap),
                       no_simplify=True)
            if records is None : self.__excluded_frames.append(dict())
            else :
                excl=dict()
                current_stid=-1
                for stid, fnum in records :
                    if stid!=current_stid :
                        current_stid=stid
                        excl[current_stid]=set([fnum])
                    else : excl[stid].add(fnum)
                self.__excluded_frames.append(excl)

    def __independent_variables(self) :
        """ Returns the independent variables that participate in the EPD fit
        function as a dictionary. """

        indep_var=dict()
        for var in self.__varnames :
            varc=var+'c'
            if var in self.__lccolumns :
                indep_var[var]=array(
                    self.__lc.data[self.__lccolumns.index(var)])
            elif varc in self.__lccolumns :
                indep_var[var]=array(
                    self.__lc.data[self.__lccolumns.index(var)])
            elif var in ['FX', 'FY'] :
                indep_var[var]=array(
                    self.__lc.data[self.__lccolumns.index(var[1])])%1
            else :
                match=self.__sin_cos_re.match(var)
                if match :
                    func, freq, xy=match.groups()
                    freq=2.0*pi*eval(freq)
                    if func=='S' :
                        indep_var[var]=sin(freq*array(
                            self.__lc.data[self.__lccolumns.index(xy)]))
                    else : indep_var[var]=cos(freq*array(
                        self.__lc.data[self.__lccolumns.index(xy)]))
                else :
                    raise Error.Database('Unrecognized variable %s in EPD '
                                         'function'%repr(var))
        return indep_var

    def __update_db(self, fit_coef, station, phot_flag, aperture, fit_res,
                    input_points, non_rej_points) :
        """ Stores the given coefficients to the database. """

        self.__db('REPLACE INTO `'+self.config.dest_table+'` ('
                  '`prim_field`, `source`, `image_type`, `object`, `cmpos`, '
                  '`station_id`, `phot_flag`, `aperture`, `epd_cfg`, '
                  '`input_pts`, `non_rej_pts`, `rms_residuals`, `'
                  +'`, `'.join(self.__coefficients)+'`) VALUES (%s'+
                  ', %s'*(len(self.__coefficients)+11)+')',
                  (self.__fit_object[0], self.__fit_object[1],
                   self.__image_type, self.__obs_field, self.__cmpos,
                   station, phot_flag, aperture, self.config.version,
                   input_points, non_rej_points, fit_res)+tuple(fit_coef))

    def __get_columns(self, aperture) :
        """ Returns the indices of the fitted magnitude column, the
        photometry quality flag column, the station id column and the frame
        number column. """

        return (self.__lccolumns.index('MFIT%d'%aperture),
                self.__lccolumns.index('PHOTFLAG%d'%aperture),
                self.__lccolumns.index('STID'),
                self.__lccolumns.index('FNUM'))

    def __fit(self, station, phot_flag, aperture) :
        """ Returns the EPD coefficients for the subset of the magfit
        magnitudes of the current light curve which correspond to the given
        station, have the given photometry flag and contain no nan values in
        either the magnitudes or the independent variables. """

        MFIT_COL, FLAG_COL, STID_COL, FNUM_COL=self.__get_columns(aperture)
        if station in self.__excluded_frames[aperture] :
            excluded_frames=self.__excluded_frames[aperture][station]
        else : excluded_frames=set()
        magnitudes=array(self.__lc.data[MFIT_COL])
        min_mag, max_mag=nan, nan

        def outlier_limits() :
            """ Returns the range in which the magfit magnitude should lie in
            order for the corresponding record to participate in the EPD fit.
            """

            excl_ind=filter(
                lambda i: (self.__lc.data[FNUM_COL][i] in excluded_frames or
                           isnan(magnitudes[i])), range(len(magnitudes)))
            mag_copy=delete(magnitudes, excl_ind)
            dropped=True
            while dropped :
                med=median(mag_copy)
                res2=(mag_copy-med)**2
                lim2=self.config.outlier_limit**2*mean(res2)
                dropped=filter(lambda i: res2[i]>lim2, range(len(res2)))
                mag_copy=delete(mag_copy, dropped)
            res=sqrt(mean(res2))
            return med-res, med+res

        def accept_ind(i) :
            """ Returns True if the given index should be included in the EPD
            fit. """

            mag=self.__lc.data[MFIT_COL][i]
            if (self.__lc.data[STID_COL][i]==station and
                self.__lc.data[FLAG_COL][i]==phot_flag and
                (not isnan(mag)) and mag>min_mag and mag<max_mag and
                self.__lc.data[FNUM_COL][i] not in excluded_frames) :
                for v in self.__indep_var.values() :
                    if isnan(v[i]) : return False
                return True
            else : return False

        def rejected_indices(fit_res) :
            """ Returns a list of indices that have fit residuals that fall
            outside the configured limits. """

            fit_diff2=fit_res**2
            res2=self.config.fit_res_avg(fit_diff2)
            max_diff2=(self.config.fit_res_limit**2*res2)
            return filter(lambda i: fit_diff2[i]>max_diff2,
                          range(len(fit_diff2))), res2

        min_mag, max_mag=outlier_limits()
        fit_ind=filter(accept_ind, range(len(magnitudes)))
        fit_var=dict()
        for k, v in self.__indep_var.iteritems() : fit_var[k]=v.take(fit_ind)
        predictors=linear_predictors(self.__func_param, fit_var)
        magnitudes=magnitudes.take(fit_ind)
        num_free_coef=len(self.__coefficients)
        min_non_rej_pts=self.config.min_pts_mult*num_free_coef
        input_points=len(magnitudes)
        error_func=lambda coef: dot(coef, predictors)-magnitudes
        deriv_func=lambda coef: predictors
        initial_guess=zeros(num_free_coef-1)
        for rej_iter in range(self.config.max_rej_iter) :
            if len(magnitudes)<min_non_rej_pts :
                coefficients=None
                fit_res2=None
                magnitudes=[]
                break
            coefficients, covariance, info_dict, msg, status=leastsq(
                error_func, initial_guess, Dfun=deriv_func, col_deriv=1,
                full_output=1)
            if status not in [1, 2, 3, 4] :
                raise Error.Numeric("Linear least squares EPD fitting for "
                                    "aperture %d failed for "%ap_ind+
                                    hat_id_fmt%(self.__fit_object)+msg)
            rej_ind, fit_res2=rejected_indices(info_dict['fvec'])
            if not rej_ind : break
            if rej_iter==self.config.max_rej_iter-1 : break
            predictors=map(lambda p: delete(p, rej_ind), predictors)
            magnitudes=delete(magnitudes, rej_ind)
        if self.config.dest_table :
            db_coef=([None]*len(predictors) if coefficients is None else
                     coefficients)
            self.__update_db(db_coef, station, phot_flag, aperture,
                             sqrt(fit_res2), input_points, len(magnitudes))
        return coefficients

    def __solved(self, station, phot_flag, aperture) :
        """ Checks if the EPD fit for given aperture, station and photometry
        flag for the currently processed light curve is in the database, and
        if so returns the stored coefficients. """

        return False, False
        if not self.config.dest_table : return False, False
        coef=db('SELECT `applied`, `'+'`, `'.join(self.__coefficients)+'` '
                'FROM `'+self.config.dest_table+'` WHERE `prim_field`=%s '
                'AND `source`=%s AND `image_type`=%s AND `object`=%s AND '
                '`cmpos`=%s AND `station_id`=%s AND `phot_flag`=%s AND '
                '`aperture`=%s AND `epd_cfg`=%s',
                (self.__fit_object[0], self.__fit_object[1],
                 self.__image_type, self.__obs_field, self.__cmpos, station,
                 phot_flag, aperture, self.config.version))
        if coef is None : return False, False
        return (None, coef[0]) if coef[1] is None else (coef[1:], coef[0])

    def __apply_fit(self, coefficients, cat_magnitude, station, phot_flag,
                    aperture) :
        """ Applies the derived fit to the specified portion of the current
        light curve. """

        MFIT_COL, FLAG_COL, STID_COL, FNUM_COL=self.__get_columns(aperture)
        fit_ind=filter(
            lambda i: (self.__lc.data[STID_COL][i]==station and
                       self.__lc.data[FLAG_COL][i]==phot_flag),
            range(len(self.__lc.data[0])))
        fit_var=dict()
        for k, v in self.__indep_var.iteritems() : fit_var[k]=v.take(fit_ind)
        predictors=linear_predictors(self.__func_param, fit_var)
        magnitudes=array(self.__lc.data[MFIT_COL]).take(fit_ind)
        if coefficients is None :
            epd_mag=magnitudes-median(magnitudes)+cat_magnitude
        else :
            epd_mag=magnitudes-dot(coefficients, predictors)
            epd_mag=epd_mag-median(epd_mag)+cat_magnitude
        dest=self.__lc.data[self.__lccolumns.index('MEPD%d'%aperture)]
        for i, m in zip(fit_ind, epd_mag) : dest[i]=m

    def __init__(self, image_type, obs_field, cmpos, num_apertures, db,
                 track_lc_through_db=False, dest_table=None) :
        """ Creates the EPD fitting class. """

        self.__image_type=image_type
        self.__obs_field=obs_field
        self.__cmpos=cmpos
        self.__db=db
        self.__num_apertures=num_apertures
        self.__sin_cos_re=re.compile('([SC])(\d*)([XY])')
        self.__get_config()
        self.__lc=Structure()
        self.__track_lc_through_db=track_lc_through_db
        self.config.dest_table=dest_table
        rawlc_cfgver=db('SELECT `rawlc_cfg` FROM `LCStatus` WHERE '
                        '`project_id`=%s AND `sphotref_id`=%s',
                        (project_id, sphotref_id))[0]
        self.__raw_lc_cfg=rawlc_config(db, project_id, sphotref_id,
                                       rawlc_cfgver)
        self.__get_excluded()

    def __call__(self, fit_obj) :
        """ Derives EPD magnitudes for the given source (should be a tuple
        containing hat_id and 2MASS magnitude), updates the corresponding
        light and inserts the coefficients in the database. """

        try :
            self.__fit_object=parse_hat_id(fit_obj[0])
            lc_basename=self.config.lcdir.substitute(
                IMAGETYP=self.__image_type, OBJECT=self.__obs_field,
                CMPOS=self.__cmpos)+fit_obj[0]
            lc_ext=self.__db.extensions['lightcurve']
            self.__lc.hdr, self.__lc.data=read_lc(lc_basename+lc_ext['raw'],
                                                 self.__raw_lc_cfg.hdrreclen)
            assert len(self.__lc.hdr['APERTURES'])==self.__num_apertures
            self.__lccolumns=map(lambda c: c[0], self.__lc.hdr['COLUMNS'])
            station_list=sorted(set(
                self.__lc.data[self.__lccolumns.index('STID')]))
            self.__indep_var=self.__independent_variables()
            output_lc=False
            for station in station_list :
                for aperture in range(self.__num_apertures) :
                    for phot_flag in range(2) :
                        coefficients, applied=self.__solved(
                            station, phot_flag, aperture)
                        if coefficients==False :
                            coefficients=self.__fit(station, phot_flag,
                                                    aperture)
                        if not applied :
                            self.__apply_fit(coefficients, fit_obj[1],
                                             station, phot_flag, aperture)
                            output_lc=True
            if output_lc :
                add_epd_to_lc(lc_basename+lc_ext['raw'], self.__lc,
                              self.__raw_lc_cfg)
                symlink(lc_basename+lc_ext['raw'], lc_basename+lc_ext['epd'])
                if self.config.dest_table :
                    self.__db('UPDATE `'+self.config.dest_table+'` SET '
                              '`applied`=%s WHERE `prim_field`=%s AND '
                              '`source`=%s AND `image_type`=%s AND '
                              '`object`=%s AND `cmpos`=%s',
                              (1, self.__fit_object[0], self.__fit_object[1],
                               self.__image_type, self.__obs_field,
                               self.__cmpos))
            if self.__track_lc_through_db :
                light_curves_to_db(self.__image_type, self.__obs_field,
                                   self.__cmpos, self.__db,
                                   [self.__fit_object], 'epd')
        except Exception as ex :
            print_exc()
            raise

def get_num_apertures(image_type, field, cmpos, db) :
    """ Returns the number of apertures used in the photometry of the given
    set of frames. """

    return db('SELECT COUNT(*) FROM MasterPhotRef WHERE `object`=%s AND '
              '`cmpos`=%s AND `imagetype`=%s', (field, cmpos, image_type))[0]

def do_epd(db, options, logger, recover) :
    """ Derives the best EPD fit for the set currently existing light curves
    for which this has not already been done and fills in the corrected
    magnitudes. """

    work=get_work(db)
    if work is None : return
    for project_id, sphotref_id, image_type, field, cmpos, check_outliers in work:
        if options.restrict_to_fields :
            for allowed_field, allowed_cameras in options.restrict_to_fields:
                if field==allowed_field and (cmpos in allowed_cameras or
                                             allowed_cameras==()) :
                    break
            else:
                continue
        num_apertures=get_num_apertures(image_type, field, cmpos, db)
        stars=pending_stars(db, image_type, field, cmpos, 'raw', 'epd',
                            False)
        if check_outliers :
            logger.info('Checking for outlier %s frames in field %s, camera '
                        '%d'%(image_type, field, cmpos),
                               extra=log_extra)
            for ap in range(num_apertures) :
                exclude_frames(stars, image_type, field, cmpos, 'mfit%d'%ap,
                               db, logger)
            logger.info('Outlier %s frames in field %s, camera %d marked '
                        'successfully'%(image_type, field, cmpos),
                        extra=log_extra)
        logger.info('Starting EPD for the %s frames of field %s, camera %d'%
                    (image_type, field, cmpos), extra=log_extra)
        db_config_file=join_paths(expanduser(options.config_dir),
                                  options.localdb)
        epd_command=['epd_process.py', str(project_id), image_type, field,
                     str(cmpos), '--database', db_config_file,
                     '--processes', str(options.max_threads)]
        logger.debug("EPD command: '"+"' '".join(epd_command)+"'",
                     extra=log_extra);
        epd=Popen(epd_command, stdout=PIPE, stderr=PIPE)
        epd_out, epd_err=epd.communicate()
        if epd.returncode :
            raise Error.External(
                'The epd command for the %s frames of field %s, camera %d '
                'failed:'%(image_type, field, cmpos)+'\nstdout:\n'+epd_out+
                '\nstderr:\n'+epd_err)
        else :
            logger.info('Successfully completed EPD for the %s frames of '
                        'field %s, camera %d:'%(image_type, field, cmpos)+
                        '\nstdout:\n'+epd_out+'\nstderr:\n'+epd_err)

if __name__=='__main__' :
    project_id, image_type, field, cmpos, options=parse_command_line()
    db = get_db(options.database)
    # get sphotref_id from the SinglePhotRef table
    sphotref_id = db('SELECT `sphotref_id` FROM SinglePhotRef WHERE '
                     '`project_id`= %s AND `imagetype`=%s AND `object`=%s AND '
                     '`cmpos`=%s', (project_id, image_type, field, cmpos))[0]
    epd=EPDFit(image_type, field, cmpos,
               get_num_apertures(image_type, field, cmpos, db), db)
    stars=pending_stars(db, image_type, field, cmpos, 'raw', 'epd', True)
    if options.debug :
        epd_stars=set(options.debug)
        stars=filter(lambda s: s[0] in epd_stars, stars)
    if stars :
        chunks=int(min(epd.max_chunk_size,
                   max(10, len(stars)/(10*options.processes))))
        workers=Pool(processes=options.processes, maxtasksperchild=chunks)
        workers.map_async(epd, stars, chunksize=chunks).get()
        workers.close()
        workers.join()
    if not options.debug :
        db('UPDATE `LCStatus` SET `epd_cfg`=%s WHERE `project_id`=%s '
           'AND `sphotref_id`=%s', (epd.config.version, project_id,
                                              sphotref_id))
