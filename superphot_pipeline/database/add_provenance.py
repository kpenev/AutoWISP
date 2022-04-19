#!/usr/bin/env python3

"""Script for manually adding provenance information for the pipeline."""

import readline
from argparse import ArgumentParser, FileType

from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase

from superphot_pipeline.database.data_model.provenance import\
    camera,\
    camera_access,\
    camera_type,\
    mount,\
    mount_type,\
    mount_access,\
    telescope,\
    telescope_type,\
    telescope_access,\
    observatory,\
    observer

import regular_expressions as rex

def parse_command_line():
    """Parse the command line into the attributes of an object."""

    parser = ArgumentParser(
        description='Add or update provenance information. Any command line arguments '
        'specified will be held fixed and the user will be prompted for the '
        'remaining information.'
    )
    parser.add_argument(
        '--camera', '--first', '-f',
        help='The camera specifications to add/update.'
    )

    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='If passed, instead of creating new entries, a check is '
        'performed whether the grade just intered matches what is already on '
        'record.'
    )
    parser.add_argument(
        '--change-existing',
        action='store_true',
        default=False,
        help='If passed, any grades entered must already have entries in the '
        'database, and their score/weight is updated.'
    )

    return parser.parse_args()

#The interface of the class is required by readline module.
#pylint: disable=too-few-public-methods

class SimpleCompleter():
    """Handle readline autocomplete suggestions."""

    def __init__(self, options):
        """
        Set-up autocompletion among a given list of words.
        Args:
            options:    The list of words to autocomplete on.
        Returns:
            None
        """
        self.options = sorted(options)
        self.matches = None

    def __call__(self, text, state):
        """Return the state-th word which starts with the given text."""

        response = None
        text = text.strip()
        if state == 0:
            if text:
                self.matches = [s
                                for s in self.options
                                if s and s.startswith(text)]
            else:
                self.matches = self.options[:]

        try:
            response = self.matches[state]
        except IndexError:
            response = None
        return response
#pylint: enable=too-few-public-methods

class InputProvenanceEntry:
    """Handle input of provenance infromation by user."""

    def __init__(self, db_session):
        """
        Prepare for inputting provenance.
        Args:
            db_session:    A session to query the database.
        Returns:
            None
        """

    @staticmethod
    def get_camera_type_list(db_session, s):
        """
        Get list of camera types in database.
        Args:
            db_session:    An open database session for queries.

        Returns:
            ids:    A list of the IDs in the database of all selected cameras.

        """

        query = db_session.query(CameraType.id,
                                 CameraType.make,
                                 CameraType.model,
                                 CameraType.version,
                                 CameraType.sensor_type)

        return  ##how the hell do I return all these identifiers

    @staticmethod
    def get_camera_list(db_session, serial_number=None, x_resolution=None, y_resolution=None):
        """
        Get list of cameras in database.
        Args:
            db_session:    An open database session for queries.
            serial_number:    The serial number of the camera. Leave None if not known.
            x_resolution:    The x resolution of the camera. Leave None if not known
            y_resolution:    The y resolution of the camera. Leave None if not known
        Returns:
            ids:    A list of the IDs in the database of all selected cameras.

        """

        query = db_session.query(Camera)
        if serial_number is not None:
            query = query.filter(Camera.serial_number == serial_number)
        if x_resolution is not None:
            query = query.filter(Camera.x_resolution == x_resolution)
        if y_resolution is not None:
            query = query.filter(Camera.y_resolution == y_resolution)
        return query.all()
#SEQID is the observing session (this goes into notes of the observingsession)
#target is the field 'TESS_SEC07_CAM01' (this is the name),#RA & DEC is the RA_MNT and DEC_Mnt

#     result = db_session.query(
#             SerializedInterpolator
#         ).join(
#             track_counts,
#             SerializedInterpolator.id == track_counts.c.id
#         ).filter(
#             *match_config
#         ).one_or_none()
#         if result is None:
#             return result
#First use query join the object then filter it, then db.add the objects or object lists
#PRACTICE JOIN STATEMENTS ON SQL, find a tutorial

#Inner join selects records that have matching values in both tables
#Left join (or left outer join) returns all records from the left table and the matching records from the right table, the result is 0 records from the right side if there is no match
#Right join (or right outer join) returns all records from the right table and the matching records from the left table, the result is 0 records from the left side if there is no match
#Full join (or full outer join) returns all records when there is a match in left OR right table records
#Self join is a regular join but the table is joined with itself


#     @staticmethod
#     def get_assignment_type_ids(db_session):
#         """
#         Return a list of the assignment types registered in the database.
#         Args:
#             db_session:    An open database session for queries.
#         Returns;
#             assignment_type_ids:    A dictionary indexed by the assignment type
#                 description of the IDs of the assignment types available.
#         """
#
#         db_result = db_session.query(AssignmentType.description,
#                                      AssignmentType.id).all()
#         assignment_type_ids = dict(db_result)
#         db_result = db_session.query(BonusAssignment.description,
#                                      BonusAssignment.id).all()
#         for bonus_name, bonus_id in db_result:
#             assignment_type_ids['+' + bonus_name] = bonus_id
#         return assignment_type_ids
#
#     def get_student_name(self, db_session, which, other_name=None):
#         """
#         Prompt and help the user input one the names of a student.
#         Args:
#             which:    Which name to get. Should be either 'first' or 'last'.
#             other_name:    The name of the student other than the one being
#                 input, if known. Leave None if unknown.
#         Returns:
#             first_name:    The first name entered by the user.
#         """
#
#         if other_name is None:
#             readline.set_completer(self._name_completer[which])
#         else:
#             name_index = 1 if which == 'first' else 2
#             known_name = 'last' if which == 'first' else 'first'
#             name_list = self.get_student_list(
#                 db_session=db_session,
#                 **{known_name + '_name': other_name}
#             )[name_index]
#
#             readline.set_completer(
#                 SimpleCompleter(name_list)
#             )
#         line = input(which + ' name> ')
#         return line.strip()
#
#     def get_assignment(self,
#                        db_session,
#                        assignment_type=None,
#                        assignment_id=None,
#                        assignment_date=None):
#         """
#         Prompt and help the user to select an assignment.
#         Args:
#             db_session:    An open database session for queries.
#         Returns:
#             assignment_type_id:    The type id of the assignment selected.
#             assignment_id:    The id of the particular assignment of that type
#                 selected.
#         """
#
#         if assignment_type is None:
#             readline.set_completer(
#                 SimpleCompleter(self._assignment_type_ids.keys())
#             )
#             assignment_type = input('assignment type> ')
#
#         type_id = self._assignment_type_ids[assignment_type]
#         if assignment_type[0] == '+':
#             return 'bonus', type_id
#
#         if assignment_id is not None:
#             return type_id, assignment_id
#
#         db_result = db_session.query(
#             Assignment.due_date,
#             Assignment.id
#         ).filter_by(
#             type_id=type_id
#         )
#         date_to_id = {record[0].strftime('%m%d'): record[1]
#                       for record in db_result}
#
#         if assignment_date is None:
#             readline.set_completer(
#                 SimpleCompleter(date_to_id.keys())
#             )
#             assignment_date = input('assignment date> ')
#
#         return type_id, date_to_id[assignment_date]
#
#     def __call__(self,
#                  db_session,
#                  ignore_unmatched_students=False,
#                  **predefined):
#         """
#         Prompt the user as necessary to define new grade entry for the records.
#         Args:
#             db_session:    An open database session for queries.
#             **predefined:    A list of pre-defined entries for the grade,
#                 anything missing is queried from the user. Could contain any
#                 combination of the following:
#                 - student_first: First name of the student whose grade is being
#                   updated
#                 - student_last: Last name of student (see above)
#                 - utdid: The 10-digit UTD ID of the student. If supplied first
#                   and last name of the student are ignored.
#                 - netid: The 3-letter + 6 digit Net ID of the student. If
#                   supplied, previous ways of selecting a student are ignored.
#                 - student_id: The ID within the database of the student whose
#                   record is being updated. If this is supplied, previous ways of
#                   selecting a student are ignored.
#                 - assignment_type: The type of assignment being added.
#                 - assignment_id: Which particular assignment from the given type
#                   it is.
#                 - score: The score to assign for the assignment.
#                 - weight:    The student & assignment specific weight to assign.
#         Returns:
#             grade:    The new record to create. A dictionary with keys:
#                 - student_id: The id of the student whose grade is being
#                   updated.
#                 - assignment_type: The type of assignment the grade refers to.
#                 - assignment_id: The ID of the assignment the grade refers to.
#                 - score: The score the student received on the assignment.
#                 - weight: The student-specific weight for this grade.
#         """
#
#         grade = dict()
#
#         grade['student_id'] = predefined.get('student_id', None)
#         if grade['student_id'] is None:
#             select_student = dict()
#             for id_method in ['utdid', 'netid', 'email']:
#                 if id_method in predefined:
#                     select_student[id_method] = predefined[id_method]
#                     break
#             if not select_student:
#                 for name in ['first', 'last']:
#                     select_student[name + '_name'] = (
#                         predefined.get('student_' + name, None)
#                         or
#                         self.get_student_name(db_session=db_session, which=name)
#                     )
#             print(repr(select_student))
#             grade['student_id'] = db_session.query(
#                 Student.id
#             ).filter_by(
#                 **select_student
#             ).one_or_none()
#             if ignore_unmatched_students and grade['student_id'] is None:
#                 return
#             grade['student_id'] = grade['student_id'][0]
#
#         grade['assignment_type'], grade['assignment_id'] = (
#             self.get_assignment(
#                 db_session,
#                 assignment_type=predefined.get('assignment_type', None),
#                 assignment_date=predefined.get('assignment_date', None),
#                 assignment_id=predefined.get('assignment_id', None)
#             )
#         )
#
#         grade['score'] = predefined.get('score', None)
#         if grade['score'] is None:
#             grade['score'] = input('score' + '> ')
#
#         if grade['assignment_type'] != 'bonus':
#             grade['weight'] = predefined.get('weight', None)
#             if grade['weight'] is None:
#                 grade['weight'] = input('weight' + '> ')
#
#         return grade
#
# def update_loop(check=False,
#                 change_existing=False,
#                 student_list=None,
#                 **kwargs):
#     """
#     Keep updating student records until the user indicates a stop.
#     Args:
#         check(bool):    If True, user entered grades are checked against
#             existing records failing an assert if they don't match, instead of
#             inserting new entries.
#         change_existing(bool):    If False attempting to specify a grade already
#             in the database results in exception. If True, grades are
#             overwritten.
#         student_list(iterable or None):    If specified should contain a list of
#             student UTD IDs or Net IDs for which to add grades.
#         **kwargs:    Passed directly to InputGradeEntry.__call__, as in:
#         >>> with db_session_scope() as db_session:
#         >>>     InputGradeEntry()(db_session, **kwargs)
#     Returns:
#         None
#     """
#
#     with db_session_scope() as db_session:
#         input_grade = InputGradeEntry(db_session)
#
#     while True:
#         with db_session_scope() as db_session:
#             for student_id in student_list or [None]:
#                 select_student = dict()
#                 if student_id is not None:
#                     print('Student ID: ' + repr(student_id))
#                     for id_format in ['netid', 'email', 'utdid']:
#                         if getattr(rex, id_format).match(student_id.strip()):
#                             print('Appears to be ' + id_format)
#                             select_student = {id_format: student_id.strip()}
#                             break
#                     assert select_student
#                 grade = input_grade(db_session, **kwargs, **select_student)
#                 if not grade:
#                     if student_list is None:
#                         break
#                     else:
#                         assert kwargs.get('ignore_unmatched_students', False)
#                         continue
#                 if grade['assignment_type'] == 'bonus':
#                     if check:
#                         assert(
#                             db_session.query(Bonus).filter_by(**grade).count()
#                             ==
#                             1
#                         )
#                         print(80*'-' + '\nMatches\n' + 80*'-')
#                     else:
#                         db_session.add(Bonus(student=grade['student_id'],
#                                              assignment=grade['assignment_id'],
#                                              score=grade['score']))
#                 else:
#                     if check:
#                         assert(
#                             db_session.query(Grade).filter_by(**grade).count()
#                             ==
#                             1
#                         )
#                         print(80*'-' + '\nMatches\n' + 80*'-')
#                     elif change_existing:
#                         db_session.query(Grade).filter_by(
#                             student_id=grade['student_id'],
#                             assignment_type=grade['assignment_type'],
#                             assignment_id=grade['assignment_id']
#                         ).update(
#                             dict(score=grade['score'],
#                                  weight=gr
#                         ade['weight'])
#                         )
#                     else:
#                         db_session.add(Grade(**grade))
#         if student_list:
#             break