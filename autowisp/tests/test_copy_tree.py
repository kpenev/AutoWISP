"""Tests of copying a project's files into another home."""

import os
import shutil
import stat
import tempfile
import unittest
from pathlib import PurePath
from unittest import mock

from autowisp.exceptions import ResourceError
from autowisp.project_experiments import ProjectExperiments


class TestCopyTree(unittest.TestCase):
    """Copying a project's files: what is left out, linked and replaced."""

    # The files of a project's database, which no copy of its files takes.
    _database_files = [
        "autowisp.db",
        "autowisp.db-journal",
        "autowisp.db.pre-0020_pipeline_run_identity",
        "autowisp_db.url",
    ]

    def _make_home(self):
        """Return a new, empty directory, removed after the test."""

        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home)
        return home

    @staticmethod
    def _write_files(home, contents):
        """Write files under the home, ``{relative POSIX path: content}``."""

        for fname, content in contents.items():
            path = os.path.join(home, *fname.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="ascii") as file:
                file.write(content)

    @staticmethod
    def _read_files(home):
        """Return the contents of the files under the home, by POSIX path."""

        contents = {}
        for directory, _, fnames in os.walk(home):
            for fname in fnames:
                path = os.path.join(directory, fname)
                with open(path, encoding="ascii") as file:
                    contents[
                        PurePath(os.path.relpath(path, home)).as_posix()
                    ] = file.read()
        return contents

    def test_copy_tree(self):
        """Each kind of file, copied into a home already holding some.

        As a merge copies an experiment into its original.
        """

        source_home = self._make_home()
        target_home = self._make_home()
        # Each file of the source: whether it is linked, and whether copied.
        source_files = {
            **{fname: (False, False) for fname in self._database_files},
            "experiments/exp1/autowisp.db": (False, False),
            # The database's name, but not at the top of the home.
            "MASTERS/autowisp.db": (False, True),
            "DR/frame.h5": (False, True),
            "LC/star.h5": (False, True),
            "RAW/frame1.fits": (True, True),
            "RAW/frame2.fits": (True, True),
        }
        self._write_files(
            source_home, {fname: f"{fname} (source)" for fname in source_files}
        )
        os.mkdir(os.path.join(source_home, "LOGS"))
        target_files = {
            # Replaced: steps write to it.
            "LC/star.h5": "LC/star.h5 (target)",
            # Kept: nothing writes to it, so it is a copy of the source's.
            "RAW/frame2.fits": "RAW/frame2.fits (target)",
        }
        self._write_files(target_home, target_files)
        if os.name == "posix":
            # Windows needs privileges to create one.
            outside = self._make_home()
            self._write_files(outside, {"file.txt": "linked/file.txt (source)"})
            os.symlink(outside, os.path.join(source_home, "linked"))
            source_files["linked/file.txt"] = (False, True)

        expected = {
            fname: f"{fname} (source)"
            for fname, (_, copied) in source_files.items()
            if copied
        }
        expected.update({"RAW/frame2.fits": target_files["RAW/frame2.fits"]})
        needed = sum(
            len(expected[fname])
            for fname, (linked, copied) in source_files.items()
            if copied and not linked
        )
        linked_files = {
            PurePath(fname)
            for fname, (linked, _) in source_files.items()
            if linked
        }

        # pylint: disable=protected-access
        with mock.patch.object(
            shutil, "disk_usage", return_value=mock.Mock(free=needed - 1)
        ):
            with self.assertRaises(ResourceError):
                ProjectExperiments._copy_tree(
                    source_home, target_home, linked_files
                )
        refused_contents = self._read_files(target_home)
        with mock.patch.object(
            shutil, "disk_usage", return_value=mock.Mock(free=needed)
        ):
            ProjectExperiments._copy_tree(
                source_home, target_home, linked_files
            )
        # pylint: enable=protected-access

        self.assertEqual(refused_contents, target_files)
        self.assertEqual(self._read_files(target_home), expected)
        self.assertEqual(
            {
                fname
                for fname in expected
                if os.path.samefile(
                    os.path.join(source_home, fname),
                    os.path.join(target_home, fname),
                )
            },
            {"RAW/frame1.fits"},
        )
        self.assertTrue(os.path.isdir(os.path.join(target_home, "LOGS")))
        self.assertFalse(os.path.islink(os.path.join(target_home, "linked")))

    def test_clear_tree(self):
        """Clearing a home leaves only its database's files.

        Read-only files are removed too, which Windows refuses to do as they
        are.
        """

        home = self._make_home()
        read_only = ["notes.txt", "RAW/frame.fits"]
        self._write_files(
            home,
            {
                fname: ""
                for fname in self._database_files + read_only + ["DR/frame.h5"]
            },
        )
        for fname in read_only:
            os.chmod(os.path.join(home, *fname.split("/")), stat.S_IREAD)

        # pylint: disable-next=protected-access
        ProjectExperiments._clear_tree(home)

        self.assertEqual(sorted(os.listdir(home)), sorted(self._database_files))


if __name__ == "__main__":
    unittest.main()
