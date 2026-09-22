# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXPROJ    = AutoWISP
SOURCEDIR     = source
MODULEDIR     = ../autowisp
BUILDDIR      = build
SPHINX_APIDOC_OPTIONS=members,show-inheritance
APIDOCOPTS    = --force -M --separate -o "$(SOURCEDIR)/implementation"
# apidoc reads the source tree while autodoc imports the installed
# package, so anything deliberately left out of the install gets a page it
# cannot fill. These four are excluded from the install on purpose -- see
# CLAUDE.md -- and are excluded here to match.
APIDOCSKIP    = "$(MODULEDIR)/fake_image" \
		"$(MODULEDIR)/magnitude_fitting/tests" \
		"$(MODULEDIR)/tests/generate_catalog_test_data.py" \
		"$(MODULEDIR)/tests/update_hdf5_contents.py"
# data_model/__init__.py re-exports every table into __all__, so the
# package page documents all of them. Giving each module a page of its own
# as well documents every class twice, which leaves each one with two
# targets and every reference to one ambiguous.
APIDOCSKIP   += "$(MODULEDIR)/database/data_model/[!_]*.py"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	python3 source/document_options.py
# sphinx-apidoc overwrites the pages it generates but never deletes the
# ones whose module is gone, so a deleted module stays published until
# the directory is cleared out first.
	rm -rf "$(SOURCEDIR)/implementation"
	$(SPHINXAPIDOC) "$(MODULEDIR)" $(APIDOCOPTS) $(APIDOCSKIP)
# An incremental build only writes the pages it re-reads, but ../docs is
# replaced with the whole of build/html below, so anything skipped goes
# missing from the published site. Build from scratch every time.
	rm -rf "$(BUILDDIR)"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf ../docs
	mv build/html ../docs
