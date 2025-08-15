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

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	python3 source/document_options.py
	$(SPHINXAPIDOC) "$(MODULEDIR)" $(APIDOCOPTS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf ../docs
	mv build/html ../docs
