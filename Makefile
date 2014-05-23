default: grb_beams

grb_beams:
	rm -f grb_beams.aux grb_beams.bbl
	latex grb_beams
	bibtex grb_beams
	latex grb_beams
	latex grb_beams
	dvipdf grb_beams
	
	
clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.lot *.lof *.toc *.bbl *.blg *.out *pdf
	
