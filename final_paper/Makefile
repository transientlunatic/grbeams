default: grb_beams_paper

grb_beams_paper:
	rm -f grb_beams_paper.aux grb_beams_paper.bbl
	latex grb_beams_paper
	bibtex grb_beams_paper
	latex grb_beams_paper
	latex grb_beams_paper
	dvipdf grb_beams_paper
	
	
clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.lot *.lof *.toc *.bbl *.blg *.out *pdf *Notes.bib
	
