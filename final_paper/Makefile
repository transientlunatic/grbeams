default: grb_beams_paper

grb_beams_paper:
	rm -f grb_beams_paper.aux grb_beams_paper.bbl
	latexmk --silent --pdf grb_beams_paper

debug:
	rm -f grb_beams_paper.aux grb_beams_paper.bbl
	latexmk --pdf grb_beams_paper
	
clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.fls *.fdb_latexmk *.lot *.lof *.toc *.bbl *.blg *.out grb_beams_paper.pdf *Notes.bib
	
