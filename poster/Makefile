default: grb_beams

grb_beams:
	rm -f *.aux *.bbl
	pdflatex grbeams_poster
	bibtex grbeams_poster
	pdflatex grbeams_poster
	pdflatex grbeams_poster
	
	
clean:
	@echo "Cleaning directory of backups and logs"
	rm -f *~ *.log *.aux *.dvi *.lot *.lof *.toc *.bbl *.blg *.out *pdf
	
