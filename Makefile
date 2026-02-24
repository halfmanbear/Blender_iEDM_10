all:

zip:
	rm iEDM_10.zip
	zip -r iEDM_10.zip iEDM_10 -x '*/__pycache__/*'
