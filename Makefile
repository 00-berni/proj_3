# To install all the required packages
requirements:
	python3 -m pip install -r requirements.txt

spectra_r:
	python3 ./targ_spec.py > ./results/output-target.txt

abs_cal_r:
	python3 ./abs_cal.py > ./results/output-abscal.txt

jupiter_r:
	python3 ./jup_period.py > ./results/output-jup.txt

spectra:
	python3 ./targ_spec.py

abs_cal:
	python3 ./abs_cal.py

jupiter:
	python3 ./jup_period.py

all:
	make spectra
	make jupiter
