# To install all the required packages
requirements:
	python3 -m pip install -r requirements.txt

spectra:
	python3 ./targ_spec.py > ./results/output-target.txt

jupiter:
	python3 ./jup_rad_vel.py > ./results/output-jup.txt

all:
	make spectra
	make jupiter
