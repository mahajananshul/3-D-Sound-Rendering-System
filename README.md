# 3-D-Sound-Rendering-System
This project aims at the basics of sound localization to produce a 3-D sound by using some of the filtering concepts of Digital Signal Processing. It finds applications in entertainment systems such as two stereo speakers.
Download Project.zip for the full project code and data.
To view the project using the open files present here follow the guide below: 
(*Input files = helicopter.wav, large_pinna_final.mat, large_pinna_frontal.mat)
a) Using experimentally measured HRIRs
	i) Time domain convolution equation for FIR causal filters.
		CODE:
			For Azimuth:  see "time_domain_convo_azimuth_re.py"
			For Elevation: see "time_domain_convo_elevation_re.py"
		OUTPUT AUDIO FILES:
			For Azimuth:  see "time_domain_azimuth.wav"
			For Elevation:  see "time_domain_elevation.wav"
	ii) Frequency domain FIR filtering using the “overlap and add” block processing method.
		CODE:
			For Azimuth:  see "frequency_domain_convo_azimuth_re.py"
			For Elevation: see "frequency_domain_convo_elevation_re.py"
		OUTPUT AUDIO FILES:
			For Azimuth:  see "frequency_domain_azimuth.wav"
			For Elevation:  see "frequency_domain_elevation.wav"
b) Using HRIR filters designed from a basic synthetic model of HRTFs: 
	CODE:
		for frequency domain: see "frequency_hrtf_re.py"
		for time domain: see "time_hrtf_re.py"
	OUTPUT AUDIO FILE:
		for frequency domain: see "frequency_hrtf.wav"
		for time domain: see "time_hrtf.wav"
