# denoise-mad

MultiAlpha Denoising Algorithm --- Denoise monochannel files by using alpha-stable theory applied on a parameterized Wiener filter. 


Parameters: 

1 -- Number of iterations
nb_it: int (between 1 and 5) /!\ BE CAREFUL /!\
      number of iteration of MAD (above 5 iterations, some instabilities occurs and target signal go into noise signal)


2 -- characteristic exponent
The difference between the two alpha must not be too large (|alpha_s - alpha_n| < 1)

alpha_s: 0 < double <= 2
            characteristic exponent (impulsiveness) of the target source. The smaller the alpha, the more impulsive the signal is assumed to be. 

alpha_no: 0 < double <= 2
            characteristic exponent (impulsiveness) of the noise. he smaller the alpha, the more impulsive the noise is assumed to be. 


3 -- Time frame for average horizon 
The larger the number of time frames, the more stationary the signal is assumed to be. 

deltaTs: int
    number of time frame for average horizon (speech) (in general, between 3 an 10 will be great)

deltaTno: int
    number of time frame for average horizon (noise)
          (ALWAYS CONSIDER deltaTno >> deltaTs. If the noise seems to be stationnary, take a large deltaTno)
