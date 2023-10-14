import numpy as np
from spectralpy.data import *

if __name__ == '__main__':

    ch_obs = 0
    ch_obj = 'betaLyr'
    display_plots = False


    # collecting data
    obj_fit, lims_fit, obj_lamp, lims_lamp = extract_data(ch_obs, ch_obj,sel=['obj','lamp'])
    # extracting fits data and correcting for inclination
    hdul, sp_data, angle = get_data(ch_obj,obj_fit,lims_fit, display_plots=display_plots)

    _, sp_lamp = get_data_fit(obj_lamp,lims=lims_lamp, title='Row spectrum lamp', n=1, display_plots=display_plots)
    _, sp_lamp = angle_correction(sp_lamp, angle=angle, display_plots=display_plots)
    
    max_lamp = np.argmax(sp_lamp,axis=0)
    height = int(np.mean(max_lamp))

    print(sp_lamp.shape)

    plt.figure()
    plt.plot(np.arange(sp_lamp.shape[1]),sp_lamp[max_lamp,:],'+r')
    plt.axhline(height,0,1,color='b')
    plt.imshow(sp_lamp)
    plt.show()
