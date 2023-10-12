from spectralpy.data import *

if __name__ == '__main__':

    night = 0
    obj_name = 'pCygni'


    # extracting informations
    obj = DATA_ALL[NIGHTS[night]][obj_name]
    # collecting in different variables
    obj_fit, obj_lamp = obj[:2] 

    # appending the path
    obj_fit = data_file_path(night, obj_name, obj_fit)
    obj_lamp = data_file_path(night, obj_name, obj_lamp)

    files = [obj_fit,obj_lamp]

    if obj_name == 'giove' or obj_name == 'arturo':
        obj_flat = obj[-1]
        obj_flat = data_file_path(night, obj_name, obj_flat)
        files += [obj_flat]

    for element in files:
        _ = get_data_fit(element)
        plt.show()