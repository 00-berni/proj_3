import numpy as np
import matplotlib.pyplot as plt
import spectralpy as spc
# from spectralpy import TARGETS

label = lambda i,arr,name : name if i==arr[0] else ''

def plot_line(lines: list[float], name: str, color: str, minpos: float) -> None:
    for line in lines:
        plt.axvline(line,0,1,color=color,label=label(line,lines,name))
        plt.annotate(name,(line,minpos),(line+10,minpos))


b_name = ['H$\\alpha$', 'H$\\beta$', 'H$\\gamma$', 'H$\\delta$', 'H$\\epsilon$', 'H$\\xi$', 'H$\\eta$','H$\\theta$','H$\\iota$','H$\\kappa$']
balmer = [6562.79, 4861.350, 4340.472, 4101.734, 3970.075, 3889.064, 3835.397, 3797.909, 3770.633, 3750.151]
bal_err = [0.03,0.05]+[0.006]*7
feI  = [7610.2676,  7635.8482,  5896.7357, 5274.9807, 4300.2036, 4384.6718, 4401.4425,  4459.3521, 4351.5437]
feII = [7636.2373, 7611.2601, 6871.6994, 6496.9415, 6497.2764, 6497.4985,  5175.3973, 5274.5277, 4384.31313, 4459.67779, 4351.76199, 4336.30962]
tiI  = [6497.683, 4300.4848, 4300.5538, 4301.0787, 4322.1141, 4321.7709 ]
tiII = [4300.0424,  4301.29545, 4350.83776  ]
neI  = [ 5274.0406, 4402.380, 4460.1758, 4336.2251 ]
neII = [4384.1063, 4384.2194, 4322.372]
oI  = [5274.967,5275.123]
oII = [4384.446, 4351.260, 4336.859, 4322.4477]
mgI  = [4351.9057]
mgII = [4384.637]
arI  = []
arII = [4401.75478] 

def display_lines(minpos: float, edges: tuple[float, float]) -> None:
    for (b,err,name) in zip(balmer,bal_err,b_name):
        # b += 100
        if edges[0] <= b <= edges[1]:
            plt.axvline(b,0,1, color='blue',label=label(b,balmer,'H I'))
            plt.annotate(name,(b,minpos),(b+10,minpos))
            plt.axvspan(b-err,b+err,0,1,color='blue',alpha=0.3)
    # plot_line(feI, 'Fe I','orange',minpos)
    # plot_line(feII,'Fe II','yellow',minpos)
    # plot_line(tiI, 'Ti I','violet',minpos)
    # plot_line(tiII,'Ti II','plum',minpos)
    # plot_line(neI, 'Ne I','green',minpos)
    # plot_line(neII,'Ne II','lime',minpos)
    # plot_line(oI, 'O I','deeppink',minpos)
    # plot_line(oII,'O II','hotpink',minpos)
    # plot_line(mgI, 'Mg I','red',minpos)
    # plot_line(mgII,'Mg II','tomato',minpos)
    # plot_line(arI, 'Ar I','aqua',minpos)
    # plot_line(arII,'Ar II','cyan',minpos)
    plt.legend()



if __name__ == '__main__':

    ## Calibration with Vega
    obs_night = '22-07-26_ohp'
    target_name = 'vega' 
    selection = 0

    ord_lamp = 2
    ord_balm = 2
    display_plots = False
    vega, v_lamp = spc.calibration(obs_night, target_name, selection, ord_lamp=ord_lamp, ord_balm=ord_balm, display_plots=False)

    target_name = 'giove'    
    jupiter, lamp = spc.calibration(obs_night, target_name, selection, angle=vega.angle, other_lamp=v_lamp, display_plots=display_plots)

    spc.show_fits(jupiter,show=True)
    sel1 = 46
    sel2 = 112
    j1 = jupiter.data[sel1,:].copy()
    j2 = jupiter.data[sel2,:].copy()

    plt.figure()
    plt.plot(jupiter.lines,j1,'.-',label=sel1)
    plt.plot(jupiter.lines,j2,'.-',label=sel2)
    plt.legend()
    plt.show()

    ## Jupiter 

