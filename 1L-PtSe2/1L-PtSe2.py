'''
    object: relax monolayer 1T-PtSe2 and calculate the band structure
    author: pfsuo@whu.edu.cn
'''
from ase.build.surface import mx2
from ase.calculators.vasp import Vasp
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

## build model
def my_model(a=3.77,thickness=2.6,vacuum=10): 
    atoms = mx2('PtSe2','1T',a=a, thickness=thickness)  # monolayer 1T-PtSe2
    atoms.center(vacuum=vacuum,axis=2)    # set the vacuum layer: 20 angstrom
    atoms.pbc = True
    return atoms

def calc_scf():
    calc = Vasp(
            xc = 'optb86b-vdw',
            setups='recommended',
            kpts=(12,12,1),
            istart=0,
            icharg=2,
            encut=520,
            ncore=4,
            ismear=-5,
            prec='Accurate',
            ediff=1e-6)     # the scf calculator template
    return calc

## bandstructure calculation
def band_calc(atoms):
    calc = calc_scf()
    atoms.calc = calc
    calc.set(
            directory='band',
            lsorbit=True,
            command='mpirun -np 24 vasp_ncl')   # SOC scf calculator
    atoms.get_potential_energy()
    calc.set(isym=0,
            kpts={'path':'MKGM','npoints':200},
            istart=1,
            icharg=11,
            ismear=0,
            lwave=False,
            lcharg=False,
            lorbit=11)           # SOC band calculator
    atoms.get_potential_energy()
    homo, lumo = calc.get_homo_lumo()       # get the HOMO and LUMO from band calculation
    e_nk = calc.band_structure().energies[0].T - homo     # get band data with reference to HOMO
    path = atoms.cell.bandpath('MKGM',npoints=200)
    x, X, _ = path.get_linear_kpoint_axis()
    np.savetxt('band/e_nk.dat',e_nk)           # save the band data
    with open('band/kpath.dat','w') as f:
        for k in x:
            print(k,file=f)          # save the kpath axis data
    with open('band/highk.dat','w') as f:
        for k in X:
            print(k,file=f)         # save the high K data

## plot bandstructure
def plot_band(figsize=(6,5)):
    plt.figure(figsize=figsize)
    e_nk = np.loadtxt('band/e_nk.dat')
    x = np.loadtxt('band/kpath.dat')
    X = np.loadtxt('band/highk.dat')
    for e_n in e_nk:
        plt.plot(x, e_n, c='r', lw=2)
    plt.axhline(y=0,c='k',alpha=0.5,lw=1,ls='--')
    for i in X:
        plt.axvline(x=i,c='k',alpha=0.5,lw=1,ls='--')
    plt.axis([x[0],x[-1],-3,3])
    plt.xticks(X,['M','K','Γ','M'],size=15)
    plt.yticks(size=14)
    plt.ylabel(r'$\varepsilon_n(k) - \varepsilon_{\mathrm{VBM}}$ (eV)', size=20)
    plt.title('band structure of 1L-PtSe$_2$ with SOC',size=20)
    plt.savefig('band/band.svg')
    plt.close()

def main():
    atoms = my_model()
    calc = calc_scf()
    atoms.calc = calc
    calc = calc.set(encut=750,ediffg=-0.01,isif=3,ibrion=2,nsw=100)    # vc-relax calculator
    with open('OPTCELL','w') as f:
        f.write('100\n110\n001')    # fixed C axis, refer to http://blog.wangruixing.cn/2019/05/05/constr/ 
    atoms.get_potential_energy()
    atoms = read('CONTCAR')
    band_calc(atoms)
    plot_band() 

if __name__=='__main__':
    main()
