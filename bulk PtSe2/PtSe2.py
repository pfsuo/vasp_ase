'''
    object: relax bulk 1T-PtSe2 and calculate the band structure
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
def my_model(a=3.77,c=5): 
    atoms = mx2('PtSe2','1T',a=a)   # monolayer 1T-PtSe2
    atoms.cell[2,2] = c    # monolayer --> bulk
    atoms.pbc = True
    return atoms

def calc_scf():
    calc = Vasp(
            xc = 'optb86b-vdw',
            setups='recommended',
            kpts=(12,12,5),
            istart=0,
            icharg=2,
            encut=520,
            ncore=4,
            ismear=0,
            sigma=0.02,
            prec='Accurate',
            ediff=1e-6)    # scf calculator template
    return calc

## relax 
def atoms_relax():
    atoms = my_model()
    calc = calc_scf()
    atoms.calc = calc
    calc.set(
            directory='relax',
            encut=750,
            ediffg=-0.01,
            isif=3,
            ibrion=2,
            nsw=100,
            lwave=False,
            lcharg=False)  # relax calculator
    atoms.get_potential_energy()
    atoms = read('relax/CONTCAR')  # the full-relaxed structure
    return atoms

## bandstructure calculation
def band_calc():
    atoms = read('relax/CONTCAR')  # the full-relaxed structure
    calc = calc_scf()
    atoms.calc = calc
    calc.set(
            directory='band',
            lsorbit=True,
            command='mpirun -np 24 vasp_ncl')  # SOC scf calculator
    atoms.get_potential_energy()
    ef = calc.get_fermi_level()      # Fermi level from scf calculation 
    calc.set(isym=0,
            kpts={'path':'GMKGALHA','npoints':800},
            istart=1,
            icharg=11,
            lwave=False,
            lcharg=False,
            lorbit=11)            # SOC band structure calculator
    atoms.get_potential_energy()
    e_nk = calc.band_structure().energies[0].T - ef    # band data relative to Fermi level
    path = atoms.cell.bandpath('GMKGALHA',npoints=800)
    x, X, _ = path.get_linear_kpoint_axis()
    np.savetxt('band/e_nk.dat',e_nk)   # save the e_nk data
    with open('band/kpath.dat','w') as f:
        for k in x:
            print(k,file=f)      # save the kpath axis data
    with open('band/highk.dat','w') as f:
        for k in X:
            print(k,file=f)      # save the high K points data

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
    plt.xticks(X,['Γ','M','K','Γ','A','L','H','A'],size=15)
    plt.yticks(size=14)
    plt.ylabel(r'$\varepsilon_n(k) - \varepsilon_{\mathrm{F}}$ (eV)', size=20)
    plt.title('band structure of bulk PtSe$_2$ with SOC',size=20)
    plt.savefig('band/band.svg')
    plt.close()

def main():
    atoms = atoms_relax()
    band_calc()
    plot_band((8,5)) 

if __name__=='__main__':
    main()
