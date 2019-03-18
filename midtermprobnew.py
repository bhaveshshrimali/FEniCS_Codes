import numpy as np
import matplotlib 
matplotlib.use('ps')
matplotlib.rcParams['ps.usedistiller']='ghostscript'

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt 
import pickle as pkl
import os
from matplotlib.ticker import AutoMinorLocator,LogLocator
from dolfin import *
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
plt.close('all')
props = dict(boxstyle='Square', facecolor='none',edgecolor='black', alpha=0,lw=1)

#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rc('text',usetex=True)

plt.rcParams['xtick.top']='True'
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.right']='True'
plt.rcParams['ytick.direction']='in'
plt.rcParams['ytick.labelsize']=22
plt.rcParams['xtick.labelsize']=22
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.major.size']=6
plt.rcParams['xtick.minor.size']=3
plt.rcParams['ytick.major.size']=6
plt.rcParams['ytick.minor.size']=3
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',
            r'\usepackage{siunitx}',
            r'\usepackage{amsfonts}',
            r'\usepackage{xcolor}']
plt.rcParams['lines.linewidth']=2

#parameters['form_compiler']['cpp_optimize']=True

#comm=mpi_comm_world()
cdir = os.getcwd()
set_log_level(30)

def plotme3D(fig,xLabel,yLabel,zLabel):
    ax=Axes3D(fig)
    ax.set_xlabel(xLabel,fontsize=22,labelpad=10)
    ax.set_ylabel(yLabel,fontsize=22,labelpad=10)
    ax.set_zlabel(zLabel,fontsize=22,labelpad=15)
    return ax

#print(matplotlib.__version__)
def mainFunction(n,elemType,aVal):
    if elemType == 'P1':
        degFE = 1
    elif elemType == 'P2' or elemType == 'Q2':
        degFE = 2
    
    if type(aVal) != str:
        aVal = str(aVal)
    
    if elemType[0] == 'P':
        mesh=UnitSquareMesh(n,n)
    elif elemType[0] == 'Q':
        mesh=UnitSquareMesh(n,n)
    os.makedirs(os.path.join(cdir,'Results',elemType),exist_ok=True)
    # meshPath=os.path.join(cdir,'Results',elemType,'mesh{}'.format(int(n))+'.eps') 
    # plt.figure(figsize=(8,8))
    # plot(mesh) 
    # plt.savefig(meshPath)
    # plt.close()

    Vh = FunctionSpace(mesh,'CG',degFE)

    #def dirch_bdry(x,on_boundary):
    #    return near(x[0],0.) or near(x[1],1.) or near(x[0],1.) or near(x[1],0.)

    u_g = Constant(0.)

    dirch_BC = DirichletBC(Vh,u_g,DomainBoundary())

    u = TrialFunction(Vh)
    v = TestFunction(Vh)
    f = Expression('1.',degree=0)
    a = Expression(aVal,degree=0)
    b = Constant((1.,0.))
    b_uv = dot(grad(a*u) , grad(v))*dx + v * dot(b,grad(u)) * dx
    f_v = f*v*dx
    u_FE = Function(Vh)

    solve(b_uv == f_v,u_FE,dirch_BC)
    uFE_sampled = np.zeros((100,100))
    
    if float(aVal) == 0.01:
        aStr='0_01'
    elif float(aVal) == 1.:
        aStr='1'
    res_path = os.path.join(cdir,'Results',elemType,'uplot{}'.format(n)+aStr+'.eps')

    for ix,x in enumerate(np.linspace(0,1,100)):
        for iy,y in enumerate(np.linspace(0,1,100)):
            uFE_sampled[ix,iy] = u_FE(x,y)
    
    smpl_x,smpl_y = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
    figr=plt.figure(figsize=(8,8))
    ax=plotme3D(figr,r'$\bf x$',r'$\bf y$',r'$u({\bf x},{\bf y})$') 
    surf=ax.plot_surface(smpl_x,smpl_y,uFE_sampled,cmap=cm.jet) 
    figr.colorbar(surf,shrink=0.8)
    figr.savefig(res_path) 
    figr.savefig(os.path.join(cdir,'Results',elemType,'uplot{}'.format(n)+aStr+'.png'))
    figr.tight_layout()
    plt.close()
    # vtkPath = os.path.join(cdir,'Results',elemType,'uplot{}'.format(n)+aStr+'.xdmf')
    # with XDMFFile(res_path) as wfil:
        # wfil.write(u_FE)
    return u_FE
#mainFunction(64,1,1.)
if __name__=='__main__':
    # elemtypes=['Q2']
    elemtypes=['P1','P2']#,'Q2']
    meshSizes = 1./np.array([2**i for i in range(2,8)],float)
# Storing the "exact" solution to compute the error norms
    for elems in elemtypes:
        l2_err_nrm = np.zeros((2,6))
        h1_err_nrm = np.zeros((2,6))
        linf_err_nrm = np.zeros((2,6))
        convratesL2 = np.zeros((2,5))
        convratesH1 = np.zeros((2,5))
        VhExact = FunctionSpace(UnitSquareMesh(512,512,),'CG',int(elems[-1]))
        for i_vals,a_val in enumerate([1.,0.01]):
            u_exact = mainFunction(512,elems,a_val)
            for meshIdxs,nMesh in enumerate([4,8,16,32,64,128]):
                # VhExact = FunctionSpace(UnitSquareMesh.create(512,512,CellType.Type.quadrilateral),'CG',5)
                u_FE = mainFunction(nMesh,elems,a_val)
                # u_exact_projected = Function(VhExact)
                # u_exact_projected.interpolate(u_exact)
                u_FE_interpolated = Function(VhExact)
                u_FE_interpolated.interpolate(u_FE)
                
                err = u_FE_interpolated - u_exact
                err_projected = project(err,VhExact)
                l2_err_nrm[i_vals,meshIdxs] = errornorm(u_FE_interpolated,u_exact,'l2',degree_rise=0)
                h1_err_nrm[i_vals,meshIdxs] = (errornorm(u_FE_interpolated,u_exact,'h1',degree_rise=0)**2 - l2_err_nrm[i_vals,meshIdxs]**2)**(0.5)
                linf_err_nrm[i_vals,meshIdxs] = norm(err_projected.vector(),'linf')
                # err = u_FE - u_exact_projected
                # err_projected = project(err,VhExact)
                # l2_err_nrm[i_vals,meshIdxs] = errornorm(u_FE,u_exact_projected,'l2',degree_rise=3)
                # h1_err_nrm[i_vals,meshIdxs] = (errornorm(u_FE,u_exact_projected,'h1',degree_rise=3)**2 - l2_err_nrm[i_vals,meshIdxs]**2)**(0.5)
                # linf_err_nrm[i_vals,meshIdxs] = norm(err_projected.vector(),'linf')
            convratesL2[i_vals,:] = np.log(l2_err_nrm[i_vals,1:]/l2_err_nrm[i_vals,:-1])/np.log(meshSizes[1:]/meshSizes[:-1])
            convratesH1[i_vals,:] = np.log(h1_err_nrm[i_vals,1:]/h1_err_nrm[i_vals,:-1])/np.log(meshSizes[1:]/meshSizes[:-1])
        
        fig,axs=plt.subplots(1,2,figsize=(15,8))
        ax1,ax2=axs
        ax1.loglog(meshSizes,l2_err_nrm[0],'-o',label='$a=1$') 
        ax1.loglog(meshSizes,l2_err_nrm[1],'-o',label='$a=0.01$')
        ax1.set_ylabel('$L^2$ norm of the error',fontsize=22)
        ax1.set_xlabel('$h$',fontsize=22)
        ax1.legend(loc=0,fontsize=22,fancybox=True,edgecolor='k')
        
        ax2.loglog(meshSizes,h1_err_nrm[0],'-o',label='$a=1$')
        ax2.loglog(meshSizes,h1_err_nrm[1],'-o',label='$a=0.01$')
        ax2.set_ylabel('$H^1$ semi-norm of the error',fontsize=22)
        ax2.set_xlabel('$h$',fontsize=22)
        ax2.legend(loc=0,fontsize=22,fancybox=True,edgecolor='k') 
        fig.tight_layout()
        fig.savefig(os.path.join(cdir,'Results',elems,'L2H1.eps'))
        fig.savefig(os.path.join(cdir,'Results',elems,'L2H1.png'))
        plt.close()


        pd.DataFrame(l2_err_nrm).to_excel(os.path.join(cdir,'Results',elems,'L2norm.xlsx'),header=False,index=False)
        pd.DataFrame(h1_err_nrm).to_excel(os.path.join(cdir,'Results',elems,'H1Seminorm.xlsx'),header=False,index=False)
        pd.DataFrame(linf_err_nrm).to_excel(os.path.join(cdir,'Results',elems,'Linfnorm.xlsx'),header=False,index=False)
        pd.DataFrame(convratesL2).to_excel(os.path.join(cdir,'Results',elems,'convergenceL2.xlsx'),header=False,index=False)
        pd.DataFrame(convratesH1).to_excel(os.path.join(cdir,'Results',elems,'convergenceH1.xlsx'),header=False,index=False)
        #print(convratesL2)
        #print(convratesH1)






