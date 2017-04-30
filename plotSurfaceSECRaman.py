import csv
from Tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
import matplotlib
matplotlib.use("qt5agg") #Faster with Qt, could also be TKAgg
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import spyctra
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

plot2D = False
dots = True
wave2raman = True
ramanlaser = 785.0
fromExcel = False
STARTVALUE = 1000
ENDVALUE = 1800
IND_SPECTRUM = 330
removeBaseline = False
plotInterpolated = True
interpX = 1
interpT = 0.05
imgInterpolation = 'spline36'
plot3D = 'wireframe'
#plot3D = 'surface'
#mycolormap = 'CMRmap'
mycolormap= 'jet'
#mycolormap='gnuplot2'

if dots:
    dotsreplace1 = ''
    dotsreplace2 = ''
    dotsreplace3 = ','
else:
    dotsreplace1 = ','
    dotsreplace2 = '.'
    dotsreplace3 = '.'

class getData():
    def __init__(self):
        # root = Tk()
        # root.withdraw()
        # root.focus_force()
        # self.filename = askopenfilename(parent=root)
        self.filename = 'data4-dot.csv'
        #self.filename = 'datatable.csv'

        self.xdata, self.zdata, self.tdata = self.get_data_from_CSV()

        startidx, startvalue = self.find_nearest(self.xdata, STARTVALUE)
        endidx, endvalue = self.find_nearest(self.xdata, ENDVALUE)

        print startidx, startvalue
        print endidx, endvalue

        self.xdata = self.xdata[startidx:endidx]
        for i in range(len(self.zdata)):
            self.zdata[i] = self.zdata[i][startidx:endidx]

        # plt.plot(xdata, ydata, lw=1.5, label='initial')


    def get_data_from_CSV(self):
        L = []
        reader = csv.reader(open(self.filename), delimiter=';')
        for row in reader:
            L.append(row)
        numrows = len(L)
        numcolumns = len(L[1])
        print numrows, numcolumns

        xdata = L[1][20:numcolumns - 10]
        for i in range(len(xdata)):
            # print xdata[i]
            if type(xdata[i]) is not float:
                xdata[i] = float(xdata[i].replace(dotsreplace3, '').replace(dotsreplace1, dotsreplace2))
                if wave2raman:
                    xdata[i] = (1 / ramanlaser - 1 / xdata[i]) * 1e7

        zdatatotal = []
        for j in range(numrows - 2):
            # print k
            if j < numrows - 3:
                zdata = L[j + 2][20:numcolumns - 10]
                zdatatotal.append(zdata)

        for zdataA in zdatatotal:
            for k in range(len(zdataA)):
                zdata = zdataA[k].replace(dotsreplace3, '').replace(dotsreplace1, dotsreplace2)
                zdata = float(zdata)
                zdataA[k] = zdata


        tdata = []
        for i in range(2, numrows-1):
            data = L[i][0].replace(dotsreplace3, '').replace(dotsreplace1, dotsreplace2)
            data = float(data)
            tdata.append(data)

        return xdata, zdatatotal, tdata


    def find_nearest(self, array,value):
        array = np.array(array)
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]

    def return_data(self):
        return self.xdata, self.zdata, self.tdata

class plot3DSECRaman():
    def __init__(self):

        getdata = getData()
        xdata, zdatatotal, tdata = getdata.return_data()

        if removeBaseline:
            baselines = []
            diffs = []
            utiles = Utiles()
            for i in range(len(zdatatotal)):
                baseline, diff = utiles.remove_baseline(zdatatotal[i])
                baselines.append(baseline)
                diffs.append(diff)
            zdatatotal = diffs

        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.gca(projection='3d')

        if plot3D == 'wireframe':
            self.plotWireframe(xdata, tdata, zdatatotal)
        else:
            self.plotSurface(xdata, tdata, zdatatotal)

    def plotSurface(self, xdata, tdata, zdata):

        if plotInterpolated:
            #Other possibility
            #Z = ndimage.zoom(Zp, 3)
            interpZ = interpolate.interp2d(xdata, tdata, zdata, kind='cubic')
            xnew = np.arange(xdata[0], xdata[len(xdata)-1], interpX)
            tnew = np.arange(tdata[0], tdata[len(tdata)-1], interpT)
            Zdata = interpZ(xnew, tnew)
            Xdata, Tdata = np.meshgrid(xnew, tnew)
        else:
            Xdata, Tdata = np.meshgrid(xdata, tdata)
            Zdata = np.array(zdata)

        surf = self.ax.plot_surface(Xdata, Tdata, Zdata, shade=True, cmap=mycolormap, linewidth=0)
        self.setAxes(Xdata, Tdata)


    def plotWireframe(self, xdata, tdata, zdata):
        #TODO: improve, use scikit-spectra
        Xdata, Tdata = np.meshgrid(xdata, tdata)
        Zdata = np.array(zdata)

        surf = self.ax.plot_surface(Xdata, Tdata, Zdata, rstride=30, cstride=30, shade=False,
                                     cmap='jet', linewidth=0.5)
        m = plt.cm.ScalarMappable(surf.norm)
        surf.set_edgecolors(m.to_rgba(surf.get_array()))
        self.setAxes(Xdata, Tdata)

    def setAxes(self, xdata, tdata):
        plt.axis([xdata.min(), xdata.max(), tdata.min(), tdata.max()])
        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel('t (s)')
        self.ax.set_zlabel('Intensity (counts)')
        plt.show()

class plot2DSECRaman():
    def __init__(self):

        getdata = getData()
        xdata, zdatatotal, tdata = getdata.return_data()

        if removeBaseline:
            baselines = []
            diffs = []
            utiles = Utiles()
            for i in range(len(zdatatotal)):
                baseline, diff = utiles.remove_baseline(zdatatotal[i])
                baselines.append(baseline)
                diffs.append(diff)
            zdatatotal = diffs


        if plotInterpolated:
            self.plotInterpolatedPlot(xdata, tdata, zdatatotal)
        else:
            self.plotImage(xdata, tdata, zdatatotal)


    def plotInterpolatedPlot(self, xdata, tdata, zdata):
        #Other possibility
        #Z = ndimage.zoom(Zp, 3)
        interpZ = interpolate.interp2d(xdata, tdata, zdata, kind='cubic')
        tnew = np.arange(xdata[0], xdata[len(xdata)-1], interpX)
        ynew = np.arange(tdata[0], tdata[len(tdata)-1], interpT)
        znew = interpZ(tnew, ynew)

        plt.pcolormesh(tnew, ynew, znew, cmap='jet', vmin=znew.min(), vmax=znew.max())

        self.setAxes(tnew, ynew)

    def plotImage(self, xdata, tdata, zdata):

        X, Y = np.meshgrid(xdata, tdata)
        Z = np.array(zdata)

        plt.imshow(Z, cmap='jet', vmin=Z.min(), vmax=Z.max(), extent=[X.min(), X.max(), Y.min(), Y.max()],
                   interpolation=imgInterpolation, origin='lower', aspect='auto')
        self.setAxes(X, Y)


    def setAxes(self, xdata, tdata):
        plt.axis([xdata.min(), xdata.max(), tdata.min(), tdata.max()])
        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel('t (s)')

        clb = plt.colorbar()
        clb.set_label('Intensity (counts)')
        plt.show()


class Utiles():
    def remove_baseline(self, zdata):
        baseline = spyctra.arPLS(zdata, lambda_=1.e3)
        diff = np.array(zdata) - np.array(baseline)

        return baseline, diff

if __name__ == "__main__":
    if plot2D:
        plot2DSECRaman()
    else:
        plot3DSECRaman()
