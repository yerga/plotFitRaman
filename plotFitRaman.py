import csv
from Tkinter import Tk
from tkFileDialog import askopenfilename
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import fitraman
import spyctra
import pandas as pd

dots = True
wave2raman = True
ramanlaser = 785.0
fromExcel = False
STARTVALUE = 900
ENDVALUE = 1400
IND_SPECTRUM = 330

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
        #self.filename = 'data4-dot.xlsx'

        if fromExcel:
            xdata, ydata = self.get_data_from_Excel()
        else:
            xdata, ydata = self.get_data_from_CSV()

        startidx, startvalue = self.find_nearest(xdata, STARTVALUE)
        endidx, endvalue = self.find_nearest(xdata, ENDVALUE)

        print startidx
        print endidx

        self.xdata = xdata[startidx:endidx]
        self.ydata = ydata[startidx:endidx]

        # plt.plot(xdata, ydata, lw=1.5, label='initial')

    def get_data_from_Excel(self):

        X1 = []
        Y1 = []

        excelfile = pd.ExcelFile(self.filename)
        datarows1 = excelfile.parse()
        datarows = datarows1.get_values()
        numrows = len(datarows)
        numcolumns = len(datarows[0])
        #headers = [item for item in datarows1.head(0)]


        xdata = datarows[0][20:numcolumns - 10]
        for i in range(len(xdata)):
            # print xdata[i]
            if type(xdata[i]) is not float:
                xdata[i] = float(xdata[i].replace(dotsreplace3, '').replace(dotsreplace1, dotsreplace2))
            if wave2raman:
                xdata[i] = (1 / ramanlaser - 1 / xdata[i]) * 1e7

        ydata = datarows[IND_SPECTRUM][20:numcolumns - 10]
        for j in range(len(ydata)):
            print type(ydata[j])
            if type(xdata[i]) is not float:
                ydataA = ydata[j].replace(dotsreplace3, '').replace(dotsreplace1, dotsreplace2)
                ydataA = float(ydataA)
                ydata[j] = ydataA

        #
        # for j in range(0, numcolumns, 2):
        #     x1 = [item[j] for item in datarows]
        #     y1 = [item[j + 1] for item in datarows]
        #     # print j
        #
        #     X1.append(x1)
        #     Y1.append(y1)


        return xdata, ydata


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

        ydata = L[IND_SPECTRUM][20:numcolumns - 10]
        for j in range(len(ydata)):
            ydataA = ydata[j].replace(dotsreplace3, '').replace(dotsreplace1, dotsreplace2)
            ydataA = float(ydataA)
            ydata[j] = ydataA

        return xdata, ydata


    def find_nearest(self, array,value):
        array = np.array(array)
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]

    def return_data(self):
        return self.xdata, self.ydata


class plotFitRaman():
    def __init__(self):

        getdata = getData()
        xdata, ydata = getdata.return_data()
        #plt.plot(xdata, ydata, lw=1.0, label='diff')
        #plt.show()

        baseline, diff = self.remove_baseline(ydata)
        # plt.plot(xdata, baseline, lw=1.0, label='baseline')
        plt.plot(xdata, diff, lw=1.0, label='diff')
        #plt.show()

        # TODO: adjust number of peaks to find, initialwidth, curve type, sigmavalue
        peaks_to_find = 7
        initialwidth = 10
        fitraman.CURVE = "Gaussian"
        # fitraman.SIGMAVALUE = np.full(len(subtract), 5)
        params, fit, ys, n_peaks = fitraman.predict_and_plot_lorentzians(xdata, diff, peaks_to_find, initialwidth)
        print 'params: ', params

        peakdata = []
        for j in range(0, len(params), 3):
            ctr = params[j]
            amp = params[j + 1]
            width = params[j + 2]
            peakdata.append(["%.2f" % ctr, "%.2f" % amp, "%.2f" % width])
            ysignal = fitraman.lorentzian(xdata, amp, ctr, width)
            ymax = np.max(ysignal)
            idxmax = np.argmax(ysignal)
            # plot max points in fitted curves
            # plt.plot(xdata[idxmax], ymax, ls='', marker='x')
            #Plot max points in experimental curve
            plt.plot(xdata[idxmax], diff[idxmax], ls='', marker='x')
            plt.plot(xdata, ysignal, ls='-')

        self.printpeakdata(peakdata)
        plt.plot(xdata, fit, 'r-', label='fit', c='red', lw=1.2, ls='--')

        plt.legend(loc='upper right', fontsize=10, shadow=True)
        plt.show()


    def remove_baseline(self, ydata):
        baseline = spyctra.arPLS(ydata, lambda_=1.e3)
        diff = np.array(ydata) - np.array(baseline)

        return baseline, diff


    def printpeakdata(self, peakdata):
        row_format = "{:>15}" * 3
        datatitles = ["Peak", "Height", "FWHM"]
        print row_format.format(*datatitles)
        for row in peakdata:
            print row_format.format(*row)


if __name__ == "__main__":
    plotFitRaman()
