import os
import numpy as np
import scipy.io
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import pandas as pd
import matplotlib.colors as colors
import scipy.misc 
import copy
import pickle
import time
from skimage import transform 
from matplotlib import pyplot as plt, image
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.cm as cmx
from scipy import stats 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot_args(which, plot_type='test_acc'):
    a = {}
    a['plot_type'] = plot_type
    a['which'] = which
    a['v_datas'] = None
    a['v_areas'] = None
    a['zoom'] = True
    a['acc'] = 'fine_acc'
    a['title'] = ''
    a['tadd'] = '' 
    a['save'] = False
    a['n_e_vs'] = [10]
    a['n_epochs'] = 100
    a['show'] = True
    a['f_skip'] = None
    a['x1'] = 85
    a['x2'] = 100
    a['y1'] =.445
    a['y2']= .505
    a['zoom_n'] = 2.5
    return a 

def get_results(fig, ax, args, f_dir, x, color, label, axins = None, ratio = None, initial_ratio = None, alpha=None, n_e_v=None, v_area=None, v_data=None, acc='fine_acc'):
    accs = list()
    c_accs = list()
    count = 0
    print('\n')
    print('Ratio = ' + str(ratio))
    print('Initial Ratio = ' + str(initial_ratio))
    print('alpha = ' + str(alpha))
    print('n_e_v = ' + str(n_e_v))
    max_avgs = list()
    max_epochs = list()

    def n_e_v_check(n_e_v, f):
        add = False 
        if n_e_v is None and 'n_e_v' not in f:
            add = True 
        elif 'n_e_v=' + str(n_e_v) in f:
            if(n_e_v == 10 and 'n_e_v=100' in f):
                add = False
            else: 
                add = True
        return add

    def v_area_check(v_area, v_data, n_e_v, f, alpha=False):
        add = False
        if(v_area is None):
            add = n_e_v_check(n_e_v, f)
        else:
            if('v_area'+str(v_area) in f and 'v_data='+str(v_data) in f):
                if(str(v_data)=='random'):
                    if 'randomnonV1' not in f:
                        if alpha:
                            return True

                        else: 
                            add = n_e_v_check(n_e_v, f)
                else:
                    if alpha:
                        return True

                    else: 
                        add = n_e_v_check(n_e_v, f)
        return add

    
    def ratio_check(ratio, f, v_area, v_data, n_e_v):
        add = False
        if('ratio=' + str(ratio) in f):
            add = v_area_check(v_area, v_data, n_e_v, f)
        return add

    def init_ratio_check(init_ratio, f, v_area, v_data, n_e_v):
        add = False
        if('initial_ratio=' + str(init_ratio) in f):
            print('Yes -- initial ratio in f')
            print(f)
            add = v_area_check(v_area, v_data, n_e_v, f)
        return add 

    def alpha_check(alpha, f, v_area, v_data):
        add = False
        if('alpha=' + str(alpha) in f):
            add = v_area_check(v_area, v_data, n_e_v=None, f=f, alpha=True)
            
        return add 
            
    
    for f in os.listdir(f_dir):
        add = False 
        if(f != args['f_skip']):
            if 'results' in f:
                if ratio is not None: 
                    add = ratio_check(ratio, f, v_area, v_data, n_e_v)
                elif initial_ratio is not None:
                    
                    add = init_ratio_check(initial_ratio, f, v_area, v_data, n_e_v)
                elif alpha is not None:
                    add = alpha_check(alpha, f, v_area, v_data)
                else:
                    add = True 

        if (add):
            #print(f)
            count += 1
            df = pd.read_csv(f_dir + f)
            accs.append(df['test_' + str(acc)].values)
            m = max(df['test_' + str(acc)])
            j = df[df['test_' +str(acc)]==m].index.values.astype(int)[0]

            max_avgs.append(100*m)
            max_epochs.append(j)
            print(m)


    if(count==0):
        print('No results found')
        return 0, 0
    else:
        acc_mean = 100*np.mean(accs, axis=0)
        #acc_error = np.std(accs, axis=0)
        acc_error = 100*scipy.stats.sem(accs)
        print('Num results ' + str(count))
        avg_epoch = int(np.mean(max_epochs))
        max_avg  = np.mean(max_avgs)
        #max_err = np.std(max_avgs)
        max_err = scipy.stats.sem(max_avgs)
        print('Max average: ' + str(max_avg) + ' +/- ' + str(max_err))
        print('Avg max epoch at ' + str(avg_epoch))

        
        if(args['plot_type']=='test_acc'):
            ax.plot(x, acc_mean, color=color, label=label)
            ax.fill_between(x, acc_mean - acc_error, acc_mean + acc_error, color = color, alpha=.4)
            if axins is not None: 
                axins.plot(x, acc_mean, color=color)
                axins.fill_between(x, acc_mean - acc_error, acc_mean + acc_error, color = color, alpha=.4)
            #plt.legend()
        return max_avg, max_err




def plot_table(fig, ax, row_labels, max_avgs, errs, fa_mean, fa_error, colors):
    print(len(colors))
    cell_text = [str(round(100*fa_mean, 2)) + ' +-' + str(round(fa_error, 3))]
    for i, m in enumerate(max_avgs):
        cell_text.append(str(round(100*m, 2)) + ' +/-' + str(round(errs[i], 3)))
    cell_text = np.array(cell_text).reshape(1, len(cell_text))
    print(cell_text)
    print(row_labels)
    table = ax.table(cellText=cell_text, colLabels=row_labels, cellLoc='center', rowLoc='center', loc='bottom', bbox=[-.25, -0.4, 1.35, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(13)

    for i in range(len(row_labels)):
        table._cells[(0, i)]._text.set_color(colors[i])
        table._cells[(1, i)]._text.set_color(colors[i])
        table._cells[(1, i)]._text.set_fontsize(13)

    
    plt.subplots_adjust(left=0.2, bottom=0.3)


def get_color(ratio):
    values = range(26)
    #jet = cm = plt.get_cmap('nipy_spectral')
    #cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #
    #return scalarMap.to_rgba(values[idx])
    #ratios = [0, 0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    ratios = [ 0  ,    0.0001,    0.001   ,    0.01    ,       0.1    ,     1.0   ,      2.0   ,        3.0   ,       4.0]
    idx = ratios.index(ratio)
    #########   0         0.001         0.01     0.1        1.0         2.0         3.0      4.0
    #colors = ['#000000', '#4f086d', '#6b1191',   '#9410cc', '#6235cc',  '#1c12e2', '#096656', '#018943', 'orange', 'blue', 'green', 'purple', 'gray']

   ###########   0        0.0001    0.001       0.01           0.1         1.0         2.0           3.0          4.0
    colors = ['#000000', '#1a0000', '#330000',  '#660000',  '#800000', '#990000' ,'#b30000'    , '#e60000'    ,'#ff1a1a']
    return colors[idx]


    
def get_plot(args): 
    
    x = list(range(args['n_epochs']))
    fig, ax = plt.subplots()
    plt.ylabel('Test Set Accuracy (%)', size=20)
    if (args['plot_type']=='scatter'):
        plt.xlabel('ratio of V1 data', size=20)
        axins = None
    else: 
        plt.xlabel('Epochs', size=20)
        if(args['acc']=='fine_acc'):
            plt.axhline(y=(1), color='black', linestyle='--')
            plt.text(60, .015, 'chance accuracy', size=15)
        else:
            plt.axhline(y=(1/20), color='black', linestyle='--')
            plt.text(60, .065, 'chance accuracy', size=15)
        if(not args['zoom']):
            axins = None
        else: 
            axins = zoomed_inset_axes(ax, args['zoom_n'], loc=7)
        
    plt.xticks(size=15)
    plt.yticks(size=20)
    
    colors_list = list()
    
    ####################################### none #######################################
    fa_mean, fa_error, = get_results(fig, ax, args, axins=axins, f_dir = 'log/CORNetZ/', x=x, color=get_color(0), label='r=0', acc=args['acc'])
    colors_list.append(get_color(0))

    ####################################### rest #######################################
    max_avgs = list()
    errs = list()

    if args['plot_type'] == 'scatter':
        max_avgs.append(fa_mean)
        errs.append(fa_error)
    j = 1 

    row_labels = ['r=0']
        
    if(args['which']=='switched'):
        for vd in args['v_datas']:
            print('v data ' + str(vd))
            for va in args['v_areas']:
                print('v area ' + str(va))
                if ('ratios' in args): 
                    for ratio in args['ratios']:
                        for n_e_v in args['n_e_vs']:
                            if vd == 'shuffled':
                                label = 'V1 shuffled'
                            elif vd =='random':
                                label = 'Gaussian (V1-stats)'
                            elif vd =='randomnonV1':
                                label = 'Gaussian (non V1-stats)'
                            else:
                                label=str(vd) +  ' r=' + str(ratio)
                            if(vd==va):
                                folder = 'log/CORNetZ_' + str(va)+'/'
                                vd = None
                                va = None
                            else:
                                folder = 'log/CORNetZ_switched/'
                            if 'colors' not in args:
                                color = get_color(ratio)
                            else:
                                color = args['colors'][j-1]
                            m, e = get_results(fig, ax, args, axins=axins, f_dir = folder, x=x, color=color, label=label, ratio=ratio, n_e_v = n_e_v, v_data = vd, v_area=va, acc=args['acc'])
                            max_avgs.append(m)
                            errs.append(e)
                            colors_list.append(color)
                            j +=1
                            row_labels.append(label)


                elif('alphas' in args):
                    for alpha in args['alphas']:
                        label=' init r=' + str(alpha)
                        if(vd==va):
                            folder = 'log/CORNetZ_' + str(va)+'/'
                            vd = None
                            va = None
                        else:
                            folder = 'log/CORNetZ_switched/'
                        m, e = get_results(fig, ax, args, axins=axins, f_dir=folder, x=x, color=color, label=label, alpha=alpha, v_data = vd, v_area=va, acc=args['acc'])
                        max_avgs.append(m)
                        errs.append(e)
                        colors_list.append(get_color(alpha))
                        j += 1
                        row_labels.append(label)

    else:
        folder = 'log/CORNetZ_' + str(args['which'] + '/')
        if ('ratios' in args): 
            for ratio in args['ratios']:
                for n_e_v in args['n_e_vs']: 
                    label = 'r=' + str(ratio)
                    if 'colors' not in args:
                        color = get_color(ratio)
                    else:
                        color = args['colors'][j-1]
                    m, e = get_results(fig, ax,args, axins=axins, f_dir = folder, x=x, color=color, label=label, ratio=ratio, n_e_v = n_e_v, acc=args['acc'])
                    max_avgs.append(m)
                    errs.append(e)
                    colors_list.append(color)
                    j +=1
                    row_labels.append(label)

        elif('alphas' in args):
            for alpha in args['alphas']:
                label='init r=' + str(alpha)
                if 'colors' not in args:
                        color = get_color(alpha)
                else:
                    color = args['colors'][j-1]
                m, e = get_results(fig, ax, args, axins=axins, f_dir=folder, x=x, color=color, label=label, alpha=alpha, acc=args['acc'])
                max_avgs.append(m)
                errs.append(e)
                colors_list.append(get_color(alpha))
                j += 1
                row_labels.append(label)
        elif('initial_ratios' in args):
            for init_r in args['initial_ratios']:
                label = 'init r=' + str(init_r)
                if 'colors' not in args:
                        color = get_color(init_r)
                else:
                    color = args['colors'][j-1]
                m, e = get_results(fig, ax, args, axins=axins, n_e_v = args['n_e_vs'][0], f_dir=folder, x=x, color=color, label=label, initial_ratio = init_r, acc=args['acc'])
                max_avgs.append(m)
                errs.append(e)
                colors_list.append(get_color(init_r))
                j += 1
                row_labels.append(label)                   
    ####################################################################################
    if(args['plot_type']=='test_acc'):
        plt.legend()
        #plot_table(fig, ax, row_labels, max_avgs, errs, fa_mean, fa_error, colors=colors_list)
        if axins is not None:
            axins.set_xlim(args['x1'], args['x2']) # apply the x-limits
            axins.set_ylim(args['y1'], args['y2']) # apply the y-limits
            #axins.set_xlim(x1, x2) # apply the x-limits
            #axins.set_ylim(y1, y2) # apply the y-limits
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            plt.yticks(visible=False)
            plt.xticks(visible=False)

    if(args['plot_type']=='scatter'):
        plt.xticks(np.arange(len(max_avgs)), row_labels)
        plt.xticks(rotation=20)
        plt.gcf().subplots_adjust(bottom=0.15)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'black']
        plt.scatter(np.arange(len(max_avgs)), max_avgs, color=colors_list)
        plt.errorbar(np.arange(len(max_avgs)), max_avgs, yerr=errs, color ='black', alpha=0.5, ecolor = colors_list )
        #plt.plot(max_avgs, color='black', alpha=0.5)

    plt.legend()
    if(args['save']):
        f = 'plots/' + str(args['title']) + args['tadd'] + '.png'
        print('saving at ' + str(f))
        plt.savefig(f, format='png', dpi=1000)
    if(args['show']):
        plt.show()


    
