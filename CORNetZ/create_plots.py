from plot_helper import *


############ V1 n_e_v = 10 ############ 
'''
args = plot_args(which='V1', plot_type='test_acc')
#args['plot'] = 'test' 
args['n_e_vs'] = [10]
args['ratios'] = [0.01, 1.0, 2.0, 3.0, 4.0]
#args['acc'] = 'coarse_acc'
args['title'] = 'V1 n_e_v=10_fine'
args['zoom'] = False
#args['acc'] = 'coarse_acc'
'''
############ V1 n_e_v = 100 ############

#args = plot_args(which='V1', plot_type='scatter')
args = plot_args(which='V1', plot_type='test_acc')

args['n_e_vs'] = [100]
args['ratios'] = [ .01, .1, 1.0]
args['zoom'] = False
args['title'] = 'V1 n_e_v=100 scatter'
#args['title'] = 'V1 n_e_v=100'

############ V1 n_e_v = 100  static ############
'''
args = plot_args(which='V1', plot_type = 'test_acc')
args['initial_ratios'] = [.001, 0.1, 1.0]
args['n_e_vs'] = [100]
args['zoom'] = False
args['title'] = 'V1 static'

'''

############ V1 n_e_v = 10 random ############
'''
args = plot_args(which='switched', plot_type='scatter')
#args = plot_args(which='switched', plot_type='test_acc')
args['v_datas'] = ['V1', 'shuffled', 'random','randomnonV1']
args['v_areas'] = ['V1']
args['ratios'] = [0.1]
args['acc'] = 'coarse_acc'
args['colors'] = ['#800000', 'red', 'purple', 'blue']
args['y1'] =.48
args['y2'] = .50
#args['zoom_n'] = 5
args['zoom'] = False
args['title'] = 'V1 random'
#args['acc'] = 'coarse_acc'
'''


############# alpha ############
'''
args = plot_args(which='V1', plot_type='scatter')
#args = plot_args(which='V1', plot_type='test_acc')
args['alphas']= [0.01, 0.1, 1.0]

args['title']='alpha_v1_scatter'
#args['title']='alpha_testacc'
#args['save'] = True

'''
############ area=V4 data=V1 n_e_v = 100 r=0.1  ############
'''
args = plot_args(which='switched', plot_type='test_acc')
args['v_datas'] = ['V1']
args['v_areas'] = ['V4', 'IT']
args['n_e_vs'] = [100]
args['ratios'] = [0.1]
args['zoom'] = False 
args['title'] = 'area=V4 data=V1'
'''
############ V4  ############
'''
args = plot_args(which='IT', plot_type='test_acc')
args['n_e_vs'] = [100]
args['n_e_cs'] = [50]
args['ratios'] = [0.1, 0.01, 1.0]
args['zoom'] = False
args['title'] = 'V4 n_e_c=50 n_e_v=100'
'''
############ IT  ############
'''
args = plot_args(which='IT')
args['n_e_vs'] = [100]
args['n_e_cs'] = [50]
args['ratios'] = [0.1]
args['title'] = 'IT n_e_c=50 n_e_v=100'
'''

#args['save'] = True
#args['f_skip'] = 'resultsCORNetZ_V1ratio=1.0num_epochs=100n_e_c=0n_e_v=10_-0.0365.csv'
get_plot(args)

