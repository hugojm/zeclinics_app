## original

#from cardio.preprocess import lifpreprocess
import cardio.preprocess as pre
import cardio.process as pro
import cardio.nnet_cardio as nn

path = '/Users/sylviadyballa/Documents/LIFs_ZC_test/20210212_p1_2_004.lif'
outpath = '/Users/sylviadyballa/Documents/LIFs_ZC_test/output'

### preprocess 
#result = pre.lifpreprocess(path, store=True, out_shape=(482,408),debug=True)
#print(type(result), result.shape)


# ### process
# masks_a,masks_v,frame_a,frame_v,metrics,video_path= pro.process_video(path, base_it=120, update_it=4, skip=1, memory_it=1, border_removal=20
#                                         , filter_signal=True, lowcut=(10), highcut=(350), fps=76, p_index=2, p_out_shape=(482,408)
#                                         , gen_video=False, video_name='output.webm', p_store=False, p_out_dir='output', debug=True)
# masks_a,masks_v,a,v,dict = pro.process_video(path, base_it=120, update_it=4, skip=1, memory_it=1, border_removal=20, filter_signal=True
#                     , lowcut=(10), highcut=(350), fps=76, p_index=2, p_out_shape=(482,408)
#                     , gen_video=False, video_name='output.webm', p_store=False, p_out_dir='output', debug=False)

frames = pre.lifpreprocess(path, out_shape = (482,408), debug=True)
masks_a, masks_v, a, v, dict, _ =pro.process_video(frames[:100],fps=76,debug=True, filter_signal=False,
                                                    gen_video=True, p_out_dir='output', video_name='vid.avi')



