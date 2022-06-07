import splitfolders

input_folder = 'Dataset To be splitted'

splitfolders.ratio(input_folder, output="Splitted Dataset", seed=1337, ratio=(.6, .2, .2), group_prefix=None)
