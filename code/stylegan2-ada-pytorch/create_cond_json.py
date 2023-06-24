import os
import json


def main(input_folder='/home/kirill/code/diploma/code/stylegan2-ada-pytorch/datasets/processed/', 
	 	 output_folder='/home/kirill/code/diploma/code/stylegan2-ada-pytorch/'):
	
	"""
	Make json file with classes for conditional gan
	"""
	
	data_dict = {}
	data_dict['labels'] = []
	label_counter = -1

	with open(os.path.join(output_folder, 'dataset.json'), 'w') as outfile:

		for root, _, files in os.walk(input_folder):
			current_subdir = os.path.split(root)[1]
			
			for filename in files:
				file_path = os.path.join(current_subdir, filename)
				data_dict['labels'].append([file_path, label_counter])
				
			label_counter += 1

		json.dump(data_dict, outfile)


if __name__ == "__main__":
	main()


	