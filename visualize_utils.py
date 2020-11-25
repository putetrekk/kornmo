
import os
from datetime import date, datetime

def make_image_path(image_description):
	"""
	Generates the image path. Makes a new folder each day. Will not overwrite existing images when doing experiments.
	example use: plt.savefig(make_image_path(f'sammenlikne_levert_med_andre_sorter_{farmer["orgnr"]}'), format='png')
	"""
	today = date.today().strftime("%b-%d-%Y")

	if not os.path.exists(f'screens/{today}'):
		os.makedirs(f'screens/{today}')

	current_time = datetime.now().strftime("%H-%M")
	return f'screens/{today}/{image_description}_{current_time}.png'
