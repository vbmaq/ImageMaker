import threading


class TPlotter(threading.Thread):
	def __init__(self, savefile, fig):
		threading.Thread.__init__(self)
		self.fig = fig
		self.savefile = savefile

	def run(self):
		print(F"Activate thread: {self.savefile}")
		self.fig.savefig(self.savefile)

