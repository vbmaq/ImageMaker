import glob
import os
import sys

pathdir = ""

def main(datadir="Data"):
	for g in glob.glob(os.path.join(pathdir, f"{datadir}/*")):
		print(g)
		print("Train\t\t" + str(len(glob.glob(os.path.join(g, "train/0/*")))) + " " + str(len(glob.glob(os.path.join(g, "train/1/*")))))
		print("Validate\t" + str(len(glob.glob(os.path.join(g, "validation/0/*")))) + " " + str(len(glob.glob(os.path.join(g, "validation/1/*")))))
		print("Test\t\t" + str(len(glob.glob(os.path.join(g, "test/0/*")))) + " " + str(len(glob.glob(os.path.join(g, "test/1/*")))))


if __name__ == '__main__':
	try:
		main(sys.argv[1])
	except IndexError:
		main()