from setuptools import setup, find_packages

# with open("README.md", "r") as f:
# 	long_description = f.read()

with open("requirements.txt", "r") as f:
	requirements = f.read().strip().split("\n")

setup(
	name="None",
	description="Pata Nahi",
	classifiers=[
		"Development Status :: 2 - Pre-Alpha",
		"Environment :: Console",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Programming Language :: Python :: 3.7",
		"Topic :: Text Processing :: Linguistic",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
	],
	author="KV Aditya Srivatsa <k.v.aditya@research.iiit.ac.in>, Monil Gokani <monil.gokani@research.iiit.ac.in>",
	install_requires=requirements,
)