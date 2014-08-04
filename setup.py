from setuptools import setup

__version__ = "0.1.9.5"

setup(name='disclosuregame',
        version=__version__,
        description='Simulator for the disclosure game between midwives and women.',
        entry_points={'console_scripts': ['disclosure-game=disclosuregame.run:main']},
        url='https://github.com/greenape/disclosure-game',
        author='Jonathan Gray',
        author_email='j.gray@soton.ac.uk',
        license='MPL',
        packages=['disclosuregame', 'disclosuregame.Agents', 'disclosuregame.Measures', 'disclosuregame.Games', 'disclosuregame.Util'],
        include_package_data=True,
        zip_safe=False
)

with open("disclosuregame/_version.py", "w") as fp:
        fp.write("__version__ = '%s'\n" % (__version__,))