from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='sltcls',
      version='0.1',
      description='Short/Long Text Classifier using clustering-based enrichment',
      long_description=readme(),
      keywords='short text classification',
      url='https://github.com/bagheria/sltcls',
      author='Ayoub Bagheri',
      author_email='ayoub.bagheri@gmail.com',
      license='MIT',
      packages=['sltcls'],
      include_package_data=True,
      zip_safe=False)
