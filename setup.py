from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='saltclass',
      version='0.2.1',
      description='Short and Long Text Classifier using clustering-based enrichment',
      long_description=readme(),
      long_description_content_type='text/markdown',
      keywords='short text classification',
      url='https://github.com/bagheria/saltclass',
      author='Ayoub Bagheri',
      author_email='ayoub.bagheri@gmail.com',
      license='MIT',
      packages=['saltclass'],
      include_package_data=True,
      zip_safe=False)
